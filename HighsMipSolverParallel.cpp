/**
 * @file    HighsMipSolverParallel.h
 * @brief   the parallel version of branch and bound trees
 *
 * @author  Jinyu Zhang, Shuo Wang
 * @date    2025-04-26
 */

#pragma once
#include <barrier>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include "mip/HighsMipSolver.h"

// Master-worker parallel MIP solver with static load balancing
class HighsParallelMipSolver : public HighsMipSolver {
 public:
  HighsParallelMipSolver(HighsCallback& callback, 
                         const HighsOptions& options,
                         const HighsLp& lp, 
                         const HighsSolution& solution,
                         bool submip = false, 
                         HighsInt submip_level = 0,
                         int num_workers = std::thread::hardware_concurrency())
      : HighsMipSolver(callback, options, lp, solution, submip, submip_level),
        numWorkers(num_workers),
        syncBarrier(num_workers + 1) {}  // +1 for master

  void run() override {
    // 1. ���ڵ㴮�����
    runRootNodeSolve();  // ��ȡ�ɵ����������㸴��

    // 2. ���з�֧����
    masterLoop();
  }

 private:
  // ���Ĳ��в���
  const int numWorkers;
  std::vector<std::thread> workers;
  std::barrier<> syncBarrier;
  std::mutex mtx;

  // ������С�ȫ�����½硢���Ž��
  HighsNodeQueue sharedQueue;
  double globalUpper;
  double globalLower;

  // �߳�ѭ��
  void masterLoop() {
    // ��ʼ���������½�
    globalUpper = mipdata_->upper_bound;
    globalLower = mipdata_->lower_bound;
    sharedQueue = mipdata_->nodequeue;

    // ���� worker
    for (int i = 0; i < numWorkers; ++i)
      workers.emplace_back(&HighsParallelMipSolver::workerLoop, this, i);

    // ��ѭ��
    while (!sharedQueue.empty() && !mipdata_->checkLimits()) {
      // 1) ѡ����� numWorkers ���ڵ�
      std::vector<HighsNodeQueue::OpenNode> batch =
          selectBatch(sharedQueue, numWorkers);

      // 2) �� batch ���ȷ������ worker (��̬��ȷ����)
      dispatchBatch(batch);

      // 3) barrier �ȴ����� worker ���
      syncBarrier.arrive_and_wait();

      // 4) �ռ��� worker �Ľ��
      collectResults(batch);

      // 5) ȫ���򴫲�����֦�����½����
      updateGlobalState();

      // 6) ׼����һ��
    }

    // ֪ͨ worker �˳�
    dispatchEmptyBatch();
    syncBarrier.arrive_and_wait();

    // �ȴ����� worker �˳�
    for (auto& t : workers) t.join();

    // ���� cleanup
    cleanupSolve();
  }

  void workerLoop(int workerId) {
    while (true) {
      // �ȴ� master ��������
      syncBarrier.arrive_and_wait();

      // ���ղ��� batch
      auto myBatch = receiveBatch(workerId);
      if (myBatch.empty()) break;  // ��ֹ�ź�

      // ���� dive
      for (auto& node : myBatch) {
        HighsSearch search{*this, mipdata_->pseudocost};
        search.setLpRelaxation(&mipdata_->lp);
        search.installNode(node);
        search.dive();  // ���ô��� dive
        // ����ֲ����ſ��н���½ڵ�
        saveLocalResults(workerId, search);
      }

      // ֪ͨ master �ռ�
      syncBarrier.arrive_and_wait();
    }
  }

  // --------- ��������ʾ�� ---------
  void runRootNodeSolve() {
    // �봮�д�����ͬ�ĸ��ڵ�����߼�
    analysis_.mipTimerStart(kMipClockEvaluateRootNode);
    mipdata_->evaluateRootNode();
    analysis_.mipTimerStop(kMipClockEvaluateRootNode);
  }

  std::vector<HighsNodeQueue::OpenNode> selectBatch(HighsNodeQueue& queue,
                                                    int k) {
    std::vector<HighsNodeQueue::OpenNode> batch;
    for (int i = 0; i < k && !queue.empty(); ++i)
      batch.push_back(queue.popBestBoundNode());
    return batch;
  }

  void dispatchBatch(const std::vector<HighsNodeQueue::OpenNode>& batch) {
    std::lock_guard<std::mutex> lock(mtx);
    // �� batch �� workerId = idx % numWorkers �����������
    for (int i = 0; i < (int)batch.size(); ++i)
      localBatches[i % numWorkers].push_back(batch[i]);
  }

  void dispatchEmptyBatch() {
    std::lock_guard<std::mutex> lock(mtx);
    for (int i = 0; i < numWorkers; ++i) localBatches[i].clear();
  }

  std::vector<HighsNodeQueue::OpenNode> receiveBatch(int workerId) {
    std::lock_guard<std::mutex> lock(mtx);
    return std::move(localBatches[workerId]);
  }

  void collectResults(const std::vector<HighsNodeQueue::OpenNode>& batch) {
    std::lock_guard<std::mutex> lock(mtx);
    // �������� worker �ı��ؽ�������� sharedQueue��globalUpper
    for (int wid = 0; wid < numWorkers; ++wid) {
      // �ϲ��²����Ľڵ�
      for (auto& n : localNewNodes[wid]) sharedQueue.push(n);
      // �������ſ��н�
      globalUpper = std::min(globalUpper, localBestSol[wid]);
    }
  }

  void updateGlobalState() {
    // ���� mipdata_ �е����½�� queue
    mipdata_->nodequeue = sharedQueue;
    mipdata_->upper_bound = globalUpper;
    mipdata_->lower_bound = sharedQueue.getBestLowerBound();
    mipdata_->domain.propagate();
    mipdata_->nodequeue.pruneInfeasibleNodes(mipdata_->domain,
                                             mipdata_->feastol);
  }

  // �洢�� worker ��ʱ����
  std::vector<std::vector<HighsNodeQueue::OpenNode>> localBatches =
      std::vector<std::vector<HighsNodeQueue::OpenNode>>(numWorkers);
  std::vector<std::vector<HighsNodeQueue::OpenNode>> localNewNodes =
      std::vector<std::vector<HighsNodeQueue::OpenNode>>(numWorkers);
  std::vector<double> localBestSol = std::vector<double>(numWorkers, kHighsInf);
};
