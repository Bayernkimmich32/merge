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
    // 1. 根节点串行求解
    runRootNodeSolve();  // 提取成单独函数方便复用

    // 2. 并行分支定界
    masterLoop();
  }

 private:
  // 核心并行参数
  const int numWorkers;
  std::vector<std::thread> workers;
  std::barrier<> syncBarrier;
  std::mutex mtx;

  // 共享队列、全局上下界、最优解等
  HighsNodeQueue sharedQueue;
  double globalUpper;
  double globalLower;

  // 线程循环
  void masterLoop() {
    // 初始化共享上下界
    globalUpper = mipdata_->upper_bound;
    globalLower = mipdata_->lower_bound;
    sharedQueue = mipdata_->nodequeue;

    // 创建 worker
    for (int i = 0; i < numWorkers; ++i)
      workers.emplace_back(&HighsParallelMipSolver::workerLoop, this, i);

    // 主循环
    while (!sharedQueue.empty() && !mipdata_->checkLimits()) {
      // 1) 选择最多 numWorkers 个节点
      std::vector<HighsNodeQueue::OpenNode> batch =
          selectBatch(sharedQueue, numWorkers);

      // 2) 将 batch 均匀分配给各 worker (静态、确定性)
      dispatchBatch(batch);

      // 3) barrier 等待所有 worker 完成
      syncBarrier.arrive_and_wait();

      // 4) 收集各 worker 的结果
      collectResults(batch);

      // 5) 全局域传播、剪枝、上下界更新
      updateGlobalState();

      // 6) 准备下一轮
    }

    // 通知 worker 退出
    dispatchEmptyBatch();
    syncBarrier.arrive_and_wait();

    // 等待所有 worker 退出
    for (auto& t : workers) t.join();

    // 串行 cleanup
    cleanupSolve();
  }

  void workerLoop(int workerId) {
    while (true) {
      // 等待 master 分配任务
      syncBarrier.arrive_and_wait();

      // 接收并行 batch
      auto myBatch = receiveBatch(workerId);
      if (myBatch.empty()) break;  // 终止信号

      // 本地 dive
      for (auto& node : myBatch) {
        HighsSearch search{*this, mipdata_->pseudocost};
        search.setLpRelaxation(&mipdata_->lp);
        search.installNode(node);
        search.dive();  // 复用串行 dive
        // 保存局部最优可行解和新节点
        saveLocalResults(workerId, search);
      }

      // 通知 master 收集
      syncBarrier.arrive_and_wait();
    }
  }

  // --------- 辅助函数示例 ---------
  void runRootNodeSolve() {
    // 与串行代码相同的根节点求解逻辑
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
    // 将 batch 按 workerId = idx % numWorkers 存入各自容器
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
    // 遍历所有 worker 的本地结果，更新 sharedQueue、globalUpper
    for (int wid = 0; wid < numWorkers; ++wid) {
      // 合并新产生的节点
      for (auto& n : localNewNodes[wid]) sharedQueue.push(n);
      // 更新最优可行解
      globalUpper = std::min(globalUpper, localBestSol[wid]);
    }
  }

  void updateGlobalState() {
    // 更新 mipdata_ 中的上下界和 queue
    mipdata_->nodequeue = sharedQueue;
    mipdata_->upper_bound = globalUpper;
    mipdata_->lower_bound = sharedQueue.getBestLowerBound();
    mipdata_->domain.propagate();
    mipdata_->nodequeue.pruneInfeasibleNodes(mipdata_->domain,
                                             mipdata_->feastol);
  }

  // 存储各 worker 临时数据
  std::vector<std::vector<HighsNodeQueue::OpenNode>> localBatches =
      std::vector<std::vector<HighsNodeQueue::OpenNode>>(numWorkers);
  std::vector<std::vector<HighsNodeQueue::OpenNode>> localNewNodes =
      std::vector<std::vector<HighsNodeQueue::OpenNode>>(numWorkers);
  std::vector<double> localBestSol = std::vector<double>(numWorkers, kHighsInf);
};
