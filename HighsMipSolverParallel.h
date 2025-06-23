/**
 * @file    HighsMipSolverParallel.h
 * @brief   the parallel version of branch and bound trees
 *
 * @author  Jinyu Zhang, Shuo Wang
 * @date    2025-04-26
 */

#pragma once

#include <barrier>
#include <mutex>
#include <thread>
#include <vector>

#include "mip/HighsMipSolver.h"
#include "mip/HighsNodeQueue.h"

// Master¨Cworker parallel MIP solver with static load balancing
class HighsParallelMipSolver : public HighsMipSolver {
 public:
  // Constructor: initializes base MIP solver and sets up worker count
  HighsParallelMipSolver(HighsCallback& callback, const HighsOptions& options,
                         const HighsLp& lp, const HighsSolution& solution,
                         bool submip = false, HighsInt submip_level = 0,
                         int num_workers = std::thread::hardware_concurrency());

   Override run: root node solve serially, then launch master¨Cworker loop
  void run() override;

 private:
  using OpenNode = HighsNodeQueue::OpenNode;

  // --- Core parallel workflow ---
  void runRootNodeSolve();
  void masterLoop();
  void workerLoop(int workerId);

  // Batch selection and dispatching
  std::vector<OpenNode> selectBatch(HighsNodeQueue& queue, int k);
  void dispatchBatch(const std::vector<OpenNode>& batch);
  void dispatchEmptyBatch();
  std::vector<OpenNode> receiveBatch(int workerId);

  // Result collection and global state update
  void collectResults(const std::vector<OpenNode>& batch);
  void updateGlobalState();

  // --- Parallel control and data structures ---
  const int numWorkers;
  std::vector<std::thread> workers;
  std::barrier<> syncBarrier;
  std::mutex mtx;

  // Shared queue and bounds
  HighsNodeQueue sharedQueue;
  double globalUpper;
  double globalLower;

  // Per-worker storage
  std::vector<std::vector<OpenNode>> localBatches;
  std::vector<std::vector<OpenNode>> localNewNodes;
  std::vector<double> localBestSol;
};
