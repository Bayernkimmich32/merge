/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef MIP_HIGHS_MIP_SOLVER_H_
#define MIP_HIGHS_MIP_SOLVER_H_

#include "Highs.h"
#include "lp_data/HighsCallback.h"
#include "lp_data/HighsOptions.h"
#include "mip/HighsMipAnalysis.h"
#include "mip/HighsNodeQueue.h"

// parallel
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

struct HighsMipSolverData;
class HighsCutPool;
struct HighsPseudocostInitialization;
class HighsCliqueTable;
class HighsImplications;
class HighsSearch;
class HighsMipWorker;

// parallel
struct NodeTask {
  HighsNodeQueue::OpenNode node;
  int worker_id;
};

class HighsMipSolver {
 public:
  HighsCallback* callback_;
  const HighsOptions* options_mip_;
  const HighsLp* model_;
  const HighsLp* orig_model_;
  HighsModelStatus modelstatus_;
  std::vector<double> solution_;
  double solution_objective_;
  double bound_violation_;
  double integrality_violation_;
  double row_violation_;
  // The following are only to return data to HiGHS, and are set in
  // HighsMipSolver::cleanupSolve
  double dual_bound_;
  double primal_bound_;
  double gap_;
  int64_t node_count_;
  int64_t total_lp_iterations_;
  double primal_dual_integral_;

  FILE* improving_solution_file_;
  std::vector<HighsObjectiveSolution> saved_objective_and_solution_;

  bool submip;
  HighsInt submip_level;
  HighsInt max_submip_level;
  const HighsBasis* rootbasis;
  const HighsPseudocostInitialization* pscostinit;
  const HighsCliqueTable* clqtableinit;
  const HighsImplications* implicinit;
  
  // parallel configuration
  int num_workers_ = 2;
  bool parallel_enabled_ = true;

  // parallel synchronization
  std::mutex mtx_;
  std::condition_variable cv_master_;
  std::vector<std::condition_variable> cv_workers_;
  std::atomic<int> ready_workers_{0};
  std::atomic<bool> master_notified_{false};

  std::vector<std::vector<HighsSearch*>> worker_loop_search_ptrs_; // 每个worker的search指针
  std::vector<HighsInt> worker_node_counts_; // 每个worker分配的节点数

  std::vector<HighsSearch*> loop_search_ptrs_;
  HighsInt loop_batch_size_ = 0;

  bool terminate_workers_ = false;
  int generation_ = 0;

  // parallel methods
  void runParallel();
  void workerLoop(int worker_id);

  std::unique_ptr<HighsMipSolverData> mipdata_;

  HighsMipAnalysis analysis_;

  void run();

  void runSerial();

  HighsInt numCol() const { return model_->num_col_; }

  HighsInt numRow() const { return model_->num_row_; }

  HighsInt numNonzero() const { return model_->a_matrix_.numNz(); }

  const double* colCost() const { return model_->col_cost_.data(); }

  double colCost(HighsInt col) const { return model_->col_cost_[col]; }

  const double* rowLower() const { return model_->row_lower_.data(); }

  double rowLower(HighsInt col) const { return model_->row_lower_[col]; }

  const double* rowUpper() const { return model_->row_upper_.data(); }

  double rowUpper(HighsInt col) const { return model_->row_upper_[col]; }

  const HighsVarType* variableType() const {
    return model_->integrality_.data();
  }

  HighsVarType variableType(HighsInt col) const {
    return model_->integrality_[col];
  }

  // Contstructor from reference.
  HighsMipSolver(const HighsMipSolver& mip_solver_);

  // Constructor without solution.
  HighsMipSolver(HighsCallback* callback, const HighsOptions* options,
                 const HighsLp* lp);

  HighsMipSolver(HighsCallback& callback, const HighsOptions& options,
                 const HighsLp& lp, const HighsSolution& solution,
                 bool submip = false, HighsInt submip_level = 0);

  ~HighsMipSolver();

  void setModel(const HighsLp& model) {
    model_ = &model;
    solution_objective_ = kHighsInf;
  }

  mutable HighsTimer timer_;
  void cleanupSolve();

  void runPresolve(const HighsInt presolve_reduction_limit);
  const HighsLp& getPresolvedModel() const;
  HighsPresolveStatus getPresolveStatus() const;
  presolve::HighsPostsolveStack getPostsolveStack() const;

  void callbackGetCutPool() const;
  bool solutionFeasible(const HighsLp* lp, const std::vector<double>& col_value,
                        const std::vector<double>* pass_row_value,
                        double& bound_violation, double& row_violation,
                        double& integrality_violation, HighsCDouble& obj);
  //private:
  //    //// 新增：预分配节点到workers的函数
  //    //void allocateNodesToWorkers(HighsNodeQueue& best_nodes);

  //    // 新增：确定最佳节点数量的辅助函数
  //    HighsInt determineBestNodesCount();
};

#endif
