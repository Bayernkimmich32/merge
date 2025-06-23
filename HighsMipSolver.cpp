/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsMipSolver.h"

#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsMipWorker.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsSearch.h"
#include "mip/HighsSeparation.h"
#include "mip/MipTimer.h"
#include "presolve/HPresolve.h"
#include "presolve/HighsPostsolveStack.h"
#include "presolve/PresolveComponent.h"
#include "util/HighsCDouble.h"
#include "util/HighsIntegers.h"

#include "global_mutex.h"

using std::fabs;
#include <windows.h>
#include <processthreadsapi.h>

// parallel
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <random>
#include <cstdlib>

//static std::recursive_mutex g_mutex;

HighsMipSolver::HighsMipSolver(const HighsMipSolver& mip_solver_)
    : HighsMipSolver(mip_solver_.callback_, 
                     mip_solver_.options_mip_,
                     mip_solver_.model_) {}

HighsMipSolver::HighsMipSolver(HighsCallback* callback,
                               const HighsOptions* options, 
                               const HighsLp* lp)
    : callback_(callback),
      options_mip_(options),
      model_(lp),
      orig_model_(lp),
      solution_objective_(kHighsInf),
      submip(false),
      submip_level(0),
      rootbasis(nullptr),
      pscostinit(nullptr),
      clqtableinit(nullptr),
      implicinit(nullptr) {}

HighsMipSolver::HighsMipSolver(HighsCallback& callback,
                               const HighsOptions& options, const HighsLp& lp,
                               const HighsSolution& solution, bool submip,
                               HighsInt submip_level)
    : callback_(&callback),
      options_mip_(&options),
      model_(&lp),
      orig_model_(&lp),
      solution_objective_(kHighsInf),
      submip(submip),
      submip_level(submip_level),
      rootbasis(nullptr),
      pscostinit(nullptr),
      clqtableinit(nullptr),
      implicinit(nullptr),
      cv_workers_(num_workers_) {
  assert(!submip || submip_level > 0);
  max_submip_level = 0;
  if (solution.value_valid) {
    // MIP solver doesn't check row residuals, but they should be OK
    // so validate using assert
#ifndef NDEBUG
    bool valid, integral, feasible;
    assessLpPrimalSolution("For debugging: ", options, lp, solution, valid,
                           integral, feasible);
    assert(valid);
#endif
    // Initial solution can be infeasible, but need to set values for violation
    // and objective
    HighsCDouble quad_solution_objective_;
    solutionFeasible(orig_model_, solution.col_value, &solution.row_value,
                     bound_violation_, row_violation_, integrality_violation_,
                     quad_solution_objective_);
    solution_objective_ = double(quad_solution_objective_);
    solution_ = solution.col_value;
  }
}

HighsMipSolver::~HighsMipSolver() = default;

void HighsMipSolver::run() {
  if (parallel_enabled_ && !submip) {
    return runParallel();
  } else {
    return runSerial();
  }
}

void HighsMipSolver::runParallel() {
    /* solve root node */
    modelstatus_ = HighsModelStatus::kNotset;

    if (submip) {
        analysis_.analyse_mip_time = false;
    }
    else {
        analysis_.timer_ = &this->timer_;
        analysis_.setup(*orig_model_, *options_mip_);
    }
    timer_.start();

    improving_solution_file_ = nullptr;
    if (!submip && options_mip_->mip_improving_solution_file != "")
        improving_solution_file_ =
        fopen(options_mip_->mip_improving_solution_file.c_str(), "w");

    mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
    analysis_.mipTimerStart(kMipClockPresolve);
    analysis_.mipTimerStart(kMipClockInit);
    mipdata_->init();
    analysis_.mipTimerStop(kMipClockInit);
    analysis_.mipTimerStart(kMipClockRunPresolve);
    mipdata_->runPresolve(options_mip_->presolve_reduction_limit);
    analysis_.mipTimerStop(kMipClockRunPresolve);
    analysis_.mipTimerStop(kMipClockPresolve);
    if (analysis_.analyse_mip_time && !submip)
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
            "MIP-Timing: %11.2g - completed presolve\n", timer_.read());
    // Identify whether time limit has been reached (in presolve)
    if (modelstatus_ == HighsModelStatus::kNotset &&
        timer_.read() >= options_mip_->time_limit)
        modelstatus_ = HighsModelStatus::kTimeLimit;

    if (modelstatus_ != HighsModelStatus::kNotset) {
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
            "Presolve: %s\n",
            utilModelStatusToString(modelstatus_).c_str());
        if (modelstatus_ == HighsModelStatus::kOptimal) {
            mipdata_->lower_bound = 0;
            mipdata_->upper_bound = 0;
            mipdata_->transformNewIntegerFeasibleSolution(std::vector<double>());
            mipdata_->saveReportMipSolution();
        }
        cleanupSolve();
        return;
    }

    analysis_.mipTimerStart(kMipClockSolve);

    if (analysis_.analyse_mip_time && !submip)
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
            "MIP-Timing: %11.2g - starting  setup\n", timer_.read());
    analysis_.mipTimerStart(kMipClockRunSetup);
    mipdata_->runSetup();
    analysis_.mipTimerStop(kMipClockRunSetup);
    if (analysis_.analyse_mip_time && !submip)
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
            "MIP-Timing: %11.2g - completed setup\n", timer_.read());

    // initialize the parallel infomation
    // terminate_workers_ = false;
    // ready_workers_ = 0;
    // generation_ = 0;
    // loop_batch_size_ = 0;
    // loop_search_ptrs_.clear();
    // cv_workers_.clear();
    // cv_workers_.resize(num_workers_);

    /* initilize worker threads */
    // cv_workers_.resize(num_workers_);
    std::vector<std::thread> workers;
    workers.reserve(num_workers_);
    for (int i = 0; i < num_workers_; ++i)
        workers.emplace_back(&HighsMipSolver::workerLoop, this, i);

restart:
    if (modelstatus_ == HighsModelStatus::kNotset) {
        // Check limits have not been reached before evaluating root node
        if (mipdata_->checkLimits()) {
            cleanupSolve();
            return;
        }
        // Possibly look for primal solution from the user
        if (!submip && callback_->user_callback &&
            callback_->active[kCallbackMipUserSolution])
            mipdata_->callbackUserSolution(solution_objective_,
                kUserMipSolutionCallbackOriginAfterSetup);

        // Apply the trivial heuristics
        analysis_.mipTimerStart(kMipClockTrivialHeuristics);
        HighsModelStatus model_status = mipdata_->trivialHeuristics();
        analysis_.mipTimerStop(kMipClockTrivialHeuristics);
        if (modelstatus_ == HighsModelStatus::kNotset &&
            model_status == HighsModelStatus::kInfeasible) {
            // trivialHeuristics can spot trivial infeasibility, so act on it
            modelstatus_ = model_status;
            cleanupSolve();
            return;
        }
        if (analysis_.analyse_mip_time && !submip)
            highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                "MIP-Timing: %11.2g - starting evaluate root node\n",
                timer_.read());
        analysis_.mipTimerStart(kMipClockEvaluateRootNode);
        mipdata_->evaluateRootNode();
        analysis_.mipTimerStop(kMipClockEvaluateRootNode);
        // Sometimes the analytic centre calculation is not completed when
        // evaluateRootNode returns, so stop its clock if it's running
        if (analysis_.analyse_mip_time &&
            analysis_.mipTimerRunning(kMipClockIpmSolveLp))
            analysis_.mipTimerStop(kMipClockIpmSolveLp);
        if (analysis_.analyse_mip_time && !submip)
            highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                "MIP-Timing: %11.2g - completed evaluate root node\n",
                timer_.read());
        // age 5 times to remove stored but never violated cuts after root
        // separation
        analysis_.mipTimerStart(kMipClockPerformAging0);
        mipdata_->cutpool.performAging();
        mipdata_->cutpool.performAging();
        mipdata_->cutpool.performAging();
        mipdata_->cutpool.performAging();
        mipdata_->cutpool.performAging();
        analysis_.mipTimerStop(kMipClockPerformAging0);
    }
    if (mipdata_->nodequeue.empty() || mipdata_->checkLimits()) {
        cleanupSolve();
        return;
    }

    std::shared_ptr<const HighsBasis> basis;
    HighsSearch master_search{ *this, mipdata_->pseudocost };
    mipdata_->debugSolution.registerDomain(master_search.getLocalDomain());
    HighsSeparation sepa(*this);

    // master_search.setLpRelaxation(&mipdata_->lp);
    sepa.setLpRelaxation(&mipdata_->lp);

    double prev_lower_bound = mipdata_->lower_bound;

    mipdata_->lower_bound = mipdata_->nodequeue.getBestLowerBound();

    bool bound_change = mipdata_->lower_bound != prev_lower_bound;
    if (!submip && bound_change)
        mipdata_->updatePrimalDualIntegral(prev_lower_bound, mipdata_->lower_bound,
            mipdata_->upper_bound,
            mipdata_->upper_bound);

    mipdata_->printDisplayLine();

    /* Master coordination loop */
    // Assumes mipdata_->nodequeue populated after root
    int64_t numStallNodes = 0;
    int64_t lastLbLeave = 0;
    int64_t numQueueLeaves = 0;
    HighsInt numHugeTreeEstim = 0;
    int64_t numNodesLastCheck = mipdata_->num_nodes;
    int64_t nextCheck = mipdata_->num_nodes;
    double treeweightLastCheck = 0.0;
    double upperLimLastCheck = mipdata_->upper_limit;
    double lowerBoundLastCheck = mipdata_->lower_bound;

    HighsNodeQueue best_nodes;
    best_nodes.setNumCol(this->numCol());
    master_search.installNode(mipdata_->nodequeue.popBestBoundNode());
    master_search.openNodesToQueue(best_nodes);

    // 新的数据结构：worker到nodes的映射
    std::vector<std::vector<HighsNodeQueue::OpenNode>> worker_nodes(num_workers_);
    // 记录每个worker分配的节点数 s_i
    std::vector<HighsInt> workers_node_counts(num_workers_, 0);

    analysis_.mipTimerStart(kMipClockSearch);
    while (true) {
        // basic variables
        bool limit_reached = false;
        bool considerHeuristics = true;

        // num of workers to be used in this iteration
        // HighsInt limit = std::min(num_workers_,
        // static_cast<HighsInt>(best_nodes.size()));


        // Delete:移除原有节点选择数量限制
        /*HighsInt limit =
            std::min(num_workers_, static_cast<HighsInt>(best_nodes.numNodes()));*/

            // 清空worker节点分配
        for (auto& nodes : worker_nodes) {
            nodes.clear();
        }
        std::fill(workers_node_counts.begin(), workers_node_counts.end(), 0);

        // 调用新算法确定best_nodes中的节点数量m
        //best_nodes.clear();
        //allocateNodesToWorkers(best_nodes);
        HighsInt m = best_nodes.numNodes();

        // Delete:删除之前的search初始化部分
        //// initialize the mipworker
        //auto mipworkers_ptr = std::make_unique<std::vector<HighsMipWorker>>();
        //auto& mipworkers = *mipworkers_ptr;
        //mipworkers.reserve(limit);
        //for (HighsInt i = 0; i < limit; ++i) {
        //  // HighsMipSolver& worker_mipsolver = worker_mipsolvers[i];
        //  mipworkers.emplace_back(*this);  // *this, worker_mipsolver
        //}

        // 使用简单的循环分配策略分配节点
        HighsInt node_idx = 0;
        while (best_nodes.numNodes() > 0 && node_idx < m) {
            HighsInt worker_id = node_idx % num_workers_;
            worker_nodes[worker_id].push_back(best_nodes.popBestNode());
            workers_node_counts[worker_id]++;
            node_idx++;
        }

        // 初始化每个worker的searches
        std::vector<std::vector<HighsSearch>> worker_searches(num_workers_);
        std::vector<std::vector<HighsLpRelaxation>> worker_lprelaxations(num_workers_);
        std::vector<std::vector<HighsSearch*>> worker_search_ptrs(num_workers_);

        // 为每个worker创建所需数量的search
        for (HighsInt worker_id = 0; worker_id < num_workers_; ++worker_id) {
            HighsInt s_i = workers_node_counts[worker_id];
            if (s_i == 0) continue;

            worker_searches[worker_id].reserve(s_i);
            worker_lprelaxations[worker_id].reserve(s_i);
            worker_search_ptrs[worker_id].reserve(s_i);

            for (HighsInt j = 0; j < s_i; ++j) {
                // 创建search，j+1表示该节点在best_nodes序列中的序号
                worker_searches[worker_id].emplace_back(*this, mipdata_->pseudocost);
                worker_lprelaxations[worker_id].emplace_back(mipdata_->lp);

                // 设置lprelaxation
                worker_searches[worker_id][j].setLpRelaxation(&worker_lprelaxations[worker_id][j]);
                worker_searches[worker_id][j].resetLocalDomain();

                // 安装节点，节点编号为j+1，分配给worker_id+1
                worker_searches[worker_id][j].installNode(std::move(worker_nodes[worker_id][j]));

                // 保存search指针
                worker_search_ptrs[worker_id].push_back(&worker_searches[worker_id][j]);
            }
        }

        //// initialize the lprelaxations from the workers
        //std::vector<HighsLpRelaxation> worker_lprelaxations;
        //worker_lprelaxations.reserve(limit);
        //for (HighsInt i = 0; i < limit; ++i) {
        //  // HighsMipSolver& worker_mipsolver = worker_mipsolvers[i];
        //  worker_lprelaxations.emplace_back(mipdata_->lp);
        //  // mipdata_->lp  worker_mipsolver.mipdata_->lp
        //}

        //// set the lprelaxations to the workers
        //for (HighsInt i = 0; i < limit; ++i) {
        //  concurrent_searches[i]->setLpRelaxation(&worker_lprelaxations[i]);
        //  concurrent_searches[i]->resetLocalDomain();
        //  // concurrent_searches[i]->installNode(std::move(best_nodes[i]));
        //  HighsNodeQueue::OpenNode node = best_nodes.popBestNode();
        //  concurrent_searches[i]->installNode(std::move(node));
        //}


        // send the tasks to workers
        int my_gen;
        {
            std::lock_guard<std::mutex> lg(mtx_);
            generation_++;
            my_gen = generation_;
            // 将worker_search_ptrs传递给workerLoop
            worker_loop_search_ptrs_ = worker_search_ptrs;
            worker_node_counts_ = workers_node_counts;
            /*loop_search_ptrs_ = concurrent_searches;
            loop_batch_size_ = limit;*/
            ready_workers_ = 0;
        }

        for (int i = 0; i < num_workers_; ++i) {
            //for (int i = 0; i < limit; ++i) {
              // std::this_thread::sleep_for(std::chrono::milliseconds(50));
            cv_workers_[i].notify_one();
        }


        // feature

        // wait for all workers to finish
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_master_.wait(lk, [&]() {
                return generation_ == my_gen && ready_workers_ == num_workers_;
                });
        }

        // label

        // update the global infomation
        analysis_.mipTimerStart(kMipClockOpenNodesToQueue0);
        /*for (HighsInt i = 0; i < limit; ++i) {
          HighsSearch* worker_search = concurrent_searches[i];
          worker_search->openNodesToQueue(mipdata_->nodequeue);
          worker_search->flushStatistics();
          assert(!worker_search->hasNode());
        }*/
        for (HighsInt worker_id = 0; worker_id < num_workers_; ++worker_id) {
            HighsInt s_i = workers_node_counts[worker_id];
            for (HighsInt j = 0; j < s_i; ++j) {
                HighsSearch* worker_search = worker_search_ptrs[worker_id][j];
                worker_search->openNodesToQueue(mipdata_->nodequeue);
                worker_search->flushStatistics();
                assert(!worker_search->hasNode());
            }
        }
        analysis_.mipTimerStop(kMipClockOpenNodesToQueue0);

        // propagate the global domain
        analysis_.mipTimerStart(kMipClockDomainPropgate);
        mipdata_->domain.propagate();
        analysis_.mipTimerStop(kMipClockDomainPropgate);

        analysis_.mipTimerStart(kMipClockPruneInfeasibleNodes);
        mipdata_->pruned_treeweight += mipdata_->nodequeue.pruneInfeasibleNodes(
            mipdata_->domain, mipdata_->feastol);
        analysis_.mipTimerStop(kMipClockPruneInfeasibleNodes);

        // if global propagation detected infeasibility, stop here
        if (mipdata_->domain.infeasible()) {
            mipdata_->nodequeue.clear();
            mipdata_->pruned_treeweight = 1.0;

            double prev_lower_bound = mipdata_->lower_bound;

            mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);

            bool bound_change = mipdata_->lower_bound != prev_lower_bound;
            if (!submip && bound_change)
                mipdata_->updatePrimalDualIntegral(
                    prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
                    mipdata_->upper_bound);
            mipdata_->printDisplayLine();
            break;
        }

        double prev_lower_bound = mipdata_->lower_bound;

        mipdata_->lower_bound = std::min(mipdata_->upper_bound,
            mipdata_->nodequeue.getBestLowerBound());
        bool bound_change = mipdata_->lower_bound != prev_lower_bound;
        if (!submip && bound_change)
            mipdata_->updatePrimalDualIntegral(
                prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
                mipdata_->upper_bound);
        mipdata_->printDisplayLine();
        if (mipdata_->nodequeue.empty()) break;

        //mipdata_->cutpool.performAging();
        //mipdata_->conflictPool.performAging();

        // if global propagation found bound changes, we update the local domain
        if (!mipdata_->domain.getChangedCols().empty()) {
            analysis_.mipTimerStart(kMipClockUpdateLocalDomain);
            highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                "added %" HIGHSINT_FORMAT " global bound changes\n",
                (HighsInt)mipdata_->domain.getChangedCols().size());
            mipdata_->cliquetable.cleanupFixed(mipdata_->domain);
            for (HighsInt col : mipdata_->domain.getChangedCols())
                mipdata_->implications.cleanupVarbounds(col);

            /*mipdata_->domain.setDomainChangeStack(std::vector<HighsDomainChange>());
            for (HighsInt i = 0; i < limit; ++i) {
              HighsSearch* worker_search = concurrent_searches[i];
              worker_search->resetLocalDomain();
            }*/
            mipdata_->domain.setDomainChangeStack(std::vector<HighsDomainChange>());
            for (HighsInt worker_id = 0; worker_id < num_workers_; ++worker_id) {
                HighsInt s_i = workers_node_counts[worker_id];
                for (HighsInt j = 0; j < s_i; ++j) {
                    HighsSearch* worker_search = worker_search_ptrs[worker_id][j];
                    worker_search->resetLocalDomain();
                }
            }

            mipdata_->domain.clearChangedCols();
            mipdata_->removeFixedIndices();
            analysis_.mipTimerStop(kMipClockUpdateLocalDomain);
        }

        if (!submip && mipdata_->num_nodes >= nextCheck) {
            auto nTreeRestarts = mipdata_->numRestarts - mipdata_->numRestartsRoot;
            double currNodeEstim =
                numNodesLastCheck - mipdata_->num_nodes_before_run +
                (mipdata_->num_nodes - numNodesLastCheck) *
                double(1.0 - mipdata_->pruned_treeweight) /
                std::max(
                    double(mipdata_->pruned_treeweight - treeweightLastCheck),
                    mipdata_->epsilon);

            bool doRestart = false;

            double activeIntegerRatio =
                1.0 - mipdata_->percentageInactiveIntegers() / 100.0;
            activeIntegerRatio *= activeIntegerRatio;

            if (!doRestart) {
                double gapReduction = 1.0;
                if (mipdata_->upper_limit != kHighsInf) {
                    double oldGap = upperLimLastCheck - lowerBoundLastCheck;
                    double newGap = mipdata_->upper_limit - mipdata_->lower_bound;
                    gapReduction = oldGap / newGap;
                }

                if (gapReduction < 1.0 + (0.05 / activeIntegerRatio) &&
                    currNodeEstim >=
                    activeIntegerRatio * 20 *
                    (mipdata_->num_nodes - mipdata_->num_nodes_before_run)) {
                    nextCheck = mipdata_->num_nodes + 100;
                    ++numHugeTreeEstim;
                }
                else {
                    numHugeTreeEstim = 0;
                    treeweightLastCheck = double(mipdata_->pruned_treeweight);
                    numNodesLastCheck = mipdata_->num_nodes;
                    upperLimLastCheck = mipdata_->upper_limit;
                    lowerBoundLastCheck = mipdata_->lower_bound;
                }

                // Possibly prevent restart - necessary for debugging presolve
                // errors: see #1553
                if (options_mip_->mip_allow_restart) {
                    int64_t minHugeTreeOffset =
                        (mipdata_->num_leaves - mipdata_->num_leaves_before_run) / 1000;
                    int64_t minHugeTreeEstim = HighsIntegers::nearestInteger(
                        activeIntegerRatio * (10 + minHugeTreeOffset) *
                        std::pow(1.5, nTreeRestarts));

                    doRestart = numHugeTreeEstim >= minHugeTreeEstim;
                }
                else {
                    doRestart = false;
                }
            }
            else {
                // count restart due to many fixings within the first 1000 nodes as
                // root restart
                ++mipdata_->numRestartsRoot;
            }

            if (doRestart) {
                highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                    "\nRestarting search from the root node\n");
                mipdata_->performRestart();
                analysis_.mipTimerStop(kMipClockSearch);
                goto restart;
            }
        }  // if (!submip && mipdata_->num_nodes >= nextCheck))

        // remove the iteration limit when installing a new node
        mipdata_->lp.setIterationLimit();

        best_nodes.clear();
        assert(!best_nodes.numNodes());
        HighsSearch* search = &master_search;  // 使用主search评估节点
        analysis_.mipTimerStart(kMipClockNodeSearch);

        // 预分配结构
        struct NodeWithTime {
            HighsNodeQueue::OpenNode node;
            int64_t predicted_time;
            HighsInt best_nodes_index;

            NodeWithTime(NodeWithTime&& other) noexcept
                : node(std::move(other.node)),
                predicted_time(other.predicted_time),
                best_nodes_index(other.best_nodes_index) {
            }

            NodeWithTime& operator=(NodeWithTime&& other) noexcept {
                if (this != &other) {
                    node = std::move(other.node);
                    predicted_time = other.predicted_time;
                    best_nodes_index = other.best_nodes_index;
                }
                return *this;
            }

            NodeWithTime(const NodeWithTime&) = delete;
            NodeWithTime& operator=(const NodeWithTime&) = delete;

            NodeWithTime(HighsNodeQueue::OpenNode&& n, int64_t time, HighsInt index)
                : node(std::move(n)), predicted_time(time), best_nodes_index(index) {
            }
        };

        // 初始化预分配结构
        std::vector<std::vector<NodeWithTime>> prealloc_queues(num_workers_);
        std::vector<int64_t> queue_times(num_workers_, 0);
        HighsInt best_nodes_idx = 0;
        bool stop_selecting = false;

        while (!mipdata_->nodequeue.empty() && !stop_selecting) {
            assert(!search->hasNode());

            // 安装节点
            if (numQueueLeaves - lastLbLeave >= 10) {
                search->installNode(mipdata_->nodequeue.popBestBoundNode());
                lastLbLeave = numQueueLeaves;
            }
            else {
                HighsInt bestBoundNodeStackSize =
                    mipdata_->nodequeue.getBestBoundDomchgStackSize();
                double bestBoundNodeLb = mipdata_->nodequeue.getBestLowerBound();
                HighsNodeQueue::OpenNode nextNode = mipdata_->nodequeue.popBestNode();
                if (nextNode.lower_bound == bestBoundNodeLb &&
                    (HighsInt)nextNode.domchgstack.size() == bestBoundNodeStackSize)
                    lastLbLeave = numQueueLeaves;
                search->installNode(std::move(nextNode));
            }
            ++numQueueLeaves;

            if (search->getCurrentEstimate() >= mipdata_->upper_limit) {
                ++numStallNodes;
                if (options_mip_->mip_max_stall_nodes != kHighsIInf &&
                    numStallNodes >= options_mip_->mip_max_stall_nodes) {
                    limit_reached = true;
                    modelstatus_ = HighsModelStatus::kSolutionLimit;
                    break;
                }
            }
            else {
                numStallNodes = 0;
            }

            analysis_.mipTimerStart(kMipClockEvaluateNode1);
            const HighsSearch::NodeResult evaluate_node_result = search->evaluateNode();
            analysis_.mipTimerStop(kMipClockEvaluateNode1);
            if (evaluate_node_result == HighsSearch::NodeResult::kSubOptimal) {
                analysis_.mipTimerStart(kMipClockCurrentNodeToQueue);
                search->currentNodeToQueue(mipdata_->nodequeue);
                analysis_.mipTimerStop(kMipClockCurrentNodeToQueue);
            }

            analysis_.mipTimerStart(kMipClockNodePrunedLoop);
            if (search->currentNodePruned()) {
                analysis_.mipTimerStart(kMipClockSearchBacktrack);
                search->backtrack();
                analysis_.mipTimerStop(kMipClockSearchBacktrack);
                ++mipdata_->num_leaves;
                ++mipdata_->num_nodes;
                search->flushStatistics();

                mipdata_->domain.propagate();
                mipdata_->pruned_treeweight +=
                    mipdata_->nodequeue.pruneInfeasibleNodes(mipdata_->domain,
                        mipdata_->feastol);
                if (mipdata_->domain.infeasible()) {
                    mipdata_->nodequeue.clear();
                    mipdata_->pruned_treeweight = 1.0;
                    double prev_lower_bound = mipdata_->lower_bound;
                    mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);
                    if (!submip && mipdata_->lower_bound != prev_lower_bound)
                        mipdata_->updatePrimalDualIntegral(prev_lower_bound,
                            mipdata_->lower_bound,
                            mipdata_->upper_bound,
                            mipdata_->upper_bound);
                    analysis_.mipTimerStop(kMipClockNodePrunedLoop);
                    break;
                }

                if (mipdata_->checkLimits()) {
                    limit_reached = true;
                    break;
                }

                analysis_.mipTimerStart(kMipClockStoreBasis);
                double prev_lower_bound = mipdata_->lower_bound;
                mipdata_->lower_bound =
                    std::min(mipdata_->upper_bound, mipdata_->nodequeue.getBestLowerBound());
                if (!submip && mipdata_->lower_bound != prev_lower_bound)
                    mipdata_->updatePrimalDualIntegral(prev_lower_bound,
                        mipdata_->lower_bound,
                        mipdata_->upper_bound,
                        mipdata_->upper_bound);
                mipdata_->printDisplayLine();

                if (!mipdata_->domain.getChangedCols().empty()) {
                    highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                        "added %" HIGHSINT_FORMAT " global bound changes\n",
                        (HighsInt)mipdata_->domain.getChangedCols().size());
                    mipdata_->cliquetable.cleanupFixed(mipdata_->domain);
                    for (HighsInt col : mipdata_->domain.getChangedCols())
                        mipdata_->implications.cleanupVarbounds(col);
                    mipdata_->domain.setDomainChangeStack({});
                    search->resetLocalDomain();
                    mipdata_->domain.clearChangedCols();
                    mipdata_->removeFixedIndices();
                }
                analysis_.mipTimerStop(kMipClockStoreBasis);
                analysis_.mipTimerStop(kMipClockNodePrunedLoop);
                continue;
            }
            analysis_.mipTimerStop(kMipClockNodePrunedLoop);

            analysis_.mipTimerStart(kMipClockNodeSearchSeparation);
            sepa.separate(search->getLocalDomain());
            analysis_.mipTimerStop(kMipClockNodeSearchSeparation);

            if (mipdata_->domain.infeasible()) {
                search->cutoffNode();
                analysis_.mipTimerStart(kMipClockOpenNodesToQueue1);
                search->openNodesToQueue(mipdata_->nodequeue);
                analysis_.mipTimerStop(kMipClockOpenNodesToQueue1);
                mipdata_->nodequeue.clear();
                mipdata_->pruned_treeweight = 1.0;
                analysis_.mipTimerStart(kMipClockStoreBasis);
                double prev_lower_bound = mipdata_->lower_bound;
                mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);
                if (!submip && mipdata_->lower_bound != prev_lower_bound)
                    mipdata_->updatePrimalDualIntegral(prev_lower_bound,
                        mipdata_->lower_bound,
                        mipdata_->upper_bound,
                        mipdata_->upper_bound);
                break;
            }

            if (mipdata_->lp.getStatus() != HighsLpRelaxation::Status::kError &&
                mipdata_->lp.getStatus() != HighsLpRelaxation::Status::kNotSet)
                mipdata_->lp.storeBasis();

            basis = mipdata_->lp.getStoredBasis();
            if (!basis || !isBasisConsistent(mipdata_->lp.getLp(), *basis)) {
                HighsBasis b = mipdata_->firstrootbasis;
                b.row_status.resize(mipdata_->lp.numRows(), HighsBasisStatus::kBasic);
                basis = std::make_shared<const HighsBasis>(std::move(b));
                mipdata_->lp.setStoredBasis(basis);
            }

            // openNodesToQueue -> 到临时 best_nodes 队列
            HighsNodeQueue temp_nodes;
            temp_nodes.setNumCol(this->numCol());
            search->openNodesToQueue(temp_nodes);

            for (HighsInt k = 0; k < temp_nodes.numNodes(); ++k) {
                HighsNodeQueue::OpenNode node = temp_nodes.popBestNode();
                int64_t predicted_time = 1;  // 可换成你真实的预测函数
                HighsInt target_worker = best_nodes_idx % num_workers_;

                bool assigned = false;
                for (HighsInt w = target_worker; w < num_workers_; ++w) {
                    if (queue_times[w] + predicted_time <= *std::max_element(queue_times.begin(), queue_times.end()) ||
                        queue_times[w] == 0) {
                        prealloc_queues[w].emplace_back(std::move(node), predicted_time, best_nodes_idx);
                        queue_times[w] += predicted_time;
                        assigned = true;
                        break;
                    }
                }
                if (assigned) {
                    ++best_nodes_idx;
                }

                // 是否终止分配
                for (HighsInt w = 0; w < num_workers_; ++w) {
                    if (prealloc_queues[w].size() >= 3) {
                        stop_selecting = true;
                        break;
                    }
                }
                if (stop_selecting)
                    break;
            }
        }

        // 最终统一导入 best_nodes
        std::vector<NodeWithTime> all_nodes;
        for (auto& q : prealloc_queues)
            for (auto& n : q)
                all_nodes.push_back(std::move(n));

        std::sort(all_nodes.begin(), all_nodes.end(),
            [](const NodeWithTime& a, const NodeWithTime& b) {
                return a.best_nodes_index < b.best_nodes_index;
            });

        for (auto& node_info : all_nodes) {
            best_nodes.emplaceNode(std::move(node_info.node.domchgstack),
                std::move(node_info.node.branchings),
                node_info.node.lower_bound,
                node_info.node.estimate,
                node_info.node.depth);
        }

        analysis_.mipTimerStop(kMipClockNodeSearch);

   }
 }

//// 辅助函数：确定最佳节点数量m
//HighsInt HighsMipSolver::determineBestNodesCount() {
//    // 使用预分配算法选择节点
//    HighsNodeQueue temp_nodes;
//    temp_nodes.setNumCol(this->numCol());
//    allocateNodesToWorkers(temp_nodes);
//    return temp_nodes.numNodes();
//}
//
//// 节点预分配算法实现
//void HighsMipSolver::allocateNodesToWorkers(HighsNodeQueue& best_nodes) {
//    struct NodeWithTime {
//        HighsNodeQueue::OpenNode node;
//        int64_t predicted_time;
//        HighsInt best_nodes_index;
//
//        // 显式定义移动构造函数
//        NodeWithTime(NodeWithTime&& other) noexcept
//            : node(std::move(other.node)),
//            predicted_time(other.predicted_time),
//            best_nodes_index(other.best_nodes_index) {
//        }
//
//        // 显式定义移动赋值运算符
//        NodeWithTime& operator=(NodeWithTime&& other) noexcept {
//            if (this != &other) {
//                node = std::move(other.node);
//                predicted_time = other.predicted_time;
//                best_nodes_index = other.best_nodes_index;
//            }
//            return *this;
//        }
//
//        // 禁止拷贝构造和赋值
//        NodeWithTime(const NodeWithTime&) = delete;
//        NodeWithTime& operator=(const NodeWithTime&) = delete;
//
//        // 显式定义构造函数
//        NodeWithTime(HighsNodeQueue::OpenNode&& n, int64_t time, HighsInt index)
//            : node(std::move(n)), predicted_time(time), best_nodes_index(index) {
//        }
//    };
//
//    // 预分配队列，每个worker一个队列
//    std::vector<std::vector<NodeWithTime>> prealloc_queues(num_workers_);
//    // 记录每个队列的总预测时间
//    std::vector<int64_t> queue_times(num_workers_, 0);
//
//    // 随机数生成器，用于预测求解时间
//    /*std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_int_distribution<> dis(1, 5);*/
//
//    /* 预测函数接口
//    auto predict_solve_time = [&](const HighsNodeQueue::OpenNode& node) {
//     实际应用中应替换为真实的预测函数
//        return dis(gen);
//        };*/
//
//    HighsInt best_nodes_idx = 0; // best_nodes序列中的索引，从0开始
//
//    srand(static_cast<unsigned int>(time(nullptr)));
//
//    while (!mipdata_->nodequeue.empty()) {
//        // 从nodequeue中pop出一个节点
//        HighsNodeQueue::OpenNode next_node = mipdata_->nodequeue.popBestNode();
//
//        // 预测该节点的求解时间
//        //int64_t predicted_time = predict_solve_time(next_node);
//
//        // 使用简化的随机数生成
//        int64_t predicted_time = 1;//rand() % 5 + 1;
//
//        // 确定该节点应该放入的队列
//        HighsInt target_worker = best_nodes_idx % num_workers_;
//
//        // 尝试将节点放入目标队列
//        bool node_assigned = false;
//        for (HighsInt w = target_worker; w < num_workers_; ++w) {
//            if (queue_times[w] + predicted_time <= *std::max_element(queue_times.begin(), queue_times.end()) ||
//                queue_times[w] == 0) {
//                prealloc_queues[w].emplace_back(std::move(next_node), predicted_time, best_nodes_idx);
//                queue_times[w] += predicted_time;
//                node_assigned = true;
//                break;
//            }
//        }
//
//        if (!node_assigned) continue;
//        ++best_nodes_idx;
//
//        // 检查终止条件
//        bool should_terminate = false;
//        for (HighsInt w = 0; w < num_workers_; ++w) {
//            if (prealloc_queues[w].size() >= 3) {
//                should_terminate = true;
//                break;
//            }
//        }
//        if (should_terminate || mipdata_->nodequeue.empty()) {
//            // 使用移动语义收集所有节点
//            std::vector<NodeWithTime> all_nodes;
//
//            // 预分配足够空间，避免中途扩容
//            size_t total_size = 0;
//            for (auto& queue : prealloc_queues) {
//                total_size += queue.size();
//            }
//            all_nodes.reserve(total_size);
//
//            // 使用 std::move 移动每个元素，避免拷贝
//            for (auto& queue : prealloc_queues) {
//                for (auto& node_info : queue) {
//                    all_nodes.push_back(std::move(node_info));
//                }
//                // 清空队列，确保不再使用
//                queue.clear();
//            }
//
//            // 排序
//            std::sort(all_nodes.begin(), all_nodes.end(),
//                [](const NodeWithTime& a, const NodeWithTime& b) {
//                    return a.best_nodes_index < b.best_nodes_index;
//                });
//
//            // 添加节点到best_nodes
//            for (auto& node_info : all_nodes) {
//                best_nodes.emplaceNode(
//                    std::move(node_info.node.domchgstack),
//                    std::move(node_info.node.branchings),
//                    node_info.node.lower_bound,
//                    node_info.node.estimate,
//                    node_info.node.depth
//                );
//            }
//
//            break;
//        }
//    }
//}

void HighsMipSolver::workerLoop(int worker_id) {
  int seen_gen = generation_;
  auto& cv = cv_workers_[worker_id];
  while (true) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv.wait(lk, [&]() { return terminate_workers_ || generation_ != seen_gen; });
    if (terminate_workers_) break;

    seen_gen = generation_;
    lk.unlock();

    //size_t plungestart = mipdata_->num_nodes;
    bool limit_reached = false;
    bool considerHeuristics = true;

    // 获取分配给该worker的节点数 s_i
    HighsInt s_i = worker_node_counts_[worker_id];


    // 处理分配给该worker的所有节点
    for (HighsInt j = 0; j < s_i && !limit_reached; ++j) {
        // 节点索引从1开始：j+1表示该节点在best_nodes序列中的序号
        // worker_id+1表示该节点分配给的worker编号
        HighsSearch* search = worker_loop_search_ptrs_[worker_id][j];
        bool node_processed = false; // 添加节点处理状态标记
        size_t plungestart = mipdata_->num_nodes;
    //if (worker_id < loop_batch_size_) {
    //  //loop_search_ptrs_[worker_id]->dive();
      while (true) {
        // Possibly apply primal heuristics
        if (considerHeuristics && mipdata_->moreHeuristicsAllowed()) {
          std::lock_guard<std::recursive_mutex> guard(g_mutex);
          analysis_.mipTimerStart(kMipClockDiveEvaluateNode);
          const HighsSearch::NodeResult evaluate_node_result = search->evaluateNode();
          analysis_.mipTimerStop(kMipClockDiveEvaluateNode);

          if (evaluate_node_result == HighsSearch::NodeResult::kSubOptimal)
            break;
          if (search->currentNodePruned()) {
            ++mipdata_->num_leaves;
            search->flushStatistics();
            node_processed = true; // 标记节点已处理完成
            continue;
          } else {
            analysis_.mipTimerStart(kMipClockDivePrimalHeuristics);
            if (mipdata_->incumbent.empty()) {
              std::lock_guard<std::recursive_mutex> guard(g_mutex);
              analysis_.mipTimerStart(kMipClockDiveRandomizedRounding);
              mipdata_->heuristics.randomizedRounding(mipdata_->lp.getLpSolver().getSolution().col_value);
              analysis_.mipTimerStop(kMipClockDiveRandomizedRounding);
            }

            if (mipdata_->incumbent.empty()) {
              analysis_.mipTimerStart(kMipClockDiveRens);
              mipdata_->heuristics.RENS(mipdata_->lp.getLpSolver().getSolution().col_value);
              analysis_.mipTimerStop(kMipClockDiveRens);
            } else {
              analysis_.mipTimerStart(kMipClockDiveRins);
              mipdata_->heuristics.RINS(mipdata_->lp.getLpSolver().getSolution().col_value);
              analysis_.mipTimerStop(kMipClockDiveRins);
            }

            mipdata_->heuristics.flushStatistics();
            analysis_.mipTimerStop(kMipClockDivePrimalHeuristics);
          }
        }

        considerHeuristics = false;

        if (mipdata_->domain.infeasible()) break;
        
        //std::lock_guard<std::recursive_mutex> guard(g_mutex);
        if (!search->currentNodePruned()) {
            double this_dive_time;
          {
            std::lock_guard<std::recursive_mutex> guard(g_mutex);
            this_dive_time = -analysis_.mipTimerRead(kMipClockTheDive);
          }
         
          //analysis_.mipTimerStart(kMipClockTheDive);
          HighsSearch::NodeResult search_dive_result;
          //{
            //std::lock_guard<std::recursive_mutex> guard(g_mutex);
            search_dive_result = search->dive();
          //}
          //const HighsSearch::NodeResult search_dive_result = loop_search_ptrs_[worker_id]->dive();
          //analysis_.mipTimerStop(kMipClockTheDive);
         
          if (analysis_.analyse_mip_time) {
            this_dive_time += analysis_.mipTimerRead(kMipClockNodeSearch);
            std::lock_guard<std::recursive_mutex> guard(g_mutex);
            analysis_.dive_time.push_back(this_dive_time);
          }
          if (search_dive_result == HighsSearch::NodeResult::kSubOptimal) break;

          {
            std::lock_guard<std::recursive_mutex> guard(g_mutex);
            ++mipdata_->num_leaves;
          }
          {
            std::lock_guard<std::recursive_mutex> guard(g_mutex);
            search->flushStatistics();
          }
        }
        std::lock_guard<std::recursive_mutex> guard(g_mutex);
        if (mipdata_->checkLimits()) {
          limit_reached = true;
          break;
        }

        HighsInt numPlungeNodes = mipdata_->num_nodes - plungestart;
        if (numPlungeNodes >= 100) break;

        analysis_.mipTimerStart(kMipClockBacktrackPlunge);
        const bool backtrack_plunge = search->backtrackPlunge(mipdata_->nodequeue);
        analysis_.mipTimerStop(kMipClockBacktrackPlunge);
        //if (!backtrack_plunge) break;
        if (!backtrack_plunge) {
            node_processed = true; // 回溯失败，节点处理完成
            break;
        }
        assert(search->hasNode());

        if (mipdata_->conflictPool.getNumConflicts() > options_mip_->mip_pool_soft_limit) {
          std::lock_guard<std::recursive_mutex> guard(g_mutex);
          analysis_.mipTimerStart(kMipClockPerformAging2);
          mipdata_->conflictPool.performAging();
          analysis_.mipTimerStop(kMipClockPerformAging2);
        }

        search->flushStatistics();
        mipdata_->printDisplayLine();
        // printf("continue plunging due to good estimate\n");
      }  // while (true)

      /*std::lock_guard<std::mutex> lg(mtx_);
      if (++ready_workers_ == loop_batch_size_) cv_master_.notify_one();*/
      if (limit_reached) break;
    }
    std::lock_guard<std::mutex> lg(mtx_);
    //if (++ready_workers_ == num_workers_) cv_master_.notify_one();
    if (++ready_workers_ == num_workers_) {
        cv_master_.notify_one();
    }
  }
}

void HighsMipSolver::runSerial() {
  modelstatus_ = HighsModelStatus::kNotset;

  if (submip) {
    analysis_.analyse_mip_time = false;
  } else {
    analysis_.timer_ = &this->timer_;
    analysis_.setup(*orig_model_, *options_mip_);
  }
  timer_.start();

  improving_solution_file_ = nullptr;
  if (!submip && options_mip_->mip_improving_solution_file != "")
    improving_solution_file_ =
        fopen(options_mip_->mip_improving_solution_file.c_str(), "w");

  mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
  analysis_.mipTimerStart(kMipClockPresolve);
  analysis_.mipTimerStart(kMipClockInit);
  mipdata_->init();
  analysis_.mipTimerStop(kMipClockInit);
  analysis_.mipTimerStart(kMipClockRunPresolve);
  mipdata_->runPresolve(options_mip_->presolve_reduction_limit);
  analysis_.mipTimerStop(kMipClockRunPresolve);
  analysis_.mipTimerStop(kMipClockPresolve);
  if (analysis_.analyse_mip_time && !submip)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "MIP-Timing: %11.2g - completed presolve\n", timer_.read());
  // Identify whether time limit has been reached (in presolve)
  if (modelstatus_ == HighsModelStatus::kNotset &&
      timer_.read() >= options_mip_->time_limit)
    modelstatus_ = HighsModelStatus::kTimeLimit;

  if (modelstatus_ != HighsModelStatus::kNotset) {
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "Presolve: %s\n",
                 utilModelStatusToString(modelstatus_).c_str());
    if (modelstatus_ == HighsModelStatus::kOptimal) {
      mipdata_->lower_bound = 0;
      mipdata_->upper_bound = 0;
      mipdata_->transformNewIntegerFeasibleSolution(std::vector<double>());
      mipdata_->saveReportMipSolution();
    }
    cleanupSolve();
    return;
  }

  analysis_.mipTimerStart(kMipClockSolve);

  if (analysis_.analyse_mip_time && !submip)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "MIP-Timing: %11.2g - starting  setup\n", timer_.read());
  analysis_.mipTimerStart(kMipClockRunSetup);
  mipdata_->runSetup();
  analysis_.mipTimerStop(kMipClockRunSetup);
  if (analysis_.analyse_mip_time && !submip)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "MIP-Timing: %11.2g - completed setup\n", timer_.read());
restart:
  if (modelstatus_ == HighsModelStatus::kNotset) {
    // Check limits have not been reached before evaluating root node
    if (mipdata_->checkLimits()) {
      cleanupSolve();
      return;
    }
    // Possibly look for primal solution from the user
    if (!submip && callback_->user_callback &&
        callback_->active[kCallbackMipUserSolution])
      mipdata_->callbackUserSolution(solution_objective_,
                                     kUserMipSolutionCallbackOriginAfterSetup);

    // Apply the trivial heuristics
    analysis_.mipTimerStart(kMipClockTrivialHeuristics);
    HighsModelStatus model_status = mipdata_->trivialHeuristics();
    analysis_.mipTimerStop(kMipClockTrivialHeuristics);
    if (modelstatus_ == HighsModelStatus::kNotset &&
        model_status == HighsModelStatus::kInfeasible) {
      // trivialHeuristics can spot trivial infeasibility, so act on it
      modelstatus_ = model_status;
      cleanupSolve();
      return;
    }
    if (analysis_.analyse_mip_time && !submip)
      highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                   "MIP-Timing: %11.2g - starting evaluate root node\n",
                   timer_.read());
    analysis_.mipTimerStart(kMipClockEvaluateRootNode);
    mipdata_->evaluateRootNode();
    analysis_.mipTimerStop(kMipClockEvaluateRootNode);
    // Sometimes the analytic centre calculation is not completed when
    // evaluateRootNode returns, so stop its clock if it's running
    if (analysis_.analyse_mip_time &&
        analysis_.mipTimerRunning(kMipClockIpmSolveLp))
      analysis_.mipTimerStop(kMipClockIpmSolveLp);
    if (analysis_.analyse_mip_time && !submip)
      highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                   "MIP-Timing: %11.2g - completed evaluate root node\n",
                   timer_.read());
    // age 5 times to remove stored but never violated cuts after root
    // separation
    analysis_.mipTimerStart(kMipClockPerformAging0);
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
    mipdata_->cutpool.performAging();
    analysis_.mipTimerStop(kMipClockPerformAging0);
  }
  if (mipdata_->nodequeue.empty() || mipdata_->checkLimits()) {
    cleanupSolve();
    return;
  }

  std::shared_ptr<const HighsBasis> basis;
  HighsSearch search{*this, mipdata_->pseudocost};
  mipdata_->debugSolution.registerDomain(search.getLocalDomain());
  HighsSeparation sepa(*this);

  search.setLpRelaxation(&mipdata_->lp);
  sepa.setLpRelaxation(&mipdata_->lp);

  double prev_lower_bound = mipdata_->lower_bound;

  mipdata_->lower_bound = mipdata_->nodequeue.getBestLowerBound();

  bool bound_change = mipdata_->lower_bound != prev_lower_bound;
  if (!submip && bound_change)
    mipdata_->updatePrimalDualIntegral(prev_lower_bound, mipdata_->lower_bound,
                                       mipdata_->upper_bound,
                                       mipdata_->upper_bound);

  mipdata_->printDisplayLine();
  search.installNode(mipdata_->nodequeue.popBestBoundNode());
  int64_t numStallNodes = 0;
  int64_t lastLbLeave = 0;
  int64_t numQueueLeaves = 0;
  HighsInt numHugeTreeEstim = 0;
  int64_t numNodesLastCheck = mipdata_->num_nodes;
  int64_t nextCheck = mipdata_->num_nodes;
  double treeweightLastCheck = 0.0;
  double upperLimLastCheck = mipdata_->upper_limit;
  double lowerBoundLastCheck = mipdata_->lower_bound;
  analysis_.mipTimerStart(kMipClockSearch);
  while (search.hasNode()) {
    // Possibly look for primal solution from the user
    if (!submip && callback_->user_callback &&
        callback_->active[kCallbackMipUserSolution])
      mipdata_->callbackUserSolution(solution_objective_,
                                     kUserMipSolutionCallbackOriginBeforeDive);

    analysis_.mipTimerStart(kMipClockPerformAging1);
    mipdata_->conflictPool.performAging();
    analysis_.mipTimerStop(kMipClockPerformAging1);
    // set iteration limit for each lp solve during the dive to 10 times the
    // average nodes

    HighsInt iterlimit = 10 * std::max(mipdata_->lp.getAvgSolveIters(),
                                       mipdata_->avgrootlpiters);
    iterlimit = std::max({HighsInt{10000}, iterlimit,
                          HighsInt((3 * mipdata_->firstrootlpiters) / 2)});

    mipdata_->lp.setIterationLimit(iterlimit);

    // perform the dive and put the open nodes to the queue
    size_t plungestart = mipdata_->num_nodes;
    bool limit_reached = false;
    bool considerHeuristics = true;
    analysis_.mipTimerStart(kMipClockDive);
    while (true) {
      // Possibly apply primal heuristics
      if (considerHeuristics && mipdata_->moreHeuristicsAllowed()) {
        analysis_.mipTimerStart(kMipClockDiveEvaluateNode);
        const HighsSearch::NodeResult evaluate_node_result =
            search.evaluateNode();
        analysis_.mipTimerStop(kMipClockDiveEvaluateNode);

        if (evaluate_node_result == HighsSearch::NodeResult::kSubOptimal) break;

        if (search.currentNodePruned()) {
          ++mipdata_->num_leaves;
          search.flushStatistics();
        } else {
          analysis_.mipTimerStart(kMipClockDivePrimalHeuristics);
          if (mipdata_->incumbent.empty()) {
            analysis_.mipTimerStart(kMipClockDiveRandomizedRounding);
            mipdata_->heuristics.randomizedRounding(
                mipdata_->lp.getLpSolver().getSolution().col_value);
            analysis_.mipTimerStop(kMipClockDiveRandomizedRounding);
          }

          if (mipdata_->incumbent.empty()) {
            analysis_.mipTimerStart(kMipClockDiveRens);
            mipdata_->heuristics.RENS(
                mipdata_->lp.getLpSolver().getSolution().col_value);
            analysis_.mipTimerStop(kMipClockDiveRens);
          } else {
            analysis_.mipTimerStart(kMipClockDiveRins);
            mipdata_->heuristics.RINS(
                mipdata_->lp.getLpSolver().getSolution().col_value);
            analysis_.mipTimerStop(kMipClockDiveRins);
          }

          mipdata_->heuristics.flushStatistics();
          analysis_.mipTimerStop(kMipClockDivePrimalHeuristics);
        }
      }

      considerHeuristics = false;

      if (mipdata_->domain.infeasible()) break;

      if (!search.currentNodePruned()) {
        double this_dive_time = -analysis_.mipTimerRead(kMipClockTheDive);
        analysis_.mipTimerStart(kMipClockTheDive);
        const HighsSearch::NodeResult search_dive_result = search.dive();
        analysis_.mipTimerStop(kMipClockTheDive);
        if (analysis_.analyse_mip_time) {
          this_dive_time += analysis_.mipTimerRead(kMipClockNodeSearch);
          analysis_.dive_time.push_back(this_dive_time);
        }
        if (search_dive_result == HighsSearch::NodeResult::kSubOptimal) break;

        ++mipdata_->num_leaves;

        search.flushStatistics();
      }

      if (mipdata_->checkLimits()) {
        limit_reached = true;
        break;
      }

      HighsInt numPlungeNodes = mipdata_->num_nodes - plungestart;
      if (numPlungeNodes >= 100) break;

      analysis_.mipTimerStart(kMipClockBacktrackPlunge);
      const bool backtrack_plunge = search.backtrackPlunge(mipdata_->nodequeue);
      analysis_.mipTimerStop(kMipClockBacktrackPlunge);
      if (!backtrack_plunge) break;

      assert(search.hasNode());

      if (mipdata_->conflictPool.getNumConflicts() >
          options_mip_->mip_pool_soft_limit) {
        analysis_.mipTimerStart(kMipClockPerformAging2);
        mipdata_->conflictPool.performAging();
        analysis_.mipTimerStop(kMipClockPerformAging2);
      }

      search.flushStatistics();
      mipdata_->printDisplayLine();
      // printf("continue plunging due to good estimate\n");
    }  // while (true)
    analysis_.mipTimerStop(kMipClockDive);

    analysis_.mipTimerStart(kMipClockOpenNodesToQueue0);
    search.openNodesToQueue(mipdata_->nodequeue);
    analysis_.mipTimerStop(kMipClockOpenNodesToQueue0);

    search.flushStatistics();

    if (limit_reached) {
      double prev_lower_bound = mipdata_->lower_bound;

      mipdata_->lower_bound = std::min(mipdata_->upper_bound,
                                       mipdata_->nodequeue.getBestLowerBound());

      bool bound_change = mipdata_->lower_bound != prev_lower_bound;
      if (!submip && bound_change)
        mipdata_->updatePrimalDualIntegral(
            prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
            mipdata_->upper_bound);
      mipdata_->printDisplayLine();
      break;
    }

    // the search datastructure should have no installed node now
    assert(!search.hasNode());

    // propagate the global domain
    analysis_.mipTimerStart(kMipClockDomainPropgate);
    mipdata_->domain.propagate();
    analysis_.mipTimerStop(kMipClockDomainPropgate);

    analysis_.mipTimerStart(kMipClockPruneInfeasibleNodes);
    mipdata_->pruned_treeweight += mipdata_->nodequeue.pruneInfeasibleNodes(
        mipdata_->domain, mipdata_->feastol);
    analysis_.mipTimerStop(kMipClockPruneInfeasibleNodes);

    // if global propagation detected infeasibility, stop here
    if (mipdata_->domain.infeasible()) {
      mipdata_->nodequeue.clear();
      mipdata_->pruned_treeweight = 1.0;

      double prev_lower_bound = mipdata_->lower_bound;

      mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);

      bool bound_change = mipdata_->lower_bound != prev_lower_bound;
      if (!submip && bound_change)
        mipdata_->updatePrimalDualIntegral(
            prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
            mipdata_->upper_bound);
      mipdata_->printDisplayLine();
      break;
    }

    double prev_lower_bound = mipdata_->lower_bound;

    mipdata_->lower_bound = std::min(mipdata_->upper_bound,
                                     mipdata_->nodequeue.getBestLowerBound());
    bool bound_change = mipdata_->lower_bound != prev_lower_bound;
    if (!submip && bound_change)
      mipdata_->updatePrimalDualIntegral(
          prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
          mipdata_->upper_bound);
    mipdata_->printDisplayLine();
    if (mipdata_->nodequeue.empty()) break;

    // if global propagation found bound changes, we update the local domain
    if (!mipdata_->domain.getChangedCols().empty()) {
      analysis_.mipTimerStart(kMipClockUpdateLocalDomain);
      highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                  "added %" HIGHSINT_FORMAT " global bound changes\n",
                  (HighsInt)mipdata_->domain.getChangedCols().size());
      mipdata_->cliquetable.cleanupFixed(mipdata_->domain);
      for (HighsInt col : mipdata_->domain.getChangedCols())
        mipdata_->implications.cleanupVarbounds(col);

      mipdata_->domain.setDomainChangeStack(std::vector<HighsDomainChange>());
      search.resetLocalDomain();

      mipdata_->domain.clearChangedCols();
      mipdata_->removeFixedIndices();
      analysis_.mipTimerStop(kMipClockUpdateLocalDomain);
    }

    if (!submip && mipdata_->num_nodes >= nextCheck) {
      auto nTreeRestarts = mipdata_->numRestarts - mipdata_->numRestartsRoot;
      double currNodeEstim =
          numNodesLastCheck - mipdata_->num_nodes_before_run +
          (mipdata_->num_nodes - numNodesLastCheck) *
              double(1.0 - mipdata_->pruned_treeweight) /
              std::max(
                  double(mipdata_->pruned_treeweight - treeweightLastCheck),
                  mipdata_->epsilon);
      // printf(
      //     "nTreeRestarts: %d, numNodesThisRun: %ld, numNodesLastCheck: %ld,
      //     " "currNodeEstim: %g, " "prunedTreeWeightDelta: %g,
      //     numHugeTreeEstim: %d, numLeavesThisRun:
      //     "
      //     "%ld\n",
      //     nTreeRestarts, mipdata_->num_nodes -
      //     mipdata_->num_nodes_before_run, numNodesLastCheck -
      //     mipdata_->num_nodes_before_run, currNodeEstim, 100.0 *
      //     double(mipdata_->pruned_treeweight - treeweightLastCheck),
      //     numHugeTreeEstim,
      //     mipdata_->num_leaves - mipdata_->num_leaves_before_run);

      bool doRestart = false;

      double activeIntegerRatio =
          1.0 - mipdata_->percentageInactiveIntegers() / 100.0;
      activeIntegerRatio *= activeIntegerRatio;

      if (!doRestart) {
        double gapReduction = 1.0;
        if (mipdata_->upper_limit != kHighsInf) {
          double oldGap = upperLimLastCheck - lowerBoundLastCheck;
          double newGap = mipdata_->upper_limit - mipdata_->lower_bound;
          gapReduction = oldGap / newGap;
        }

        if (gapReduction < 1.0 + (0.05 / activeIntegerRatio) &&
            currNodeEstim >=
                activeIntegerRatio * 20 *
                    (mipdata_->num_nodes - mipdata_->num_nodes_before_run)) {
          nextCheck = mipdata_->num_nodes + 100;
          ++numHugeTreeEstim;
        } else {
          numHugeTreeEstim = 0;
          treeweightLastCheck = double(mipdata_->pruned_treeweight);
          numNodesLastCheck = mipdata_->num_nodes;
          upperLimLastCheck = mipdata_->upper_limit;
          lowerBoundLastCheck = mipdata_->lower_bound;
        }

        // Possibly prevent restart - necessary for debugging presolve
        // errors: see #1553
        if (options_mip_->mip_allow_restart) {
          int64_t minHugeTreeOffset =
              (mipdata_->num_leaves - mipdata_->num_leaves_before_run) / 1000;
          int64_t minHugeTreeEstim = HighsIntegers::nearestInteger(
              activeIntegerRatio * (10 + minHugeTreeOffset) *
              std::pow(1.5, nTreeRestarts));

          doRestart = numHugeTreeEstim >= minHugeTreeEstim;
        } else {
          doRestart = false;
        }
      } else {
        // count restart due to many fixings within the first 1000 nodes as
        // root restart
        ++mipdata_->numRestartsRoot;
      }

      if (doRestart) {
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                     "\nRestarting search from the root node\n");
        mipdata_->performRestart();
        analysis_.mipTimerStop(kMipClockSearch);
        goto restart;
      }
    }  // if (!submip && mipdata_->num_nodes >= nextCheck))

    // remove the iteration limit when installing a new node
    // mipdata_->lp.setIterationLimit();

    // loop to install the next node for the search
    double this_node_search_time = -analysis_.mipTimerRead(kMipClockNodeSearch);
    analysis_.mipTimerStart(kMipClockNodeSearch);

    while (!mipdata_->nodequeue.empty()) {
      // printf("popping node from nodequeue (length = %" HIGHSINT_FORMAT ")\n",
      // (HighsInt)nodequeue.size());
      assert(!search.hasNode());

    if (numQueueLeaves - lastLbLeave >= 10) {
      search.installNode(mipdata_->nodequeue.popBestBoundNode());
      lastLbLeave = numQueueLeaves;
    } else {
      HighsInt bestBoundNodeStackSize =
          mipdata_->nodequeue.getBestBoundDomchgStackSize();
      double bestBoundNodeLb = mipdata_->nodequeue.getBestLowerBound();
      HighsNodeQueue::OpenNode nextNode(mipdata_->nodequeue.popBestNode());
      if (nextNode.lower_bound == bestBoundNodeLb &&
          (HighsInt)nextNode.domchgstack.size() == bestBoundNodeStackSize)
        lastLbLeave = numQueueLeaves;
      search.installNode(std::move(nextNode));
    }

    ++numQueueLeaves;

    if (search.getCurrentEstimate() >= mipdata_->upper_limit) {
      ++numStallNodes;
      if (options_mip_->mip_max_stall_nodes != kHighsIInf &&
          numStallNodes >= options_mip_->mip_max_stall_nodes) {
        limit_reached = true;
        modelstatus_ = HighsModelStatus::kSolutionLimit;
        break;
      }
    } else
      numStallNodes = 0;

    assert(search.hasNode());

    // we evaluate the node directly here instead of performing a dive
    // because we first want to check if the node is not fathomed due to
    // new global information before we perform separation rounds for the node
    analysis_.mipTimerStart(kMipClockEvaluateNode1);
      const HighsSearch::NodeResult evaluate_node_result =
          search.evaluateNode();
    analysis_.mipTimerStop(kMipClockEvaluateNode1);
    if (evaluate_node_result == HighsSearch::NodeResult::kSubOptimal) {
      analysis_.mipTimerStart(kMipClockCurrentNodeToQueue);
      search.currentNodeToQueue(mipdata_->nodequeue);
      analysis_.mipTimerStop(kMipClockCurrentNodeToQueue);
    }

    // if the node was pruned we remove it from the search and install the
    // next node from the queue
    analysis_.mipTimerStart(kMipClockNodePrunedLoop);
    if (search.currentNodePruned()) {
      //	analysis_.mipTimerStart(kMipClockSearchBacktrack);
      search.backtrack();
      //	analysis_.mipTimerStop(kMipClockSearchBacktrack);
      ++mipdata_->num_leaves;
      ++mipdata_->num_nodes;
      search.flushStatistics();

      mipdata_->domain.propagate();
      mipdata_->pruned_treeweight += mipdata_->nodequeue.pruneInfeasibleNodes(
          mipdata_->domain, mipdata_->feastol);

      if (mipdata_->domain.infeasible()) {
        mipdata_->nodequeue.clear();
        mipdata_->pruned_treeweight = 1.0;

        double prev_lower_bound = mipdata_->lower_bound;

        mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);

        bool bound_change = mipdata_->lower_bound != prev_lower_bound;
        if (!submip && bound_change)
          mipdata_->updatePrimalDualIntegral(
              prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
              mipdata_->upper_bound);
        analysis_.mipTimerStop(kMipClockNodePrunedLoop);
        break;
      }

      if (mipdata_->checkLimits()) {
        limit_reached = true;
        break;
      }

      //	analysis_.mipTimerStart(kMipClockStoreBasis);
      double prev_lower_bound = mipdata_->lower_bound;

        mipdata_->lower_bound = std::min(
            mipdata_->upper_bound, mipdata_->nodequeue.getBestLowerBound());

      bool bound_change = mipdata_->lower_bound != prev_lower_bound;
      if (!submip && bound_change)
        mipdata_->updatePrimalDualIntegral(
            prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
            mipdata_->upper_bound);
      mipdata_->printDisplayLine();

      if (!mipdata_->domain.getChangedCols().empty()) {
        highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                    "added %" HIGHSINT_FORMAT " global bound changes\n",
                    (HighsInt)mipdata_->domain.getChangedCols().size());
        mipdata_->cliquetable.cleanupFixed(mipdata_->domain);
        for (HighsInt col : mipdata_->domain.getChangedCols())
          mipdata_->implications.cleanupVarbounds(col);

          mipdata_->domain.setDomainChangeStack(
              std::vector<HighsDomainChange>());
        search.resetLocalDomain();

        mipdata_->domain.clearChangedCols();
        mipdata_->removeFixedIndices();
      }
      //	analysis_.mipTimerStop(kMipClockStoreBasis);

      analysis_.mipTimerStop(kMipClockNodePrunedLoop);
      continue;
    }
    analysis_.mipTimerStop(kMipClockNodePrunedLoop);

    // the node is still not fathomed, so perform separation
    analysis_.mipTimerStart(kMipClockNodeSearchSeparation);
    sepa.separate(search.getLocalDomain());
    analysis_.mipTimerStop(kMipClockNodeSearchSeparation);

    if (mipdata_->domain.infeasible()) {
      search.cutoffNode();
      analysis_.mipTimerStart(kMipClockOpenNodesToQueue1);
      search.openNodesToQueue(mipdata_->nodequeue);
      analysis_.mipTimerStop(kMipClockOpenNodesToQueue1);
      mipdata_->nodequeue.clear();
      mipdata_->pruned_treeweight = 1.0;

      analysis_.mipTimerStart(kMipClockStoreBasis);
      double prev_lower_bound = mipdata_->lower_bound;

      mipdata_->lower_bound = std::min(kHighsInf, mipdata_->upper_bound);

      bool bound_change = mipdata_->lower_bound != prev_lower_bound;
      if (!submip && bound_change)
        mipdata_->updatePrimalDualIntegral(
            prev_lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
            mipdata_->upper_bound);
      break;
    }

    // after separation we store the new basis and proceed with the outer loop
    // to perform a dive from this node
    if (mipdata_->lp.getStatus() != HighsLpRelaxation::Status::kError &&
        mipdata_->lp.getStatus() != HighsLpRelaxation::Status::kNotSet)
      mipdata_->lp.storeBasis();

    basis = mipdata_->lp.getStoredBasis();
    if (!basis || !isBasisConsistent(mipdata_->lp.getLp(), *basis)) {
      HighsBasis b = mipdata_->firstrootbasis;
      b.row_status.resize(mipdata_->lp.numRows(), HighsBasisStatus::kBasic);
      basis = std::make_shared<const HighsBasis>(std::move(b));
      mipdata_->lp.setStoredBasis(basis);
    }

    break;
  }  // while(!mipdata_->nodequeue.empty())
  analysis_.mipTimerStop(kMipClockNodeSearch);
  if (analysis_.analyse_mip_time) {
    this_node_search_time += analysis_.mipTimerRead(kMipClockNodeSearch);
    analysis_.node_search_time.push_back(this_node_search_time);
  }
  if (limit_reached) break;
}  // while(search.hasNode())
  analysis_.mipTimerStop(kMipClockSearch);

  cleanupSolve();
}

void HighsMipSolver::cleanupSolve() {
  // Force a final logging line
  mipdata_->printDisplayLine(kSolutionSourceCleanup);
  // Stop the solve clock - which won't be running if presolve
  // determines the model status
  if (analysis_.mipTimerRunning(kMipClockSolve))
    analysis_.mipTimerStop(kMipClockSolve);

  // Need to complete the calculation of P-D integral, checking for NO
  // gap change
  mipdata_->updatePrimalDualIntegral(
      mipdata_->lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
      mipdata_->upper_bound, false);
  analysis_.mipTimerStart(kMipClockPostsolve);

  bool havesolution = solution_objective_ != kHighsInf;
  bool feasible;
  if (havesolution)
    feasible =
        bound_violation_ <= options_mip_->mip_feasibility_tolerance &&
        integrality_violation_ <= options_mip_->mip_feasibility_tolerance &&
        row_violation_ <= options_mip_->mip_feasibility_tolerance;
  else
    feasible = false;

  dual_bound_ = mipdata_->lower_bound;
  if (mipdata_->objectiveFunction.isIntegral()) {
    double rounded_lower_bound =
        std::ceil(mipdata_->lower_bound *
                      mipdata_->objectiveFunction.integralScale() -
                  mipdata_->feastol) /
        mipdata_->objectiveFunction.integralScale();
    dual_bound_ = std::max(dual_bound_, rounded_lower_bound);
  }
  dual_bound_ += model_->offset_;
  primal_bound_ = mipdata_->upper_bound + model_->offset_;
  node_count_ = mipdata_->num_nodes;
  total_lp_iterations_ = mipdata_->total_lp_iterations;
  dual_bound_ = std::min(dual_bound_, primal_bound_);
  primal_dual_integral_ = mipdata_->primal_dual_integral.value;

  // adjust objective sense in case of maximization problem
  if (orig_model_->sense_ == ObjSense::kMaximize) {
    dual_bound_ = -dual_bound_;
    primal_bound_ = -primal_bound_;
  }

  if (modelstatus_ == HighsModelStatus::kNotset ||
      modelstatus_ == HighsModelStatus::kInfeasible) {
    if (feasible && havesolution)
      modelstatus_ = HighsModelStatus::kOptimal;
    else
      modelstatus_ = HighsModelStatus::kInfeasible;
  }

  analysis_.mipTimerStop(kMipClockPostsolve);
  timer_.stop();

  std::string solutionstatus = "-";

  if (havesolution) {
    bool feasible =
        bound_violation_ <= options_mip_->mip_feasibility_tolerance &&
        integrality_violation_ <= options_mip_->mip_feasibility_tolerance &&
        row_violation_ <= options_mip_->mip_feasibility_tolerance;
    solutionstatus = feasible ? "feasible" : "infeasible";
  }

  gap_ = fabs(primal_bound_ - dual_bound_);
  if (primal_bound_ == 0.0)
    gap_ = dual_bound_ == 0.0 ? 0.0 : kHighsInf;
  else if (primal_bound_ != kHighsInf)
    gap_ = fabs(primal_bound_ - dual_bound_) / fabs(primal_bound_);
  else
    gap_ = kHighsInf;

  std::array<char, 128> gapString = {};

  if (gap_ == kHighsInf)
    std::strcpy(gapString.data(), "inf");
  else {
    double printTol = std::max(std::min(1e-2, 1e-1 * gap_), 1e-6);
    auto gapValString = highsDoubleToString(100.0 * gap_, printTol);
    double gapTol = options_mip_->mip_rel_gap;

    if (options_mip_->mip_abs_gap > options_mip_->mip_feasibility_tolerance) {
      gapTol = primal_bound_ == 0.0
                   ? kHighsInf
                   : std::max(gapTol,
                              options_mip_->mip_abs_gap / fabs(primal_bound_));
    }

    if (gapTol == 0.0)
      std::snprintf(gapString.data(), gapString.size(), "%s%%",
                    gapValString.data());
    else if (gapTol != kHighsInf) {
      printTol = std::max(std::min(1e-2, 1e-1 * gapTol), 1e-6);
      auto gapTolString = highsDoubleToString(100.0 * gapTol, printTol);
      std::snprintf(gapString.data(), gapString.size(),
                    "%s%% (tolerance: %s%%)", gapValString.data(),
                    gapTolString.data());
    } else
      std::snprintf(gapString.data(), gapString.size(), "%s%% (tolerance: inf)",
                    gapValString.data());
  }

  bool timeless_log = options_mip_->timeless_log;
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "\nSolving report\n");
  if (this->orig_model_->model_name_.length())
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  Model             %s\n",
                 this->orig_model_->model_name_.c_str());
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  Status            %s\n"
               "  Primal bound      %.12g\n"
               "  Dual bound        %.12g\n"
               "  Gap               %s\n",
               utilModelStatusToString(modelstatus_).c_str(), primal_bound_,
               dual_bound_, gapString.data());
  if (!timeless_log)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  P-D integral      %.12g\n",
                 mipdata_->primal_dual_integral.value);
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  Solution status   %s\n", solutionstatus.c_str());
  if (solutionstatus != "-")
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "                    %.12g (objective)\n"
                 "                    %.12g (bound viol.)\n"
                 "                    %.12g (int. viol.)\n"
                 "                    %.12g (row viol.)\n",
                 solution_objective_, bound_violation_, integrality_violation_,
                 row_violation_);
  if (!timeless_log)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  Timing            %.2f (total)\n"
                 "                    %.2f (presolve)\n"
                 "                    %.2f (solve)\n"
                 "                    %.2f (postsolve)\n",
                 timer_.read(), analysis_.mipTimerRead(kMipClockPresolve),
                 analysis_.mipTimerRead(kMipClockSolve),
                 analysis_.mipTimerRead(kMipClockPostsolve));
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  Max sub-MIP depth %d\n"
               "  Nodes             %llu\n"
               "  Repair LPs        %llu (%llu feasible; %llu iterations)\n"
               "  LP iterations     %llu (total)\n"
               "                    %llu (strong br.)\n"
               "                    %llu (separation)\n"
               "                    %llu (heuristics)\n",
               int(max_submip_level), (long long unsigned)mipdata_->num_nodes,
               (long long unsigned)mipdata_->total_repair_lp,
               (long long unsigned)mipdata_->total_repair_lp_feasible,
               (long long unsigned)mipdata_->total_repair_lp_iterations,
               (long long unsigned)mipdata_->total_lp_iterations,
               (long long unsigned)mipdata_->sb_lp_iterations,
               (long long unsigned)mipdata_->sepa_lp_iterations,
               (long long unsigned)mipdata_->heuristic_lp_iterations);

  if (!timeless_log) analysis_.reportMipTimer();

  assert(modelstatus_ != HighsModelStatus::kNotset);
}

// Only called in Highs::runPresolve
void HighsMipSolver::runPresolve(const HighsInt presolve_reduction_limit) {
  mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
  mipdata_->init();
  mipdata_->runPresolve(presolve_reduction_limit);
}

const HighsLp& HighsMipSolver::getPresolvedModel() const {
  return mipdata_->presolvedModel;
}

HighsPresolveStatus HighsMipSolver::getPresolveStatus() const {
  return mipdata_->presolve_status;
}

presolve::HighsPostsolveStack HighsMipSolver::getPostsolveStack() const {
  return mipdata_->postSolveStack;
}

void HighsMipSolver::callbackGetCutPool() const {
  assert(callback_->user_callback);
  assert(callback_->callbackActive(kCallbackMipGetCutPool));
  HighsCallbackDataOut& data_out = callback_->data_out;

  std::vector<double> cut_lower;
  std::vector<double> cut_upper;
  HighsSparseMatrix cut_matrix;

  mipdata_->lp.getCutPool(data_out.cutpool_num_col, data_out.cutpool_num_cut,
                          cut_lower, cut_upper, cut_matrix);

  data_out.cutpool_num_nz = cut_matrix.numNz();
  data_out.cutpool_start = cut_matrix.start_.data();
  data_out.cutpool_index = cut_matrix.index_.data();
  data_out.cutpool_value = cut_matrix.value_.data();
  data_out.cutpool_lower = cut_lower.data();
  data_out.cutpool_upper = cut_upper.data();
  callback_->user_callback(kCallbackMipGetCutPool, "MIP cut pool",
                           &callback_->data_out, &callback_->data_in,
                           callback_->user_callback_data);
}

bool HighsMipSolver::solutionFeasible(
    const HighsLp* lp, const std::vector<double>& col_value,
    const std::vector<double>* pass_row_value, double& bound_violation,
    double& row_violation, double& integrality_violation, HighsCDouble& obj) {
  bound_violation = 0;
  row_violation = 0;
  integrality_violation = 0;
  const double mip_feasibility_tolerance =
      options_mip_->mip_feasibility_tolerance;

  obj = lp->offset_;

  if (kAllowDeveloperAssert) assert(HighsInt(col_value.size()) == lp->num_col_);
  for (HighsInt i = 0; i != lp->num_col_; ++i) {
    const double value = col_value[i];
    obj += lp->col_cost_[i] * value;

    if (lp->integrality_[i] == HighsVarType::kInteger) {
      integrality_violation =
          std::max(fractionality(value), integrality_violation);
    }

    const double lower = lp->col_lower_[i];
    const double upper = lp->col_upper_[i];
    double primal_infeasibility;
    if (value < lower - mip_feasibility_tolerance) {
      primal_infeasibility = lower - value;
    } else if (value > upper + mip_feasibility_tolerance) {
      primal_infeasibility = value - upper;
    } else
      continue;

    bound_violation = std::max(bound_violation, primal_infeasibility);
  }

  // Check row feasibility if there are a positive number of rows.
  //
  // If there are no rows and pass_row_value is nullptr, then
  // row_value_p is also nullptr since row_value is not resized
  if (lp->num_row_ > 0) {
    std::vector<double> row_value;
    if (pass_row_value) {
      if (kAllowDeveloperAssert)
        assert(HighsInt((*pass_row_value).size()) == lp->num_col_);
    } else {
      calculateRowValuesQuad(*lp, col_value, row_value);
    }
    const double* row_value_p =
        pass_row_value ? (*pass_row_value).data() : row_value.data();
    assert(row_value_p);

    for (HighsInt i = 0; i != lp->num_row_; ++i) {
      const double value = row_value_p[i];
      const double lower = lp->row_lower_[i];
      const double upper = lp->row_upper_[i];

      double primal_infeasibility;
      if (value < lower - mip_feasibility_tolerance) {
        primal_infeasibility = lower - value;
      } else if (value > upper + mip_feasibility_tolerance) {
        primal_infeasibility = value - upper;
      } else
        continue;

      row_violation = std::max(row_violation, primal_infeasibility);
    }
  }

  const bool feasible = bound_violation <= mip_feasibility_tolerance &&
                        integrality_violation <= mip_feasibility_tolerance &&
                        row_violation <= mip_feasibility_tolerance;
  return feasible;
}
