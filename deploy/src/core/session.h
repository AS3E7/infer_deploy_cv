#ifndef GDDEPLOY_CORE_SESSION_H_
#define GDDEPLOY_CORE_SESSION_H_

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "cache.h"
#include "core/infer_server.h"
#include "priority.h"
#include "profile.h"
#include "request_ctrl.h"
#include "util/thread_pool.h"
#include "common/logger.h"
#include "engine.h"

namespace gddeploy {

class Executor;
using Executor_t = Executor*;

class Session {
 public:
  Session(const std::string& name, Executor_t executor, bool sync_link, bool show_perf) noexcept
      : name_(name), executor_(executor), running_(true), is_sync_link_(sync_link), show_perf_(show_perf) {
#ifdef CNIS_RECORD_PERF
    profiler_.SetSelfUpdate(false);
    // update and print performance information every 2 second
    perf_timer_.NotifyEvery(2000, [this, show_perf]() {
      profiler_.Update();
      if (show_perf) {
        LOG(INFO) << "[InferServer] [" << name_ << "] Session rps (total): " << profiler_.RequestPerSecond();
        LOG(INFO) << "[InferServer] [" << name_ << "] Session ups (total): " << profiler_.UnitPerSecond();
        LOG(INFO) << "[InferServer] [" << name_ << "] Session rps (realtime): "
                  << profiler_.RequestThroughoutRealtime();
        LOG(INFO) << "[InferServer] [" << name_ << "] Session ups (realtime): "
                  << profiler_.UnitThroughoutRealtime();
      }
    });
#endif
  }

  ~Session() {
    if (running_.load()) {
      running_.store(false);
    }
    auto check = [this]() { return request_list_.empty() && !in_response_.load(); };
    std::unique_lock<std::mutex> lk(request_mutex_);
    if (!check()) {
      GDDEPLOY_INFO("[InferServer] [Session] session {} wait all task done in destructor", name_);
      sync_cond_.wait(lk, check);
    }
    lk.unlock();

#ifdef CNIS_RECORD_PERF
    // stop print perf
    if (!perf_timer_.Idle()) {
      perf_timer_.Cancel();
      if (show_perf_) recorder_.PrintPerformance(name_);
    }
#endif
  }

  /* ---------------- Observer -------------------*/
  const std::string& GetName() const noexcept { return name_; }
  Executor_t GetExecutor() const noexcept { return executor_; }
  Observer* GetRawObserver() const noexcept { return observer_.get(); }
  bool IsSyncLink() const noexcept { return is_sync_link_; }
  /* -------------- Observer END -----------------*/

  void SetObserver(std::shared_ptr<Observer> observer) noexcept { 
    // observer_ = std::move(observer); 
    observer_ = observer; 
  }

  RequestControl* Send(PackagePtr&& data, std::function<void(Status, PackagePtr)>&& notifier) noexcept;

  void CheckAndResponse(const RequestControl* caller) noexcept;

  void WaitTaskDone(const std::string& tag) noexcept;

  void DiscardTask(const std::string& tag) noexcept;

#ifdef CNIS_RECORD_PERF
  const std::map<std::string, LatencyStatistic>& GetPerformance() const noexcept { return recorder_.GetPerformance(); }
  ThroughoutStatistic GetThroughout(const std::string& tag) noexcept { return profiler_.Summary(tag); }
  ThroughoutStatistic GetThroughout() noexcept { return profiler_.Summary(); }
#endif

 private:
  std::string name_;
  Executor_t executor_;
  std::mutex request_mutex_;
  std::condition_variable sync_cond_;
  std::list<RequestControl*> request_list_;
  std::shared_ptr<Observer> observer_{nullptr};

#ifdef CNIS_RECORD_PERF
  // performance statistics
  LatencyRecorder recorder_;
  Timer perf_timer_;
  TagSetProfiler profiler_;
#endif

  int64_t request_id_{0};
  std::atomic<bool> running_{false};
  std::atomic<bool> in_response_{false};
  bool is_sync_link_{false};
  bool show_perf_{false};
};  // class Session

class Executor {
 public:
    Executor(const SessionDesc& desc, PriorityThreadPool* tp, int device_id);

    ~Executor();

    void Link(Session_t session) noexcept {
        std::unique_lock<std::mutex> lk(link_mutex_);
        GDDEPLOY_INFO("[InferServer] [Session] Executor {}] link session ", desc_.name , session->GetName());
        link_set_.insert(session);
    }

    void Unlink(Session_t session) noexcept {
        std::unique_lock<std::mutex> lk(link_mutex_);
        if (link_set_.count(session)) {
            GDDEPLOY_INFO("[InferServer] [Session] Executor {}] unlink session ", desc_.name ,session->GetName());
            link_set_.erase(session);
        } else {
            GDDEPLOY_WARN("[InferServer] [Session] Unlink session, but it is not found in this executor");
        }
    }

    bool WaitIfCacheFull(int timeout) noexcept {
        auto idle_pred = [this]() {
            // printf("processing_unit_.load():%d, %d\n", processing_unit_.load(), processing_req_.load());
            return processing_unit_.load() < max_processing_num_ && processing_req_.load() < max_processing_num_;
        };
        if (!idle_pred()) {
            std::unique_lock<std::mutex> lk(limit_mutex_);
            if (timeout > 0) {
                return limit_cond_.wait_for(lk, std::chrono::milliseconds(timeout), idle_pred);
            } else {
                GDDEPLOY_WARN("[InferServer] [WaitIfCacheFull] Wait for cache not full");
                limit_cond_.wait(lk, idle_pred);
                GDDEPLOY_WARN("[InferServer] [WaitIfCacheFull] Wait for cache not full done");
            }
        }
        return true;
    }

    void FlushCache() noexcept {
        cache_->Flush();
    }

    /* ------------------- Observer --------------------- */
    size_t GetSessionNum() noexcept {
        std::unique_lock<std::mutex> lk(link_mutex_);
        return link_set_.size();
    }
    
    ModelInfoPtr GetModel() noexcept { return desc_.model; };
    const SessionDesc& GetDesc() const noexcept { return desc_; }
    const Priority& GetPriority() const noexcept { return cache_->GetPriority(); }
    std::string GetName() const noexcept { return desc_.name; }
    uint32_t GetEngineNum() const noexcept { return desc_.engine_num; }
    PriorityThreadPool* GetThreadPool() const noexcept { return tp_; }
    /* ----------------- Observer END ------------------- */

    void ReleaseCount(uint32_t data_num) {
        processing_unit_.fetch_sub(data_num);
        processing_req_.fetch_sub(1);
        limit_cond_.notify_one();
    }

    bool Upload(PackagePtr&& pack, RequestControl* ctrl) noexcept {
        uint32_t data_num = pack->data.size();
        processing_unit_.fetch_add(data_num);
        processing_req_.fetch_add(1);
        return cache_->Push(std::forward<PackagePtr>(pack));
    }

    void DispatchLoop() noexcept;

 private:
    SessionDesc desc_;
    PriorityThreadPool* tp_;
    std::unique_ptr<CacheBase> cache_;

    // manage link
    std::set<Session_t> link_set_;
    std::mutex link_mutex_;

    // dispatch to engine
    std::vector<std::unique_ptr<Engine>> engines_;
    std::atomic<Engine*> idle_{nullptr};
    std::thread dispatch_thread_;
    std::mutex dispatch_mutex_;
    std::condition_variable dispatch_cond_;

    // processing number limit
    std::mutex limit_mutex_;
    std::condition_variable limit_cond_;
    std::atomic<uint32_t> processing_unit_{0};
    std::atomic<uint32_t> processing_req_{0};
    uint32_t max_processing_num_;

    LatencyStatistic batch_record_;
    std::atomic_bool running_{false};
    int device_id_;
};  // class Executor



}  // namespace gddeploy

#endif  // GDDEPLOY_CORE_SESSION_H_