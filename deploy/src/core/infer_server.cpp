#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sstream>
#include <unistd.h>

// #include "core/infer_server.h"
#include "session.h"
#include "core/processor.h"
#include "core/util/any.h"
#include "core/model.h"

#include "util/env.h"
#include "util/thread_pool.h"

namespace gddeploy {
class InferServerPrivate {
 public:
  static InferServerPrivate* Instance(int device_id) {
    // each device has a singleton instance
    static std::mutex map_mutex;
    std::unique_lock<std::mutex> lk(map_mutex);
    static std::unordered_map<int, std::unique_ptr<InferServerPrivate>> server_map;
    if (server_map.find(device_id) == server_map.end()) {
      if (device_id < 0) {
        return nullptr;
      }
      server_map.emplace(device_id, std::unique_ptr<InferServerPrivate>(new InferServerPrivate(device_id)));
    }
    return server_map[device_id].get();
  }

  InferServerPrivate(InferServerPrivate&&) = default;
  InferServerPrivate& operator=(InferServerPrivate&&) = default;
  ~InferServerPrivate() = default;

  bool ExistExecutor(Executor_t executor) noexcept {
    std::unique_lock<std::mutex> lk(executor_map_mutex_);
    return executor_map_.count(executor->GetName());
  }

  Executor_t CreateExecutor(const SessionDesc& desc) noexcept {
    std::ostringstream ss;
    ss << desc.model->GetKey();// << "_" << desc.preproc->TypeName() << "_" << desc.postproc->TypeName();
    std::string executor_name = ss.str();
    std::unique_lock<std::mutex> lk(executor_map_mutex_);
    if (executor_map_.count(executor_name)) { //查找map里面是否已经创建，创建了就直接返回，否则新建
      GDDEPLOY_INFO("[InferServer] [CreateExecutor] Executor already exist: {}", executor_name);
      return executor_map_[executor_name].get();
    }
    GDDEPLOY_INFO("[InferServer] [CreateExecutor] Create executor: {}", executor_name);
    try {
      SessionDesc executor_desc = desc;
      executor_desc.name = executor_name;
      std::unique_ptr<Executor> executor_up{new Executor(std::move(executor_desc), tp_.get(), device_id_)};
      Executor_t executor = executor_up.get();
      /* executor_map_.insert({executor_name, std::move(executor_up)}); */
      executor_map_[executor_name].swap(executor_up);
      lk.unlock();
      std::unique_lock<std::mutex> tp_lk(tp_mutex_);
      size_t thread_num = tp_->Size();
      static size_t max_thread_num = 3 * GetCpuCoreNumber();
      if (thread_num < max_thread_num) {
        tp_->Resize(std::min(thread_num + 4 * desc.engine_num, max_thread_num));
      }
      tp_lk.unlock();
      return executor;
    } catch (std::runtime_error& e) {
      GDDEPLOY_ERROR("[InferServer] [CreateExecutor] error occurs, error message: ", e.what());
      return nullptr;
    }
  }

  void CheckAndDestroyExecutor(Session_t session, Executor_t executor) noexcept {
    // CHECK(executor) << "[InferServer] Executor is null!";
    // CHECK(session) << "[InferServer] Session is null!";
    std::unique_lock<std::mutex> lk(executor_map_mutex_);
    executor->Unlink(session);
    delete session;

    // delete executor while there's no session linked to it
    if (!executor->GetSessionNum()) {
      auto name = executor->GetName();
      if (executor_map_.count(name)) {
        auto th_num = 4 * executor->GetEngineNum();
        GDDEPLOY_INFO("[InferServer] [CheckAndDestroyExecutor] Destroy executor: {}", name);
        executor_map_.erase(name);
        lk.unlock();
        // shrink to fit task load
        std::unique_lock<std::mutex> tp_lk(tp_mutex_);
        if (tp_->IdleNumber() > th_num) {
          GDDEPLOY_INFO("[InferServer] [CheckAndDestroyExecutor] Reduce thread in pool after destroy executor");
          tp_->Resize(tp_->Size() - th_num);
        }
        tp_lk.unlock();
      } else {
            GDDEPLOY_ERROR("[InferServer] [CheckAndDestroyExecutor] Executor does not belong to this InferServer");
      }
    }
  }

  PriorityThreadPool* GetThreadPool() noexcept { return tp_.get(); }
  int GetDeviceId() const noexcept { return device_id_; }

 private:
  explicit InferServerPrivate(int device_id) noexcept : device_id_(device_id) {
    tp_.reset(new PriorityThreadPool([device_id]() -> bool { return SetCurrentDevice(device_id); }));
  }
  InferServerPrivate(const InferServerPrivate&) = delete;
  InferServerPrivate& operator=(const InferServerPrivate&) = delete;

  std::map<std::string, std::unique_ptr<Executor>> executor_map_;
  std::mutex executor_map_mutex_;
  std::mutex tp_mutex_;
  std::unique_ptr<PriorityThreadPool> tp_{nullptr};
  int device_id_;
};  // class InferServerPrivate
}  // namespace gddeploy

using namespace gddeploy;

std::string ToString(BatchStrategy s) noexcept {
  switch (s) {
    case BatchStrategy::DYNAMIC:
      return "BatchStrategy::DYNAMIC";
    case BatchStrategy::STATIC:
      return "BatchStrategy::STATIC";
    case BatchStrategy::SEQUENCE:
      return "BatchStrategy::SEQUENCE";
    case BatchStrategy::STRATEGY_COUNT:
      return "BatchStrategy::STRATEGY_COUNT";
    default:
      return "Unknown";
  }
}

InferServer::InferServer(int device_id) noexcept { priv_ = InferServerPrivate::Instance(device_id); }

Session_t InferServer::CreateSession(SessionDesc desc, std::shared_ptr<Observer> observer) noexcept {
//   CHECK(desc.model) << "[InferServer] [CreateSession] model is null!";
//   CHECK(desc.preproc) << "[InferServer] [CreateSession] preproc is null!";

  // won't check postproc, use empty postproc function and output ModelIO by default
  // if (!desc.postproc) {
  //   // LOG(WARNING) << "[InferServer] [CreateSession] Postprocessor not set, use empty postprocessor by default";
  //   desc.postproc = std::make_shared<Postprocessor>();
  // }

  Executor_t executor = priv_->CreateExecutor(desc);
  if (!executor) return nullptr;

  auto* session = new Session(desc.name, executor, !(observer), desc.show_perf);
  if (observer) {
    // async link
    // session->SetObserver(std::move(observer));
    session->SetObserver(observer);
  }
  executor->Link(session);
  return session;
}

bool InferServer::DestroySession(Session_t session) noexcept {
//   CHECK(session) << "[InferServer] [DestroySession] Session is null!";
  Executor_t executor = session->GetExecutor();
  if (!priv_->ExistExecutor(executor)) {
    // LOG(WARNING) << "[InferServer] [DestroySession] Session does not belong to this InferServer";
    return false;
  }

  priv_->CheckAndDestroyExecutor(session, executor);
  return true;
}

bool InferServer::Request(Session_t session, PackagePtr input, any user_data, int timeout) noexcept {
//   CHECK(session) << "[InferServer] [Request] Session is null!";
//   CHECK(input) << "[InferServer] [Request] Input is null!";
  if (session->IsSyncLink()) {
    GDDEPLOY_ERROR("[InferServer] [Request] Sync LinkHandle cannot be invoked with async api");
    return false;
  }
  if (!session->GetExecutor()->WaitIfCacheFull(timeout)) {
    // LOG(WARNING) << "[InferServer] [Request] Session [" << session->GetName() << "] is busy, request timeout";
    return false;
  }

  return session->Send(std::move(input), std::bind(&Observer::Response, session->GetRawObserver(),
                                                   std::placeholders::_1, std::placeholders::_2, std::move(user_data)));
}

bool InferServer::RequestSync(Session_t session, PackagePtr input, Status* status, PackagePtr output,
                              int timeout) noexcept {
//   CHECK(session) << "[InferServer] [RequestSync] Session is null!";
//   CHECK(input) << "[InferServer] [RequestSync] Input is null!";
//   CHECK(output) << "[InferServer] [RequestSync] Output is null!";
//   CHECK(status) << "[InferServer] [RequestSync] Status is null!";
  if (!session->IsSyncLink()) {
    GDDEPLOY_ERROR("[InferServer] [RequestSync] Async Session cannot be invoked with sync api");
    return false;
  }
  if (input->data.empty()) {
    GDDEPLOY_ERROR("[InferServer] [RequestSync] Pass empty package is not supported");
    *status = Status::INVALID_PARAM;
    return false;
  }

  std::promise<void> done;
  std::future<void> flag = done.get_future();

  auto wait_start = std::chrono::steady_clock::now();
  if (!session->GetExecutor()->WaitIfCacheFull(timeout)) {
    // LOG(WARNING) << "[InferServer] [RequestSync] Session [" << session->GetName() << "] is busy,"
    //              << " request timeout";
    *status = Status::TIMEOUT;
    return false;
  }

  if (timeout > 0) {
    std::chrono::duration<double, std::milli> wait_time = std::chrono::steady_clock::now() - wait_start;
    timeout = timeout - wait_time.count();
    if (timeout < 1) {
    //   LOG(WARNING) << "[InferServer] [RequestSync] Session [" << session->GetName() << "] is busy,"
    //                << " request timeout";
      *status = Status::TIMEOUT;
      return false;
    }
  }

  // FIXME(dmh): maybe data race here
  // thread1: timeout ->          -> discard -> status and output deleted in user space
  // thread2:         -> response                                                       -> *output = *data -> boom
  RequestControl* ctrl = session->Send(std::move(input), [&output, status, &done](Status s, PackagePtr data) {
    *status = s;
    *output = *data;
    done.set_value();
  });
  if (!ctrl) return false;
  if (timeout > 0) {
    if (flag.wait_for(std::chrono::milliseconds(timeout)) == std::future_status::timeout) {
      ctrl->Discard();
      *status = Status::TIMEOUT;
    //   LOG(WARNING) << "[InferServer] [RequestSync] Process timeout, discard this request";
    }
  } else {
    flag.wait();
  }
  return true;
}

ModelInfoPtr InferServer::GetModel(Session_t session) noexcept {
//   CHECK(session) << "[InferServer] [GetModel] Session is null!";
  return session->GetExecutor()->GetModel();
}

void InferServer::WaitTaskDone(Session_t session, const std::string& tag) noexcept {
//   CHECK(session) << "[InferServer] [WaitTaskDone] Session is null!";
  session->WaitTaskDone(tag);
}

void InferServer::DiscardTask(Session_t session, const std::string& tag) noexcept {
//   CHECK(session) << "[InferServer] [DiscardTask] Session is null!";
  session->DiscardTask(tag);
}

bool InferServer::SetModelDir(const std::string& model_dir) noexcept {
  // check whether model dir exist
  if (access(model_dir.c_str(), F_OK) == 0) {
    ModelManager::Instance()->SetModelDir(model_dir);
    return true;
  }
  return false;
}

#ifdef CNIS_USE_MAGICMIND
ModelInfoPtr InferServer::LoadModel(const std::string& model_uri, const std::vector<Shape>& in_shape) noexcept {
  return ModelManager::Instance()->Load(model_uri, in_shape);
}

ModelInfoPtr InferServer::LoadModel(void* mem_cache, size_t size, const std::vector<Shape>& in_shape) noexcept {
  return ModelManager::Instance()->Load(mem_cache, size, in_shape);
}
#else
ModelInfoPtr InferServer::LoadModel(const std::string& pattern1, const std::string& pattern2, std::string param) noexcept {
  return ModelManager::Instance()->Load(pattern1, pattern2, param);
}
#endif
bool InferServer::UnloadModel(ModelInfoPtr model) noexcept { return ModelManager::Instance()->Unload(std::move(model)); }

void InferServer::ClearModelCache() noexcept { ModelManager::Instance()->ClearCache(); }

#ifdef CNIS_RECORD_PERF
std::map<std::string, LatencyStatistic> InferServer::GetLatency(Session_t session) const noexcept {
  return session->GetPerformance();
}

ThroughoutStatistic InferServer::GetThroughout(Session_t session) const noexcept { return session->GetThroughout(); }

ThroughoutStatistic InferServer::GetThroughout(Session_t session, const std::string& tag) const noexcept {
  return session->GetThroughout(tag);
}
#else
std::map<std::string, LatencyStatistic> InferServer::GetLatency(Session_t session) const noexcept { return {}; }
ThroughoutStatistic InferServer::GetThroughout(Session_t session) const noexcept { return {}; }
ThroughoutStatistic InferServer::GetThroughout(Session_t session, const std::string& tag) const noexcept { return {}; }
#endif


