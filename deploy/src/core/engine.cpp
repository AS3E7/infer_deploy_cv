/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include "engine.h"

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <future>

#include "profile.h"
#include "request_ctrl.h"
#include "session.h"
#include "core/device.h"

namespace gddeploy {

void TaskNode::Execute(PackagePtr pack) {
  Status s;
#if defined(CNIS_RECORD_PERF) && (!defined(NDEBUG))
  auto before_lock = Clock::Now();
#endif
  std::unique_lock<std::mutex> lk = processor_->Lock();
  const std::string& type_name = processor_->TypeName();
#ifdef CNIS_RECORD_PERF
  auto start = Clock::Now();
#endif
  // s = processor_->Process(pack);
  // 使用std::packaged_task做异步处理，然后使用future等待结果返回
  std::function<int()> func = [this, pack](){
    const std::string& type_name = this->processor_->TypeName();
    // auto t0 = std::chrono::high_resolution_clock::now();
    int ret = (int)this->processor_->Process(pack);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("%s time: %d us\n", type_name.c_str(), std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    return ret;
  };
  std::packaged_task<int()> pack_task(std::bind(func));
  
  std::future<int> res = pack_task.get_future();
  
  SingerTask task;
  task.task_func = std::make_shared<std::packaged_task<int()>>(std::move(pack_task));
  // task.func = func;
  task.priority = pack->priority;

  // Device *dev = DeviceManager::Instance()->GetDevice("SOPHGO", "SE5");
  // type_name是类似于bmnn_preproc的字符串，需要切割_，取preproc
  std::string type = type_name.substr(type_name.find_last_of("_") + 1);
  if (-1 == device_->AddTask(type, task)){
    // s = processor_->Process(pack);
    // auto t0 = std::chrono::high_resolution_clock::now();
    task.task_func->operator()();
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("%s time: %d us\n", type.c_str(), std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
  }

  res.wait();
  s = (Status)res.get();

  lk.unlock();
  
#ifdef CNIS_RECORD_PERF
  auto end = Clock::Now();
  pack->perf[type_name] = Clock::Duration(start, end);
  #ifndef NDEBUG
  pack->perf["-WaitLock-" + type_name] = Clock::Duration(before_lock, start);
  #endif
#endif
  if (s != Status::SUCCESS) {
    GDDEPLOY_ERROR("[InferServer] [TaskNode] processor [{}] execute failed", type_name);
    for (auto& it : pack->data) {
      it->ctrl->ProcessFailed(s);
    }
    done_notifier_();
  } else {
    GDDEPLOY_INFO("[InferServer] [TaskNode] Transmit data for {}", type_name);
    Transmit(std::move(pack));
  }
}


// TODO: 以后这里需要根据processor运行的硬件IP智能分配，
// 有多少个IP运行就需要有多少个engine运行，保证每个IP都有对于的engine在运行，但对于每个pipeline来说还是原先顺序
// 同样，如果有一样的IP，但是有多个数量，则需要创建多个worker
void TaskNode::Transmit(PackagePtr&& pack) noexcept {
  if (downnode_) {
    // start next processor
    pack->priority = Priority::Next(pack->priority);
    // TODO(dmh): copy TaskNode for each task transmit?
    tp_->VoidPush(pack->priority, &TaskNode::Execute, downnode_, std::forward<PackagePtr>(pack));
  } else {
    std::map<std::string, float> perf{};
#ifdef CNIS_RECORD_PERF
    for (auto& it : pack->perf) {
      perf[it.first] = it.second / pack->data.size();
    }
#endif
    // tail of process, response to user
    for (auto& it : pack->data) {
      // SUCCESS flag won't cover errors happended before
      it->ctrl->ProcessDone(Status::SUCCESS, it, it->index, std::move(perf));
    }
    done_notifier_();
  }
}

Engine::Engine(ModelPtr model, std::vector<std::shared_ptr<Processor>> processors, NotifyDoneFunc&& done_func,
               PriorityThreadPool* tp)
    : done_notifier_(std::move(done_func)), tp_(tp) {
    model_ = model;
    ModelPropertiesPtr model_properties = model_->GetModelInfoPriv();
    std::string manu = model_properties->GetProductType();
    std::string chip = model_properties->GetChipType();

  // 全局只执行一次
  // static std::once_flag init_flag;
  // std::call_once(init_flag, [manu, chip]() {
  //   // Device创建workers
  //   printf("####################!!!!!!!!!!!!engine create device\n");
  //   // Device *dev = DeviceManager::Instance()->GetDevice(manu, chip);
  //   // dev->CreateWorkers();
  // });
  Device *dev = DeviceManager::Instance()->GetDevice(manu, chip);
  
  nodes_.reserve(processors.size());
  for (size_t idx = 0; idx < processors.size(); ++idx) {
    nodes_.emplace_back(
        dev,
        processors[idx],
        [this]() {
          --task_num_;
          done_notifier_(this);
        },
        tp_);
  }
  for (size_t idx = 0; idx < nodes_.size() - 1; ++idx) {
    nodes_[idx].Link(&nodes_[idx + 1]);
  }
}

std::unique_ptr<Engine> Engine::Fork() {
  auto* fork_engine = new Engine;
  fork_engine->tp_ = tp_;
  fork_engine->done_notifier_ = done_notifier_;
  fork_engine->nodes_.reserve(nodes_.size());
  for (auto& it : nodes_) {
    fork_engine->nodes_.emplace_back(it.Fork([fork_engine]() {
      --fork_engine->task_num_;
      fork_engine->done_notifier_(fork_engine);
    }));
  }
  for (size_t idx = 0; idx < fork_engine->nodes_.size() - 1; ++idx) {
    fork_engine->nodes_[idx].Link(&fork_engine->nodes_[idx + 1]);
  }
  return std::unique_ptr<Engine>(fork_engine);
}

}  // namespace gddeploy
