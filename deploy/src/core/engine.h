#ifndef GDDEPLOY_ENGINE_H
#define GDDEPLOY_ENGINE_H

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "processor.h"
#include "core/infer_server.h"
#include "util/thread_pool.h"
#include "core/model.h"
#include "core/device.h"
#include "priority.h"

namespace gddeploy
{
class Engine;
/*
配置信息
UserConfig：用户额外设置参数
ModelConfig：解析模型得到的模型参数信息
*/
class TaskNode {
public:
    using Notifier = std::function<void()>;
    TaskNode(Device *dev, std::shared_ptr<Processor> processor, Notifier&& done_notifier, PriorityThreadPool* tp) noexcept
        : device_(dev), processor_(processor), done_notifier_(std::forward<Notifier>(done_notifier)), tp_(tp) {}

    TaskNode Fork(Notifier&& done_notifier) {
    auto fork_proc = processor_->Fork();
    if (!fork_proc) throw std::runtime_error("Fork processor failed: " + processor_->TypeName());
    // return TaskNode(std::move(fork_proc), std::forward<Notifier>(done_notifier), tp_);
    return TaskNode(device_, std::move(fork_proc), std::forward<Notifier>(done_notifier), tp_);
    }

    void Execute(PackagePtr pack);

    void Transmit(PackagePtr&& data) noexcept;

    void Link(TaskNode* node) noexcept { downnode_ = node; }

private:
    Device *device_;
    TaskNode() = delete;
    std::shared_ptr<Processor> processor_;
    Notifier done_notifier_;
    PriorityThreadPool* tp_;
    TaskNode* downnode_{nullptr};
};

class Engine {
public:
    using NotifyDoneFunc = std::function<void(Engine*)>;
    Engine() = default;
    Engine(ModelPtr model, std::vector<std::shared_ptr<Processor>> processors, NotifyDoneFunc&& done_func, PriorityThreadPool* tp);
    ~Engine() {
        while (task_num_.load()) {
            // wait for all task done
        }
    }

    std::unique_ptr<Engine> Fork();

    void Run(PackagePtr&& package) noexcept {
        ++task_num_;
        tp_->VoidPush(package->priority, &TaskNode::Execute, &nodes_[0], std::forward<PackagePtr>(package));
    }

    bool IsIdle() noexcept { return task_num_.load() < nodes_.size(); }

    size_t MaxLoad() noexcept { return nodes_.size(); }

private:
    ModelPtr model_;
    std::vector<TaskNode> nodes_;
    NotifyDoneFunc done_notifier_;
    PriorityThreadPool* tp_;
    std::atomic<uint32_t> task_num_{0};
};  // class Engine


} // namespace gddeploy

#endif