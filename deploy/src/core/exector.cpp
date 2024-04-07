#include "exector.h"
#include "core/util/any.h"
#include "engine.h"
#include "cache.h"
#include "core/processor.h"
#include "core/pipeline.h"

namespace gddeploy{

Executor::Executor(const SessionDesc& desc, PriorityThreadPool* tp, int device_id)
    : desc_(desc), tp_(tp), device_id_(device_id) {
    // CHECK(tp);
    // CHECK_GE(device_id, 0) << "[InferServer] [Executor] Device id is less than 0. device id: " << device_id;
    // CHECK_GT(desc_.engine_num, 0u) << "[InferServer] [Executor] Engine number cannot be 0";
    // CHECK(desc_.preproc) << "[InferServer] [Executor] Preprocess cannot be null";

    // init processors
    std::vector<ProcessorPtr> processors;
    ModelPtr model_ptr = std::dynamic_pointer_cast<Model>(desc_.model);
    Pipeline::Instance()->CreatePipeline("", model_ptr, processors);

    // init engines
    auto notify_done_func = [this](Engine* idle) {
        idle_.store(idle);
        dispatch_cond_.notify_one();
    };
    
    engines_.reserve(desc_.engine_num);
    engines_.emplace_back(new Engine(model_ptr, processors, std::move(notify_done_func), tp_));
    for (size_t e_idx = 1; e_idx < desc_.engine_num; ++e_idx) {
        engines_.emplace_back(engines_[0]->Fork());
    }
    idle_.store(engines_[0].get());

    // TODO(dmh): 3 is number of processors, refactor to adjustable
    max_processing_num_ = 4 * desc_.engine_num * 3 * desc_.model->BatchSize();
    // init cache
    if (desc_.strategy == BatchStrategy::DYNAMIC) {
        cache_.reset(new CacheDynamic(desc_.model->BatchSize(), Priority(desc_.priority), desc_.batch_timeout));
    } else if (desc_.strategy == BatchStrategy::STATIC) {
        cache_.reset(new CacheStatic(desc_.model->BatchSize(), Priority(desc_.priority)));
    } else {
        GDDEPLOY_ERROR("[InferServer] [Executor] Unsupported BatchStrategy");
    }
    cache_->Start();

    dispatch_thread_ = std::thread(&Executor::DispatchLoop, this);
}

Executor::~Executor() {
    std::unique_lock<std::mutex> lk(link_mutex_);
    for (auto& session : link_set_) {
        delete session;
    }
    link_set_.clear();
    lk.unlock();
    cache_->Stop();
    // VLOG(1) << "[InferServer] [Executor] " << desc_.name << "] Processed Task:\n\t"
    //         << " | total " << static_cast<uint32_t>(batch_record_.total) << " | batch number " << batch_record_.unit_cnt
    //         << " | average tasks per batch " << batch_record_.total / batch_record_.unit_cnt;
    // dispatch thread won't quit until cache is empty
    dispatch_thread_.join();
    cache_.reset();
    // CHECK(link_set_.empty()) << "[InferServer] [Executor] Should not have any session in destructor";
    idle_.store(nullptr);
    engines_.clear();
}

void Executor::DispatchLoop() noexcept {
    std::unique_lock<std::mutex> dispatch_lk(dispatch_mutex_, std::defer_lock);
    while (true) {
        // get package from cache
        PackagePtr pack = cache_->Pop();
        if (!pack) {
            if (!cache_->Running()) break;
            continue;
        }
        size_t batch_size = pack->data.size();
        batch_record_.unit_cnt += 1;
        batch_record_.total += batch_size;

        // dispatch to engine
        Engine* idle{nullptr};
        if (idle_) {
            idle = idle_.exchange(idle);
        } else {
            // find idle engine
            for (auto& it : engines_) {
                if (it->IsIdle()) {
                    idle = it.get();
                    break;
                }
            }
            if (!idle) {
                dispatch_lk.lock();
                dispatch_cond_.wait(dispatch_lk, [this]() -> bool { return idle_; });
                dispatch_lk.unlock();
                idle = idle_.exchange(idle);
            }
        }
        // GDDEPLOY_INFO("[InferServer] [Executor] {}] dispatch to engine ", desc_.name , idle_);
        idle->Run(std::move(pack));
    }
}

}