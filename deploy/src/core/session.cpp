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

#include "session.h"

#include <list>
#include <string>
#include <utility>
#include <vector>

#include "core/infer_server.h"
#include "core/processor.h"
#include "engine.h"
#include "profile.h"
#include "exector.h"

namespace gddeploy {

// constexpr is not inline in C++11
constexpr uint32_t Profiler::period_interval_;

void Session::WaitTaskDone(const std::string& tag) noexcept {
    GDDEPLOY_INFO("[InferServer] [Session] session {} wait [{}] task done", name_, tag);
    std::vector<std::string> match = {tag};
    if (!executor_->GetDesc().batch_timeout) executor_->FlushCache();
    std::unique_lock<std::mutex> lk(request_mutex_);
    auto last = std::find_first_of(request_list_.rbegin(), request_list_.rend(), match.begin(), match.end(),
                                    [](RequestControl* c, const std::string& t) { return c->Tag() == t; });
    if (last != request_list_.rend()) {
        std::future<void> flag = (*last)->ResponseDonePromise();
        lk.unlock();
        flag.get();
    } else {
        lk.unlock();
    }
    // Task is popped from request_list before response.
    // If there are `tag` task in reponse while find last `tag` task,
    // WaitTaskDone may quit before `tag` task finishing response.
    // Wait until response done to avoid that.
    while (in_response_.load()) {}
#ifdef CNIS_RECORD_PERF
    profiler_.RemoveTag(tag);
#endif
}

void Session::DiscardTask(const std::string& tag) noexcept {
    GDDEPLOY_INFO("[InferServer] [Session] session {} discard [{}] task", name_, tag);
    if (!executor_->GetDesc().batch_timeout) executor_->FlushCache();
    std::unique_lock<std::mutex> lk(request_mutex_);
    std::for_each(request_list_.begin(), request_list_.end(), [&tag](RequestControl* it) {
        if (it->Tag() == tag) {
            it->Discard();
        }
    });
#ifdef CNIS_RECORD_PERF
    profiler_.RemoveTag(tag);
#endif
}

RequestControl* Session::Send(PackagePtr&& pack, std::function<void(Status, PackagePtr)>&& response) noexcept {
    if (!running_.load()) {
        GDDEPLOY_ERROR("[InferServer] [Session] This session is not running [{}]", name_);
        return nullptr;
    }

    if (pack->predict_io && pack->predict_io->HasValue()) {
        if (executor_->GetDesc().strategy != BatchStrategy::STATIC) {
            GDDEPLOY_ERROR("[InferServer] [Session] Input continuous data to skip preprocess is only supported under" \
                        " BatchStrategy::STATIC");
            return nullptr;
        }
        if (pack->data.size() > executor_->GetModel()->BatchSize()) {
            GDDEPLOY_ERROR("[InferServer] [Session] Input continuous data to skip preprocess is only supported when \
                         data number <= model batch size");
            return nullptr;
        }
    }
    // since cannot classify data size from continuous data,
    // we use batch_size set in package instead of size of pack->data
    size_t data_size = pack->data.size();

#ifdef CNIS_RECORD_PERF
    profiler_.RequestStart(pack->tag);
#endif
    std::unique_lock<std::mutex> lk(request_mutex_);
    RequestControl* ctrl =
        new RequestControl(std::move(response), std::bind(&Session::CheckAndResponse, this, std::placeholders::_1),
                            pack->tag, request_id_++, data_size);
#ifdef CNIS_RECORD_PERF
    ctrl->BeginRecord();
#endif
    for (size_t index = 0; index < pack->data.size(); ++index) {
    pack->data[index]->ctrl = ctrl;
    pack->data[index]->index = index;
    }
    request_list_.push_back(ctrl);
    lk.unlock();

    if (data_size) {
        if (executor_->Upload(std::move(pack), ctrl))
        GDDEPLOY_INFO("[InferServer] [Session] Cache should be running");
    } else {
        GDDEPLOY_INFO("[InferServer] [Session] session: {} | No data in package with tag [{}]", name_, pack->tag);

        if (executor_->Upload(std::move(pack), ctrl))
            GDDEPLOY_INFO("[InferServer] [Session] Cache should be running");
        CheckAndResponse(ctrl);
    }
    return ctrl;
}

void Session::CheckAndResponse(const RequestControl* caller) noexcept {
    std::unique_lock<std::mutex> lk(request_mutex_);
    RequestControl* ctrl;

    // check request finished processing
    if (request_list_.empty()) {
        GDDEPLOY_INFO("[InferServer] [Session] No request in this Session " , name_ );
        // notify blocked thread by destructor
        sync_cond_.notify_one();
        return;
    }
    ctrl = request_list_.front();
    if (caller != ctrl && !ctrl->IsProcessFinished()) {
        return;
    }

    bool expected = false;
    if (!in_response_.compare_exchange_strong(expected, true, std::memory_order_release, std::memory_order_relaxed)) {
        return;
    }
    request_list_.pop_front();
    lk.unlock();
    int64_t priority = Priority::Offset(executor_->GetPriority().Get(-ctrl->RequestId()), 5);
    executor_->GetThreadPool()->VoidPush(priority, [ctrl, this] {
        auto next = ctrl;
        do {
        #ifdef CNIS_RECORD_PERF
            profiler_.RequestEnd(next->Tag(), next->DataNum());
        #endif
            if (!next->IsDiscarded()) {
        #ifdef CNIS_RECORD_PERF
            for (auto& it : next->Performance()) {
                recorder_.RecordPerformance(it.first, next->DataNum(), it.second);
            }
            recorder_.RecordPerformance("RequestLatency", 1, next->EndRecord());
        #endif
                next->Response();
            }
            executor_->ReleaseCount(next->DataNum());
            delete next;
            next = nullptr;

            std::unique_lock<std::mutex> lk(request_mutex_);
            if (request_list_.empty()) {
                in_response_.store(false);
                sync_cond_.notify_one();
                return;
            }
            next = request_list_.front();
            if (next->IsProcessFinished()) {
                request_list_.pop_front();
            } else {
                next = nullptr;
            }
        } while (next);
        in_response_.store(false);
    });
}

}  // namespace gddeploy
