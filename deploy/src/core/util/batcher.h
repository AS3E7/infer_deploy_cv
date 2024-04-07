/*************************************************************************
 * Copyright (C) 2020 by Cambricon, Inc. All rights reserved
 *
 * This source code is licensed under the Apache-2.0 license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * A part of this source code is referenced from TensorFlow project.
 * https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/util/batch_util.cc
 *
 * Copyright (C) TensorFlow.
 *
 * This source code is licensed under the Apache-2.0 license found in the
 * LICENSE file in the root directory of this source tree.
 *
 *************************************************************************/

#ifndef GDDEPLOY_UTIL_BATCHER_H_
#define GDDEPLOY_UTIL_BATCHER_H_

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "timer.h"

namespace gddeploy {
// TODO: 以后这里要改为auto batch
template <class item_type>
class Batcher {
 public:
  using notifier_type = std::function<void(std::vector<item_type>&&)>;

  // timeout == 0 means no timeout
  Batcher(notifier_type notifier, uint32_t timeout, uint32_t batch_size)
      : notifier_(notifier), timeout_(timeout), batch_size_(batch_size) {
    // CHECK(batch_size) << "batch size is 0!";
    GDDEPLOY_INFO("batcher] -------batch timeout {} ms", timeout_);
    GDDEPLOY_INFO("batcher] -------batch size {}",  batch_size_);
    cache_.reserve(batch_size_);
    first_item_.store(true);
  }

  void AddItem(const item_type& item) {
    std::unique_lock<std::mutex> lk(cache_mutex_);
    if (timeout_ && first_item_.load()) {
      timer_.Cancel();
      timer_.NotifyAfter(timeout_, &Batcher<item_type>::Emit, this);
      first_item_.store(false);
    }
    cache_.emplace_back(item);
    if (cache_.size() > batch_size_ - 1) {
      Notify(std::move(lk));
    }
  }

  void AddItem(item_type&& item) {
    std::unique_lock<std::mutex> lk(cache_mutex_);
    if (timeout_ && first_item_.load()) {
      timer_.Cancel();
      timer_.NotifyAfter(timeout_, &Batcher<item_type>::Emit, this);
      first_item_.store(false);
    }
    cache_.emplace_back(std::forward<item_type>(item));
    if (cache_.size() > batch_size_ - 1) {
      Notify(std::move(lk));
    }
  }

  size_t Size() noexcept {
    std::unique_lock<std::mutex> lk(cache_mutex_);
    return cache_.size();
  }

  void Emit() {
    Notify(std::unique_lock<std::mutex>(cache_mutex_));
  }

 private:
  void Notify(std::unique_lock<std::mutex> lk) {
    if (cache_.empty()) {
      return;
    }
    std::vector<item_type> tmp_cache;
    tmp_cache.swap(cache_);
    first_item_.store(true);
    cache_.reserve(batch_size_);
    lk.unlock();

    // VLOG(5) << "emit a batch, batch_size: " << tmp_cache.size();
    if (notifier_) {
      notifier_(std::move(tmp_cache));
    } else {
      GDDEPLOY_INFO("Batcher donot have notifier, do nothing");
    }
  }

  Batcher() = delete;
  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;

  std::vector<item_type> cache_;
  std::mutex cache_mutex_;
  notifier_type notifier_;
  Timer timer_;
  uint32_t timeout_;
  uint32_t batch_size_;
  std::atomic<bool> first_item_;
};  // class Batcher

}  // namespace gddeploy

#endif  // GDDEPLOY_UTIL_BATCHER_H_
