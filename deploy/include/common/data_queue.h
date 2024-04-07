#pragma once

#include <iostream>
#include <mutex>
#include <queue>
#include <vector>
#include <condition_variable>

#define MAX_IMAGE_QUEUE 16

template <class T>
class SafeQueue
{
private:
    std::queue<T> queue_;
    std::mutex queue_mux_;
    std::condition_variable queue_cv_;

public:
    SafeQueue(){};

    bool empty()
    {
        std::lock_guard<std::mutex> lk(queue_mux_);
        return queue_.empty();
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lk(queue_mux_);
        return queue_.size();
    }

    T front()
    {
        std::lock_guard<std::mutex> lk(queue_mux_);
        return queue_.front();
    }

    void pop()
    {
        std::lock_guard<std::mutex> lk(queue_mux_);
        return queue_.pop();
    }

    T wait_for_data()
    {
        std::unique_lock<std::mutex> lk(queue_mux_);
        queue_cv_.wait(lk, [&] { return !queue_.empty(); });
        auto t = queue_.front();
        queue_.pop();
        lk.unlock();
        return t;
    }

    void push(T t)
    {
        std::lock_guard<std::mutex> lk(queue_mux_);
        queue_.push(t);
        queue_cv_.notify_one();
        return;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lk(queue_mux_);
        while (!queue_.empty()) queue_.pop();
        queue_cv_.notify_one();
        return;
    }
};

template <class T>
class DoubleCacheQueue
{
private:
    std::string id_;
    SafeQueue<T> pre_queue_;
    SafeQueue<T> post_queue_;

public:
    DoubleCacheQueue(std::string id) : id_(id){};

    ~DoubleCacheQueue(){};

    std::string get_id()
    {
        return id_;
    };

    void push_to_pre_queue(T t)
    {
        pre_queue_.push(t);
    };

    bool pop_from_pre_queue(T &t)
    {
        if (pre_queue_.empty())
        {
            return false;
        }

        t = pre_queue_.front();
        pre_queue_.pop();
        return true;
    };

    T wait_for_pop_from_post_queue()
    {
        return post_queue_.wait_for_data();
    };

    void push_to_post_queue(T t)
    {
        post_queue_.push(t);
    };
};