#include "common_basic/thread_worker.hpp"
#include "fmt/core.h"
#include "helper/helper.hpp"
#include "indicators/setting.hpp"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/progress_spinner.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>

double time_for_decode_image(const std::vector<char> &data) {
    double dec_time = 0;
    gddi::benchmark::helper::perf_timer m([&](auto &times) { dec_time = times[0]; });
    m.metric();
    cv::imdecode(data, cv::ImreadModes::IMREAD_COLOR);
    m.metric();
    return dec_time;
}

double run_time_for(const std::function<void()> &func) {
    if (func) {
        double run_time_ = 0;
        gddi::benchmark::helper::perf_timer m([&](auto &times) { run_time_ = times[0]; });
        m.metric();
        func();
        m.metric();
        return run_time_;
    }
    return 0;
}

void bench_for(const std::vector<char> &data);

int main(int argc, char *argv[]) {
    auto &&sample_jpg_path = gddi::benchmark::helper::find_file_path("jiwei_1920x1080.jpeg");

    std::vector<char> data;

    if (gddi::benchmark::helper::read_file(sample_jpg_path, data)) {
        spdlog::info("File Size: {}", data.size());

        bench_for(data);
    }

    return 0;
}

class FunctionBenchmark {
public:
    void run_funcor(const std::function<void()> &func, std::size_t thread_n = 1, int n_sec = 10) {
        gddi::benchmark::helper::perf_timer<> t1;
        long n;
        t1.metric();
        {
            gddi::WorkerPool w(fmt::format("bench{:2}", thread_n), thread_n);
            auto t = run_time_for(func);
            t = run_time_for(func);
            t = run_time_for(func);
            n = (long)(n_sec * thread_n / t);
            for (long i = 0; i < n; i++) { w.enqueue(func); }

            using namespace indicators;
            indicators::show_console_cursor(false);
            indicators::BlockProgressBar bar{option::BarWidth{50}, option::PrefixText{"Benchmark Progress: "},
                                             option::MaxProgress{n}};
            while (w.size_approx() != 0) {
                bar.set_progress(n - w.size_approx());
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            w.quit_wait_empty();
            bar.set_progress(n);
            bar.mark_as_completed();
            indicators::show_console_cursor(true);
        }
        t1.metric();
        fmt::print("thread n  : {}\n", thread_n);
        fmt::print("run func  : {}\n", n);
        fmt::print("time total: {:.2f}\n", t1.time_last());
        fmt::print("run fn/sec: {:.2f}\n", n / t1.time_last());
        fmt::print("fn/thread : {:.2f}\n", n / t1.time_last() / thread_n);
    }
};

void bench_for(const std::vector<char> &data) {
    FunctionBenchmark b;
    auto fn = [&] { cv::imdecode(data, cv::ImreadModes::IMREAD_COLOR); };

    for (std::size_t i = 0; i < std::thread::hardware_concurrency(); i++) { b.run_funcor(fn, i + 1); }
}