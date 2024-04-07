//
// Created by cc on 2021/11/1.
//
#include "../../src/modules/endecode/decode_video.hpp"
#include "OpenCV/cv_tools.hpp"
#include "concurrentqueue.h"
#include "debug_tools.hpp"

void safe_join(std::thread &thread) {
    if (thread.joinable()) {
        thread.join();
    }
}

int main(int argc, char *argv[]) {
    moodycamel::ConcurrentQueue<AVFrame *> frame_queue;
    bool stop_signal = false;
    auto on_ready = [](const av_wrapper::DecodeUrlInfo &info) {
        if (info.hw_type != AV_HWDEVICE_TYPE_NONE) {
            std::cout << "INFO: device: " << av_hwdevice_get_type_name(info.hw_type)
                      << ", pixfmt: " << av_get_pix_fmt_name(info.hw_pixfmt) << std::endl;
        }
    };
    auto on_frame = [&frame_queue](AVFrame *frame, int64_t frame_index) {
        frame_queue.enqueue(av_frame_clone(frame));
    };

    std::thread dec([=, &stop_signal] {
        av_wrapper::DecodeUrl(
            stop_signal,
            {false, AV_HWDEVICE_TYPE_CUDA},//AV_HWDEVICE_TYPE_CUDA,
            "rtsp://admin:gddi1234@192.168.1.233:554/h264/ch1/main/av_stream",
            on_ready,
            on_frame);
    });

    while (true) {
        auto cmd_ch = cv::waitKey(1);
        if (cmd_ch == 27) {
            break;
        }
        AVFrame *frame_to_show;
        if (frame_queue.try_dequeue(frame_to_show)) {
            //av_wrapper::show_frame("test", frame_to_show);
            av_frame_free(&frame_to_show);
        }
    }

    stop_signal = true;
    safe_join(dec);
    return 0;
}