//
// Created by cc on 2021/10/29.
//
#include "../src/modules/endecode/decode_video.hpp"

void safe_join(std::thread &thread) {
    if (thread.joinable()) {
        thread.join();
    }
}

int main(int argc, char *argv[]) {
    bool stop_signal = false;
    std::thread dec([&stop_signal] {
        av_wrapper::DecodeUrl(
            stop_signal,
            {false, AV_HWDEVICE_TYPE_CUDA},
            "rtsp://admin:gddi1234@192.168.1.233:554/h264/ch1/main/av_stream",
            [](const av_wrapper::DecodeUrlInfo &info) {
                std::cout << "INFO: device: " << av_hwdevice_get_type_name(info.hw_type)
                          << ", pixfmt: " << av_get_pix_fmt_name(info.hw_pixfmt) << std::endl;
            },
            [](AVFrame *frame, int64_t frame_index) {
                std::cout << "Pts/Duration:" << std::setw(12) << frame->pts
                          << " / " << std::setw(6) << std::setprecision(2) << std::fixed
                          << (double)frame->pts / (double)(frame->pkt_duration)
                          << "," << frame_index
                          << std::endl;
            });
    });
    char cmd_ch;
    while (true) {
        std::cin >> cmd_ch;
        if (cmd_ch) {
            break;
        }
    }
    stop_signal = true;
    safe_join(dec);
    return 0;
}