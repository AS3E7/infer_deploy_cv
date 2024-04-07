//
// Created by cc on 2021/10/28.
//

#include "../src/modules/endecode/demux_stream.hpp"

bool test_stream(AVStream *av_stream) {
    auto codec_parameters = av_stream->codecpar;
//    auto nb_frames = stream->nb_frames;
//    auto frame_rate = av_q2d(stream->avg_frame_rate);
//    auto time_base = stream->time_base;
//    auto start_time = stream->start_time;
    std::cout << "profile    " << codec_parameters->profile << "." << codec_parameters->level << std::endl;
    std::cout << "width      " << codec_parameters->width << std::endl;
    std::cout << "height     " << codec_parameters->height << std::endl;
//    std::cout << "nb_frames  " << nb_frames << std::endl;
//    std::cout << "frame_rate " << frame_rate << std::endl;
//    std::cout << "start_time " << start_time << std::endl;
//    std::cout << "time_base  " << time_base.num << "/" << time_base.den << std::endl;
    return true;
}

void test_packet(AVPacket *packet) {
    static auto pre_pts = packet->pts;
    static auto pre_dts = packet->dts;

    auto pts_diff = packet->pts - pre_pts;
    if (pts_diff != packet->duration) {
        std::cout << "    "
                  << std::setw(10) << packet->pts
                  << std::setw(10) << packet->dts - pre_dts
                  << std::setw(10) << packet->pts - pre_pts
                  << ", " << packet->flags
                  << std::setw(2) << (packet->dts == packet->pts)
                  << std::endl;
    }

    pre_pts = packet->pts;
    pre_dts = packet->dts;
}

int main(int argc, char *argv[]) {
    bool stop_signal = false;
    av_wrapper::DemuxStream(stop_signal,
                           "rtsp://admin:gddi1234@192.168.1.233:554/h264/ch1/main/av_stream",
                           test_stream,
                           test_packet);
    return 0;
}