//
// Created by zhdotcai on 4/25/22.
//

#include "modules/endecode/decode_video_v2.h"
#include "modules/endecode/demux_stream_v2.h"

void running(const std::string &input_url) {
    auto demuxer = std::make_shared<av_wrapper::Demuxer_v2>();
    auto decoder = std::make_shared<av_wrapper::VideoDecoder_v2>();

    demuxer->open_stream(
        input_url,
        [decoder](const AVStream *av_stream) {
            decoder->set_decoder_callback([](const int64_t frame_idx,
                                             const std::shared_ptr<AVFrame> &avframe) {
                if (frame_idx % 200 == 0) {
                    printf("=================================== avframe count: %ld\n", frame_idx);
                }
            });
            auto codec_par = std::shared_ptr<AVCodecParameters>(
                avcodec_parameters_alloc(),
                [](AVCodecParameters *ptr) { avcodec_parameters_free(&ptr); });
            avcodec_parameters_copy(codec_par.get(), av_stream->codecpar);
            decoder->open_decoder(codec_par.get(), (AVHWDeviceType)0);
        },
        [decoder](const std::shared_ptr<AVPacket> &packet) {
            return decoder->filter_packet(packet.get());
        },
        av_wrapper::Demuxer_v2::demuxer_options{.tcp_transport = true,
                                                .jump_first_video_i_frame = true,
                                                .readrate_speed = 1});
    std::this_thread::sleep_for(std::chrono::seconds(3600));
}

int main(int argc, char **argv) {
    std::thread thread_handle[16];
    for (int j = 0; j < 16; j++) { thread_handle[j] = std::thread(running, argv[1]); }
    for (int j = 0; j < 16; j++) {
        if (thread_handle[j].joinable()) { thread_handle[j].join(); }
    }

    return 0;
}