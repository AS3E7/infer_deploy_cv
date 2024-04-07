//
// Created by zhdotcai on 4/25/22.
//

#include "modules/codec/decode_video_v3.h"
#include "modules/codec/demux_stream_v3.h"
#include "modules/wrapper/mlu220_wrapper.hpp"

#ifdef WITH_MLU220
#include "gdd_api.h"
#include "gdd_postproc.h"

void running_by_ffmpeg(int stream_id, const std::string &input_url, const std::string &model_path) {

    auto demuxer = std::make_shared<av_wrapper::Demuxer_v3>();
    auto decoder = std::make_shared<av_wrapper::Decoder_v3>();
    auto alg_impl = std::make_unique<gdd::GddInfer>();

    alg_impl->Init(0, 0, "");
    alg_impl->LoadModel(model_path, "");

    decoder->register_deocde_callback([&alg_impl](const int64_t frame_idx, const std::shared_ptr<AVFrame> &avframe) {
        // infer_server::video::VideoFrame video_frame;
        // video_frame.format = infer_server::video::PixelFmt::NV12;
        // video_frame.width = avframe->width;
        // video_frame.height = avframe->height;
        // video_frame.plane_num =
        //     gddi::image_wrapper::get_plane_num(infer_server::video::PixelFmt::NV12);
        // gddi::image_wrapper::get_stride(infer_server::video::PixelFmt::NV12, avframe->width,
        //                                 video_frame.stride);
        // video_frame.plane[0] =
        //     infer_server::Buffer(FFALIGN(avframe->width, 128) * video_frame.stride[0]);
        // video_frame.plane[1] =
        //     infer_server::Buffer(FFALIGN(avframe->width, 128) * video_frame.stride[1] / 2);
        // video_frame.plane[0].CopyFrom(avframe->data[0], avframe->height * avframe->linesize[0]);
        // video_frame.plane[1].CopyFrom(avframe->data[1], avframe->height * avframe->linesize[1] / 2);

        return true;
    });

    demuxer->register_open_callback(
        [&decoder](const std::shared_ptr<AVCodecParameters> &codecpar) { decoder->open_decoder(codecpar); });

    demuxer->register_video_callback([&decoder](const int64_t pakcet_idx, const std::shared_ptr<AVPacket> &packet) {
        return decoder->decode_packet(packet);
    });

    demuxer->open_stream(input_url,
                         av_wrapper::DemuxerOptions{.tcp_transport = true, .jump_first_video_i_frame = true});

    while (true) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
}

#endif

int main(int argc, char **argv) {

#ifdef WITH_MLU220
    std::thread thread_handle[14];
    for (int i = 0; i < 1; i++) {
        thread_handle[i] = std::thread(running_by_ffmpeg, i, argv[1], argv[2]);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (int i = 0; i < 1; i++) {
        if (thread_handle[i].joinable()) { thread_handle[i].join(); }
    }
#endif

    return 0;
}