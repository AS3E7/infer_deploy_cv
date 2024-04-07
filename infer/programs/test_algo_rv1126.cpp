//
// Created by zhdotcai on 4/25/22.
//

#include "codec/decode_video_v3.h"
#include "codec/demux_stream_v3.h"
#include "modules/wrapper/rv1126_wrapper.hpp"
#include <cstdio>

#ifdef WITH_RV1126
#include "alg.h"
#include "res.h"

void running_by_ffmpeg(int stream_id, const std::string &input_url, const std::string &model_path) {
    auto demuxer = std::make_shared<av_wrapper::Demuxer_v3>();
    auto decoder = std::make_shared<av_wrapper::Decoder_v3>();
    auto alg_impl = std::make_shared<AlgImpl>();

    if (alg_impl->Init(0, "config/alg_config.json", model_path, 1) != GDDI_SUCCESS) {
        throw std::runtime_error("Failed to load model.");
    }

    printf("Load model tyep: %d\n", alg_impl->GetModelType());

    decoder->register_deocde_callback([stream_id, &alg_impl](const int64_t frame_idx,
                                                             const std::shared_ptr<AVFrame> &frame) {
        if (fmod(frame_idx, 25 * 1.0 / 15) < 1) {
            auto image = cv::Mat(frame->height * 3 / 2, frame->width, CV_8UC1);
            memcpy(image.data, frame->data[0], frame->width * frame->height);
            memcpy(image.data + frame->width * frame->height, frame->data[1], frame->height * frame->width / 4);
            memcpy(image.data + frame->width * frame->height * 5 / 4, frame->data[2], frame->height * frame->width / 4);

            auto trans_buf = std::make_shared<TransferBuffer>("test", frame->height * 2 / 3, frame->width, image.data,
                                                              frame_idx, true);
            alg_impl->PreProcess(trans_buf);
            alg_impl->Process(trans_buf);

            AlgOutput alg_res;
            alg_impl->PostProcess(trans_buf, alg_res);

            int index = 0;
            if (alg_res.data_) {
                if (alg_impl->GetModelType() == Task::DETECT_TASK) {
                    auto vec_res = *(std::vector<DetectRes> *)alg_res.data_.get();
                    for (auto &item : *(std::vector<DetectRes> *)alg_res.data_.get()) {
                        printf("class_id: %d, prob: %.2f, x: %.2f, y: %.2f, w: %.2f, h: %.2f\n", item.class_id_,
                               item.prob_, item.bbox_[0], item.bbox_[1], item.bbox_[2], item.bbox_[3]);
                    }
                }
            }
        }

        return true;
    });

    demuxer->register_open_callback([&decoder](const std::shared_ptr<AVCodecParameters> &codecpar) {
        if (!decoder->open_decoder(codecpar)) { exit(-1); }
    });

    demuxer->register_video_callback([&decoder](const int64_t pakcet_idx, const std::shared_ptr<AVPacket> &packet) {
        return decoder->decode_packet(packet);
    });

    demuxer->open_stream(
        input_url,
        av_wrapper::DemuxerOptions{.tcp_transport = true, .jump_first_video_i_frame = true});

    while (true) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
}

#endif

int main(int argc, char **argv) {

#ifdef WITH_RV1126
    std::thread thread_handle[4];
    for (int i = 0; i < 4; i++) {
        thread_handle[i] = std::thread(running_by_ffmpeg, i, argv[1], argv[2]);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (int i = 0; i < 4; i++) {
        if (thread_handle[i].joinable()) { thread_handle[i].join(); }
    }
#endif

    return 0;
}