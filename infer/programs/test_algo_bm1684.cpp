//
// Created by zhdotcai on 4/25/22.
//

#include "codec/decode_video_v3.h"
#include "codec/demux_stream_v3.h"
#include "modules/wrapper/bm1684_wrapper.hpp"
#include "spdlog/spdlog.h"

#ifdef WITH_BM1684
#include <api/infer_api.h>
#include <core/result_def.h>

void running_by_ffmpeg(int stream_id, const std::string &input_url, const std::string &model_path) {
    auto bm_handle = std::shared_ptr<bm_handle_t>(new bm_handle_t, [](bm_handle_t *ptr) { bm_dev_free(*ptr); });
    if (bm_dev_request(bm_handle.get(), 0) != 0) {
        printf("** failed to request device\n");
        return;
    }

    auto demuxer = std::make_shared<av_wrapper::Demuxer_v3>();
    auto decoder = std::make_shared<av_wrapper::Decoder_v3>();
    auto alg_impl = std::make_unique<gddeploy::InferAPI>();
    alg_impl->Init("", model_path, "", gddeploy::ENUM_API_TYPE::ENUM_API_SESSION_API);

    decoder->register_deocde_callback(
        [stream_id, &alg_impl, &bm_handle](const int64_t frame_idx, const std::shared_ptr<AVFrame> &frame) {
            auto src_frame = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
                bm_image_destroy(*ptr);
                delete ptr;
            });

            int stride[2]{frame->width, frame->width};
            bm_image_create(*bm_handle, frame->height, frame->width, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE,
                            src_frame.get(), stride);
            bm_image_alloc_dev_mem_heap_mask(*src_frame, 6);
            gddi::image_wrapper::bm_image_from_frame(*frame, *src_frame);

            auto surf = gddi::image_wrapper::convert_bm_image_to_sufsurface(*src_frame);

            gddeploy::PackagePtr in = gddeploy::Package::Create(1);

            gddeploy::PackagePtr out = gddeploy::Package::Create(1);
            alg_impl->InferSync(in, out);

            if (out->data[0]->HasMetaValue()) {
                auto result = out->data[0]->GetMetaData<gddeploy::InferResult>();
                for (auto result_type : result.result_type) {
                    if (result_type == gddeploy::GDD_RESULT_TYPE_DETECT) {
                        for (const auto &item : result.detect_result.detect_imgs) {
                            for (const auto &obj : item.detect_objs) {
                                spdlog::info("detect result: class_id: {}, score: {}, box: [{}, {}, {}, {}]",
                                             obj.class_id, obj.score, obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h);
                            }
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

    demuxer->open_stream(input_url,
                         av_wrapper::DemuxerOptions{.tcp_transport = true, .jump_first_video_i_frame = true});

    while (true) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
}

void running_by_opencv(int stream_id, const std::string &input_url, const std::string &model_path) {
    auto bm_handle = std::shared_ptr<bm_handle_t>(new bm_handle_t, [](bm_handle_t *ptr) { bm_dev_free(*ptr); });
    if (bm_dev_request(bm_handle.get(), 0) != 0) {
        printf("** failed to request device\n");
        return;
    }

    auto alg_impl = std::make_unique<gddeploy::InferAPI>();
    alg_impl->Init("", model_path, "", gddeploy::ENUM_API_TYPE::ENUM_API_SESSION_API);

    cv::VideoCapture capture;
    capture.open(input_url);

    while (true) {
        cv::Mat frame;
        capture.read(frame);
        auto src_frame = std::shared_ptr<bm_image>(new bm_image, [frame](bm_image *ptr) {
            bm_image_destroy(*ptr);
            delete ptr;
        });
        bm_image_from_mat(*bm_handle, frame, *src_frame);

        auto surf = gddi::image_wrapper::convert_bm_image_to_sufsurface(*src_frame);

        gddeploy::PackagePtr in = gddeploy::Package::Create(1);

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        alg_impl->InferSync(in, out);

        if (out->data[0]->HasMetaValue()) {
            auto result = out->data[0]->GetMetaData<gddeploy::InferResult>();
            for (auto result_type : result.result_type) {
                if (result_type == gddeploy::GDD_RESULT_TYPE_DETECT) {
                    for (const auto &item : result.detect_result.detect_imgs) {
                        for (const auto &obj : item.detect_objs) {
                            spdlog::info("detect result: class_id: {}, score: {}, box: [{}, {}, {}, {}]", obj.class_id,
                                         obj.score, obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h);
                        }
                    }
                }
            }
        }
    }
}

#endif

int main(int argc, char **argv) {

#ifdef WITH_BM1684
    std::thread thread_handle[38];
    for (int i = 0; i < 14; i++) {
        thread_handle[i] = std::thread(running_by_ffmpeg, i, argv[1], argv[2]);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    for (int i = 0; i < 14; i++) {
        if (thread_handle[i].joinable()) { thread_handle[i].join(); }
    }
#endif

    return 0;
}