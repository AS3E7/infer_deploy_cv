/**
 * @file test_cross_border.cpp
 * @author zhdotcai
 * @brief 
 * @version 0.1
 * @date 2022-12-06
 * 
 * @copyright Copyright (c) 2022 GDDI
 * 
 */

#include "modules/bytetrack/target_tracker.h"
#include "modules/codec/decode_video_v3.h"
#include "modules/codec/demux_stream_v3.h"
#include "modules/postprocess/cross_border.h"
#include "utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#if defined(WITH_BM1684)
AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
#elif defined(WITH_MLU220) || defined(WITH_MLU270)
AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_MLU;
#elif defined(WITH_NVIDIA)
#include "alg.h"
#include <nppi_color_conversion.h>
AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_CUDA;
#elif defined(WITH_INTEL)
AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_VAAPI;
#elif defined(WITH_RV1126)
#include "alg.h"
AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
#else
AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_NONE;
#endif

void running(const std::string &input_url, const std::string model_url) {
    // 解码
    auto demuxer = std::make_shared<av_wrapper::Demuxer_v3>();
    auto decoder = std::make_shared<av_wrapper::Decoder_v3>();

    // 跟踪 + 越界
    auto tracker = std::make_shared<gddi::TargetTracker>(
        gddi::TrackOption{.track_thresh = 0.5, .high_thresh = 0.6, .match_thresh = 0.8, .max_frame_lost = 25});
    auto counter = std::make_shared<gddi::CrossBorder>();

#if defined(WITH_NVIDIA) || defined(WITH_RV1126)
    // 初始化模型
    auto alg_impl = std::make_unique<AlgImpl>();
    if (alg_impl->Init(0, "config/alg_config.json", model_url, 1) != GDDI_SUCCESS) {
        throw std::runtime_error("Failed to load model.");
    }
#endif

    // 初始化边界, 规定从左到右穿过射线视为越界
    std::vector<gddi::Point2i> border{{960, 1080}, {960, 0}};
    counter->init_border(border);

    decoder->register_deocde_callback(
#if defined(WITH_NVIDIA) || defined(WITH_RV1126)
        [&alg_impl, &tracker, &counter](const int64_t frame_idx, const std::shared_ptr<AVFrame> &avframe) {
#else
        [&tracker, &counter](const int64_t frame_idx, const std::shared_ptr<AVFrame> &avframe) {
#endif
            // 存放目标信息
            auto vec_objects = std::vector<gddi::TrackObject>();

#if defined(WITH_NVIDIA)
            auto mat_frame = cv::cuda::GpuMat(avframe->height, avframe->width, CV_8UC3);
            nppiNV12ToBGR_8u_P2C3R(avframe->data, avframe->linesize[0], mat_frame.data, mat_frame.step1(),
                                   NppiSize{avframe->width, avframe->height});
            std::vector<std::shared_ptr<TransferBuffer>> trans_buf;
            trans_buf.emplace_back(
                std::make_shared<TransferBuffer>("", mat_frame.rows, mat_frame.cols, mat_frame.data, frame_idx, true));

            alg_impl->PreProcessBatch(trans_buf);
            alg_impl->ProcessBatch(trans_buf);

            std::vector<AlgOutput> vec_res;
            alg_impl->PostProcessBatch(trans_buf, vec_res);

            for (const auto &res : vec_res) {
                for (auto &item : *(std::vector<DetectRes> *)res.data_.get()) {
                    vec_objects.push_back(
                        gddi::TrackObject{.target_id = 0,
                                          .class_id = item.class_id_,
                                          .prob = item.prob_,
                                          .rect = {item.bbox_[0], item.bbox_[1], item.bbox_[2], item.bbox_[3]}});
                }
            }
#elif defined(WITH_RV1126)
            auto buffer = std::vector<u_char>(avframe->width * avframe->height * 3 / 2);
            memcpy(buffer.data(), avframe->data[0], avframe->width * avframe->height);
            memcpy(buffer.data() + avframe->width * avframe->height, avframe->data[1],
                   avframe->height * avframe->width / 4);
            memcpy(buffer.data() + avframe->width * avframe->height * 5 / 4, avframe->data[2],
                   avframe->height * avframe->width / 4);

            auto trans_buf =
                std::make_shared<TransferBuffer>("", avframe->height, avframe->width, buffer.data(), frame_idx, true);
            alg_impl->PreProcess(trans_buf);
            alg_impl->Process(trans_buf);

            AlgOutput alg_res;
            alg_impl->PostProcess(trans_buf, alg_res);

            int index = 0;
            for (auto &item : *(std::vector<DetectRes> *)alg_res.data_.get()) {
                vec_objects.push_back(
                    gddi::TrackObject{.target_id = index,
                                      .class_id = item.class_id_,
                                      .prob = item.prob_,
                                      .rect = {item.bbox_[0], item.bbox_[1], item.bbox_[2], item.bbox_[3]}});
            }
#else
            vec_objects.push_back(
                gddi::TrackObject{.target_id = 0, .class_id = 0, .prob = 0, .rect = {0, 0, 100, 100}});
#endif
            // 目标跟踪
            std::map<int, gddi::Rect2f> rects;
            auto tracked_target = tracker->update_objects(vec_objects);
            for (auto &[track_id, target] : tracked_target) {
                rects[track_id] = gddi::Rect2f{target.rect.x, target.rect.y, target.rect.width, target.rect.height};
                // printf("track_id: %d, x: %d, y: %d, width: %d, height: %d\n", track_id, target.rect.x,
                //        target.rect.y, target.rect.width, target.rect.height);
            }

            // 更新目标位置，若 vec_direction 非空，则有新目标越界
            auto vec_direction = counter->update_position(rects);
            for (const auto [track_id, direction] : vec_direction[0]) {
#if defined(WITH_NVIDIA)
                cv::Mat cpu_mat;
                mat_frame.download(cpu_mat);
                cv::rectangle(
                    cpu_mat, cv::Point2f{rects[track_id].x, rects[track_id].y},
                    cv::Point2f{rects[track_id].x + rects[track_id].width, rects[track_id].y + rects[track_id].height},
                    cv::Scalar{0, 0, 255, 0});
                cv::imwrite(std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count())
                                + ".jpg",
                            cpu_mat);
#endif
                printf("track_id: %d, direction: %d\n", track_id, direction);
            }

            return true;
        });

    demuxer->register_open_callback(
        [decoder](const std::shared_ptr<AVCodecParameters> &codecpar) { decoder->open_decoder(codecpar, hw_type); });

    demuxer->register_video_callback([decoder](const int64_t pakcet_idx, const std::shared_ptr<AVPacket> &packet) {
        return decoder->decode_packet(packet);
    });

    demuxer->open_stream(input_url,
                         av_wrapper::DemuxerOptions{.tcp_transport = true, .jump_first_video_i_frame = true});

    std::this_thread::sleep_for(std::chrono::seconds(3600));
}

void usage() {
    printf("Options:\n");
    printf(" -s, --stream_url=        Video stream url.\n");
    printf(" -m, --model_url=         Model file path\n");
    printf(" -t, --threads=1          The number of threads\n");
}

int main(int argc, char **argv) {
    static struct option long_options[] = {{"stream_url", required_argument, 0, 's'},
                                           {"model_url", required_argument, 0, 'm'},
                                           {"threads", no_argument, 0, 't'},
                                           {"help", no_argument, NULL, 'h'}};

    size_t thread_num{1};
    std::string stream_url;
    std::string model_url;

    while (true) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "smt", long_options, &option_index);
        if (c == -1) break;

        switch (c) {
            case 's': stream_url = optarg; break;
            case 'm': model_url = optarg; break;
            case 't': thread_num = atoi(optarg); break;
            case 'h': usage(); return 1;
            default: printf("?? getopt returned character code 0%o ??\n", c);
        }
    }

    printf("stream_url: %s, model_url: %s\n", stream_url.c_str(), model_url.c_str());

    std::thread thread_handle[thread_num];
    for (int j = 0; j < thread_num; j++) { thread_handle[j] = std::thread(running, stream_url, model_url); }
    for (int j = 0; j < thread_num; j++) {
        if (thread_handle[j].joinable()) { thread_handle[j].join(); }
    }

    return 0;
}