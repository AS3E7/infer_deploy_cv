/**
 * @file test_algo_nvidia.cpp
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2022-12-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>

#if defined(WITH_NVIDIA)

#include "alg.h"
#include "modules/codec/decode_video_v3.h"
#include "modules/codec/demux_stream_v3.h"
#include <nppi_color_conversion.h>
#include <opencv2/core/cuda.hpp>

AVHWDeviceType hw_type = AV_HWDEVICE_TYPE_CUDA;

void running(const std::string &input_url, const std::string model_url) {
    auto demuxer = std::make_shared<av_wrapper::Demuxer_v3>();
    auto decoder = std::make_shared<av_wrapper::Decoder_v3>();

    auto alg_impl = std::make_unique<AlgImpl>();
    if (alg_impl->Init(0, "config/alg_config.json", model_url, 1) != GDDI_SUCCESS) {
        throw std::runtime_error("Failed to load model.");
    }

    decoder->register_deocde_callback([&alg_impl](const int64_t frame_idx, const std::shared_ptr<AVFrame> &avframe) {
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

        int index = 0;
        for (const auto &res : vec_res) {
            for (auto &item : *(std::vector<DetectRes> *)res.data_.get()) {
                printf("class_id: %d, prob: %.2f, x: %.0f, y: %.0f, width: %.0f, height: %.0f\n", item.class_id_,
                       item.prob_, item.bbox_[0], item.bbox_[1], item.bbox_[2], item.bbox_[3]);
            }
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

#endif

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

#if defined(WITH_NVIDIA)
    std::thread thread_handle[thread_num];
    for (int j = 0; j < thread_num; j++) { thread_handle[j] = std::thread(running, stream_url, model_url); }
    for (int j = 0; j < thread_num; j++) {
        if (thread_handle[j].joinable()) { thread_handle[j].join(); }
    }
#endif

    return 0;
}