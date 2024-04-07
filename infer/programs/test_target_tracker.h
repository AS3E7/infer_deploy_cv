//
// Created by zhdotcai on 4/25/22.
//

#include "codec/decode_video_v3.h"
#include "codec/demux_stream_v3.h"

#ifdef WITH_BM1684
#include "alg.h"
#include "modules/wrapper/bm1684_wrapper.hpp"
#include "res.h"
#include <gtest/gtest.h>

class TargetTrackerTest : public testing::Test {
public:
    TargetTrackerTest(int stream_id, std::string input_url, std::string model_path)
        : stream_id_(stream_id), input_url_(std::move(input_url)), model_path_(std::move(model_path)) {}

protected:
    virtual void SetUp() override {
        printf("Enter into SetUp\n");

        bm_handle_ std::shared_ptr<bm_handle_t>(new bm_handle_t, [](bm_handle_t *ptr) { bm_dev_free(*ptr); });
        if (bm_dev_request(bm_handle_.get(), 0) != 0) {
            printf("** failed to request device\n");
            return;
        }

        auto demuxer = std::make_shared<av_wrapper::Demuxer_v3>();
        auto decoder = std::make_shared<av_wrapper::Decoder_v3>();
        auto alg_impl = std::make_shared<BmnnAlg>(0, "NULL");

        if (alg_impl->loadModel(model_path_, "inference-engine") != 0) { return; }

        decoder->register_deocde_callback([stream_id_, &alg_impl, this](const int64_t frame_idx,
                                                                        const std::shared_ptr<AVFrame> &frame) {
            auto src_frame = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
                bm_image_destroy(*ptr);
                delete ptr;
            });

            int stride[2]{frame->width, frame->width};
            bm_image_create(*bm_handle_, frame->height, frame->width, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE,
                            src_frame.get(), stride);
            bm_image_alloc_dev_mem_heap_mask(*src_frame, 6);
            gddi::image_wrapper::bm_image_from_frame(*frame, *src_frame);

            auto transBuf = std::make_shared<TRANSFER_BUFFER>("", src_frame->height, src_frame->width, src_frame, 1);
            transBuf->m_streamId = stream_id_;

            ALG_OUTPUT res;
            // alg_impl->preProcess(transBuf.get());
            // alg_impl->GetResult(transBuf.get(), &res);

            return true;
        });

        demuxer->register_open_callback([&decoder](const std::shared_ptr<AVCodecParameters> &codecpar) {
            if (!decoder->open_decoder(codecpar)) { exit(-1); }
        });

        demuxer->register_video_callback([&decoder](const int64_t pakcet_idx, const std::shared_ptr<AVPacket> &packet) {
            return decoder->decode_packet(packet);
        });

        demuxer->open_stream(
            input_url_,
            av_wrapper::DemuxerOptions{.tcp_transport = true, .jump_first_video_i_frame = true, .readrate_speed = 0});
    }

    virtual void TearDown() override { printf("Exit from TearDown\n"); }

    int stream_id_;
    std::string input_url_;
    std::string model_path_;

    std::shared_ptr<bm_handle_t> bm_handle_;
};

TEST(TargetTrackerTest, testCase1) {
    TargetTrackerTest fooInstance(1, "", "");
    delete fooInstance;
}

#endif

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}