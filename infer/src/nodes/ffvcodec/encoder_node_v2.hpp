//
// Created by cc on 2021/11/5.
//

#ifndef __ENCODER_NODE_V2_HPP__
#define __ENCODER_NODE_V2_HPP__

#include "message_templates.hpp"
#include "modules/endecode/encode_video.h"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"

namespace gddi {
namespace nodes {
class Encoder_v2 : public node_any_basic<Encoder_v2> {
protected:
    message_pipe<msgs::av_encode_open> output_open_;
    message_pipe<msgs::av_packet> output_packet_;
    av_wrapper::EcnodeOptions encoder_prop_;

public:
    explicit Encoder_v2(std::string name) : node_any_basic<Encoder_v2>(std::move(name)) {
        bind_simple_property("width", encoder_prop_.width, 1, 1920);
        bind_simple_property("height", encoder_prop_.height, 1, 1080);
        bind_simple_property("framerate", encoder_prop_.framerate, 1, 30);
        bind_simple_property("bitrate", encoder_prop_.bitrate, 0, 8000000);
        bind_simple_property("gop_size", encoder_prop_.gop_size, 1, 32);

#ifdef WITH_BM1684
        bind_simple_property("codec", encoder_prop_.codec, {"h264_bm", "h265_bm"},
                             "编码器|h264_bm:H264;h264_bm:H265");
#else
        bind_simple_property("codec", encoder_prop_.codec, {"h264_nvenc", "hevc_nvenc"},
                             "编码器|h264_nvenc:H264;hevc_nvenc:H265");
#endif

        register_input_message_handler_(&Encoder_v2::_on_frame, this);

        output_open_ = register_output_message_<msgs::av_encode_open>();
        output_packet_ = register_output_message_<msgs::av_packet>();
    }

    ~Encoder_v2() = default;

private:
    void on_setup() override { video_encoder_ = std::make_shared<av_wrapper::VideoEncoder>(); }

    void _on_frame(const std::shared_ptr<msgs::av_frame> &frame) {
        if (video_encoder_->enc_frame(frame->frame_, [this](const AVPacket *packet) {
                output_packet_(std::make_shared<msgs::av_packet>(packet_idx_++, packet));
            })) {
            // tools::timming time(this);
            return;
        }

        encoder_prop_.pix_fmt = (AVPixelFormat)frame->frame_->format;
        auto success = video_encoder_->open_enc(
            encoder_prop_,
            [this](const AVCodecParameters *codec_parameters, const AVRational &time_base,
                   const AVRational &framerate) {
                spdlog::info("Success to open encode");
                output_open_(
                    std::make_shared<msgs::av_encode_open>(codec_parameters, time_base, framerate));
            });
        if (!success) { video_encoder_->close_enc(); }
    }

private:
    int64_t packet_idx_{};
    std::shared_ptr<av_wrapper::VideoEncoder> video_encoder_;
};
}// namespace nodes
}// namespace gddi

#endif//__ENCODER_NODE_V2_HPP__
