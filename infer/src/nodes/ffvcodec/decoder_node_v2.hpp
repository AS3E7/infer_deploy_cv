//
// Created by cc on 2021/11/5.
//

#ifndef __DECODE_NODE_V2_HPP__
#define __DECODE_NODE_V2_HPP__

#include "message_templates.hpp"
#include "modules/endecode/decode_video_v2.h"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"
#include <map>

namespace gddi {
namespace nodes {
class Decoder_v2 : public node_any_basic<Decoder_v2> {
private:
    message_pipe<msgs::av_decode_open> output_open_;
    message_pipe<msgs::av_frame> output_frame_;

public:
    explicit Decoder_v2(std::string name) : node_any_basic(std::move(name)) {
        register_input_message_handler_(&Decoder_v2::on_open_, this);
        register_input_message_handler_(&Decoder_v2::on_packet_, this);

        output_open_ = register_output_message_<msgs::av_decode_open>();
        output_frame_ = register_output_message_<msgs::av_frame>();

#ifdef WITH_BM1684
        bind_simple_property("hw_type", hw_type_, {(AVHWDeviceType)0}, "解码器|0:硬件解码");
#else
        bind_simple_property("hw_type", hw_type_, {(AVHWDeviceType)0, (AVHWDeviceType)2},
                             "解码器|0:软解码;2:硬件解码");
#endif
        bind_simple_property("enable_acc", enable_acc_);
    }

protected:
    void on_open_(const std::shared_ptr<msgs::av_stream_open> &stream) {
        auto tb_ = 1 / (av_q2d(stream->time_base_) * stream->fps_);
        duration_ = (int64_t)tb_;

        video_decoder_ = std::make_shared<av_wrapper::VideoDecoder_v2>();

        video_decoder_->set_decoder_on_open([this](const AVCodecParameters *codec_par) {
            output_open_(std::make_shared<msgs::av_decode_open>(codec_par));
        });

        video_decoder_->set_decoder_callback([this](uint64_t frame_idx, const std::shared_ptr<AVFrame> &frame) {
            raise_decoded_frame(frame_idx, frame);
        });

        codec_par_ = std::shared_ptr<AVCodecParameters>(
            avcodec_parameters_alloc(),
            [](AVCodecParameters *ptr) { avcodec_parameters_free(&ptr); });
        avcodec_parameters_copy(codec_par_.get(), stream->codec_parameters_);

        if (!video_decoder_->open_decoder(codec_par_.get(), (AVHWDeviceType)hw_type_)) {
            quit_runner_(TaskErrorCode::kDecoder);
        }
    }

    void on_packet_(const std::shared_ptr<msgs::av_packet> &msg) {
        ref_packet_push(msg);
        packet_count_++;
        if (video_decoder_->filter_packet(msg->packet_)) {
            spdlog::error("on_packet_: {}", msg->packet_->pts);
            if (video_decoder_->open_decoder(codec_par_.get(), (AVHWDeviceType)hw_type_)) {
                spdlog::info("Success to open decoder");
            }
        }
    }

    void raise_decoded_frame(int64_t frame_idx, const std::shared_ptr<AVFrame> &frame) {
        auto av_frame = std::make_shared<msgs::av_frame>(frame_idx, frame.get());
        ref_frame_fill(av_frame);
        output_frame_(av_frame);
    }

private:
    void ref_packet_push(const std::shared_ptr<msgs::av_packet> &msg) {
        if (msg->packet_->pts > 0) {
            ref_times_[msg->packet_->pts] = msg->timestamp.ref.created;
            if (ref_times_.size() > 50) { ref_times_.erase(ref_times_.begin()); }
        }
    }

    void ref_frame_fill(const std::shared_ptr<msgs::av_frame> &av_frame) {
        auto iter = ref_times_.lower_bound(av_frame->frame_->pts);
        if (iter != ref_times_.end()) {
            av_frame->timestamp.ref.id = iter->first;
            av_frame->timestamp.ref.created = iter->second;
            ref_times_.erase(iter);
            return;
        }
        std::cout << "No Pts cache for : " << av_frame->frame_->pts << std::endl;
    }

private:
    std::shared_ptr<AVCodecParameters> codec_par_;
    std::shared_ptr<av_wrapper::VideoDecoder_v2> video_decoder_;
    std::map<int64_t, std::chrono::high_resolution_clock::time_point> ref_times_;

    int64_t duration_{};
    int64_t frame_count_{};
    int64_t packet_count_{};

    int hw_type_ = 0;
    bool enable_acc_ = false;
};

}// namespace nodes

}// namespace gddi

#endif//__DECODE_NODE_V2_HPP__
