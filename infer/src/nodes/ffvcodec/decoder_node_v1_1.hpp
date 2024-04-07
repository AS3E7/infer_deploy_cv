//
// Created by cc on 2021/11/5.
//

#ifndef FFMPEG_WRAPPER_SRC_NODES_DECODE_V1_1_HPP_
#define FFMPEG_WRAPPER_SRC_NODES_DECODE_V1_1_HPP_

#include <map>
#include "utils.hpp"
#include "node_any_basic.hpp"
#include "message_templates.hpp"
#include "modules/endecode/decode_video_impl.hpp"
#include "demuxer_node_v1_1.hpp"

namespace gddi {

namespace nodes {

class Decoder_v1_1 : public node_any_basic<Decoder_v1_1> {
private:
    message_pipe<msgs::video_frame> output_frame_;
    message_pipe<msgs::av_status> output_status_;

public:
    explicit Decoder_v1_1(std::string name)
        : node_any_basic(std::move(name)) {
        register_input_message_handler_(&Decoder_v1_1::on_packet_, this);
        register_input_message_handler_(&Decoder_v1_1::on_open_, this);

        output_frame_ = register_output_message_<msgs::video_frame>();
        output_status_ = register_output_message_<msgs::av_status>();

        decode_url_options.disable_hw_acc = false;
        bind_simple_property("prefer_hw", decode_url_options.prefer_hw, { (AVHWDeviceType)0, (AVHWDeviceType)2 },
            "解码器|0:软解码;2:硬件解码");
        bind_simple_property("disable_hw_acc", decode_url_options.disable_hw_acc);
    }

    av_wrapper::DecodeUrlOptions decode_url_options{};
    int64_t demux_packet_count{};
protected:
    void on_open_(const std::shared_ptr<lib_av::msgs::av_stream_open> &stream) {
        auto hw_name = av_hwdevice_get_type_name(decode_url_options.prefer_hw);
        std::cout << "Open for: " << stream->codec_parameters_
                  << ", prefer hw: " << (hw_name ? hw_name : "")
                  << std::endl;
        auto tb_ = 1 / (av_q2d(stream->time_base_) * stream->fps_);
        duration_ = (int64_t)tb_;

        video_decoder_ = std::make_shared<av_wrapper::VideoDecoder>(
            [this](const av_wrapper::DecodeUrlInfo &info) {
                if (info.hw_type != AV_HWDEVICE_TYPE_NONE) {
                    std::cout << "INFO: device: " << av_hwdevice_get_type_name(info.hw_type)
                              << ", pixfmt: " << av_get_pix_fmt_name(info.hw_pixfmt) << std::endl;
                }
                output_status_(msgs::av_status::make_shared("decode.ready"));
            },
            [this](AVFrame *frame, int64_t frame_idx) {
                raise_decoded_frame(frame, frame_idx);
            }, decode_url_options);

        if (video_decoder_->open_decoder(stream->codec_parameters_)) {
            demux_packet_count = 0;
            spdlog::debug("====================success open!!");
        } else {
            video_decoder_ = nullptr;
            std::cout << "fail to open the decoder!!!\n";
            quit_runner_(TaskErrorCode::kDecoder);
        }

    }

    void on_packet_(const std::shared_ptr<lib_av::msgs::video_packet> &packet) {
        if (video_decoder_) {
            ref_packet_push(packet);
            demux_packet_count++;
            if (video_decoder_->filter_packet(packet->message, demux_packet_count)) {
                video_decoder_ = nullptr; // Release the video_decoder_
                std::cout << "on_packet_: " << packet->message << std::endl;
                output_status_(std::make_shared<msgs::av_status>("decode.error"));
                quit_runner_(TaskErrorCode::kDecoder);
            }
        } else {
            std::cout << "on_packet_(No Decoder!): " << packet->message << std::endl;
        }
    }

    void raise_decoded_frame(AVFrame *frame, int64_t /*frame_idx*/) {
        auto video_frame = std::make_shared<msgs::video_frame>(frame);
        ref_frame_fill(video_frame);
        output_frame_(video_frame);
    }

private:
    void ref_packet_push(const std::shared_ptr<lib_av::msgs::video_packet> &packet) {
        if (packet->message->pts > 0) {
            ref_times_[packet->message->pts] = packet->timestamp.ref.created;
            if (ref_times_.size() > 50) {
                ref_times_.erase(ref_times_.begin());
            }
        }
        // std::cout << "PUSH<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << packet->message->pts << std::endl;
    }

    void ref_frame_fill(const std::shared_ptr<msgs::video_frame> &video_frame) {
        // std::cout << "PULL<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << video_frame->message->pts << std::endl;
        auto iter = ref_times_.lower_bound(video_frame->message->pts);
        if (iter != ref_times_.end()) {
            video_frame->timestamp.ref.id = iter->first;
            video_frame->timestamp.ref.created = iter->second;
            ref_times_.erase(iter);
            return;
        }
        std::cout << "No Pts cache for : " << video_frame->message->pts << std::endl;
    }

private:
    int64_t duration_{};
    std::shared_ptr<av_wrapper::VideoDecoder> video_decoder_;
    std::map<int64_t, std::chrono::high_resolution_clock::time_point> ref_times_;
};

}

}

#endif //FFMPEG_WRAPPER_SRC_NODES_DECODE_V1_HPP_
