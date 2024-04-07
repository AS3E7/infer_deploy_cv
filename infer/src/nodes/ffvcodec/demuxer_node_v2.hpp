//
// Created by cc on 2021/11/3.
//

#ifndef __DEMUXER_NODE_V1_HPP__
#define __DEMUXER_NODE_V1_HPP__

#include "message_templates.hpp"
#include "modules/endecode/demux_stream_v2.h"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"

namespace gddi {
namespace nodes {
class Demuxer_v2 : public node_any_basic<Demuxer_v2> {
protected:
    message_pipe<msgs::av_stream_open> raise_open_;
    message_pipe<msgs::av_packet> raise_packet_;

public:
    explicit Demuxer_v2(std::string name) : node_any_basic(std::move(name)) {
        bind_simple_property("stream_url", stream_url_, "视频流地址");

        raise_open_ = register_output_message_<msgs::av_stream_open>();
        raise_packet_ = register_output_message_<msgs::av_packet>();
    }

    ~Demuxer_v2() override {}

protected:
    void on_setup() override { run_new_demux(); }

    void run_new_demux() {
        opts_.on_stream_exit = [this](bool normal_exit) {
            quit_runner_(normal_exit ? TaskErrorCode::kNormal : TaskErrorCode::kDemuxer);
        };

        demuxer = std::make_unique<av_wrapper::Demuxer_v2>();
        demuxer->open_stream(
            stream_url_,
            [this](const AVStream *av_stream) {
                raise_open_(std::make_shared<msgs::av_stream_open>(
                    av_stream->codecpar, av_q2d(av_stream->r_frame_rate), av_stream->time_base,
                    av_stream->r_frame_rate));
            },
            [this](const std::shared_ptr<AVPacket> &packet) {
                auto msg = std::make_shared<msgs::av_packet>(packet_idx_++, packet.get());
                msg->timestamp.ref.id = packet->pts;
                msg->timestamp.ref.created = msg->timestamp.created;
                raise_packet_(msg);
            },
            opts_);
    }

private:
    av_wrapper::Demuxer_v2::demuxer_options opts_;
    std::unique_ptr<av_wrapper::Demuxer_v2> demuxer;
    int64_t packet_idx_;

    std::string stream_url_;
};
}// namespace nodes
}// namespace gddi

#endif
