//
// Created by cc on 2021/11/5.
//

#ifndef __REMUXER_NODE_V2_HPP__
#define __REMUXER_NODE_V2_HPP__

#include "message_templates.hpp"
#include "modules/endecode/remux_stream.h"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"

namespace gddi {
namespace nodes {
class Remuxer_v2 : public node_any_basic<Remuxer_v2> {
public:
    explicit Remuxer_v2(std::string name)
        : node_any_basic<Remuxer_v2>(std::move(name)),
          stream_remuxer_(std::make_shared<av_wrapper::Remuxer>()) {
        bind_simple_property("stream_url", stream_url_, "推流地址");

        register_input_message_handler_(&Remuxer_v2::_on_open, this);
        register_input_message_handler_(&Remuxer_v2::_on_packet, this);
    }

    ~Remuxer_v2() = default;

private:
    int64_t remuxer_packet_count{};

    void _on_open(const std::shared_ptr<msgs::av_encode_open> &ctx) {
        stream_remuxer_ = std::make_shared<av_wrapper::Remuxer>();
        stream_remuxer_->init(ctx->codec_parameters_, ctx->time_base_, ctx->framerate_);
    }

    void _on_packet(const std::shared_ptr<msgs::av_packet> &msg) {
        remuxer_packet_count++;
        if (!stream_remuxer_->write(msg->packet_)
            && std::chrono::duration<double>(std::chrono::system_clock::now() - last_open_point)
                    .count()
                > 1) {
            stream_remuxer_->close();
            if (stream_remuxer_->open(stream_url_)) {
                spdlog::info("Success opening url: {}", stream_url_);

                spdlog::info("---------------- File Information ---------------");
                spdlog::info("output_stream_url: {}", stream_url_);
                spdlog::info("---------------- Codec Info -----------------");
                spdlog::info("Long Name: bm HEVC encoder wrapper");
                spdlog::info("Short Name: hevc_bm");
                spdlog::info("Codec Name: hevc");
                spdlog::info("-------------------------------------------------");
            } else {
                spdlog::info("Occurred when opening url: {}", stream_url_);
            }
            last_open_point = std::chrono::system_clock::now();
        }
    }

private:
    std::shared_ptr<av_wrapper::Remuxer> stream_remuxer_;
    std::string stream_url_;

    std::chrono::system_clock::time_point last_open_point;
};
}// namespace nodes
}// namespace gddi

#endif//__REMUXER_NODE_V2_HPP__
