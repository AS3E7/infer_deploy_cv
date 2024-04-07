//
// Created by cc on 2021/11/3.
//

#ifndef FFMPEG_WRAPPER_SRC_NODES_DEMUXER_NODE_V1_HPP_
#define FFMPEG_WRAPPER_SRC_NODES_DEMUXER_NODE_V1_HPP_
#include "utils.hpp"
#include "message_templates.hpp"
#include "node_any_basic.hpp"
#include "modules/endecode/demux_stream.hpp"

namespace gddi {
namespace nodes {
namespace lib_av {
namespace msgs {
AVCodecParameters *avcodec_parameters_clone_(const AVCodecParameters *src);

typedef simple_message<std::pair<std::string, std::string>> command;
typedef simple_av_message<AVCodecParameters, avcodec_parameters_clone_, avcodec_parameters_free> video_codecpar;
typedef simple_av_message<AVPacket, av_packet_clone, av_packet_free> video_packet;
}

class Demuxer_v1 : public node_any_basic<Demuxer_v1> {
protected:
    message_pipe<msgs::video_codecpar> raise_open_;
    message_pipe<msgs::video_packet> raise_packet_;
    message_pipe<msgs::command> queued_command_;
    std::string prop_stream_url_;
public:
    explicit Demuxer_v1(std::string name = "Demuxer_v1")
        : node_any_basic<Demuxer_v1>(std::move(name)) {

        bind_simple_property("stream_url", prop_stream_url_, "视频流地址");

        raise_packet_ = register_output_message_<msgs::video_packet>();
        raise_open_ = register_output_message_<msgs::video_codecpar>();

        queued_command_ = register_input_message_handler_({"demux.cmd"}, &Demuxer_v1::_process_command, this);
    }

    ~Demuxer_v1() override {
        _stop_old_demux();
    }

    void open_url(const std::string &stream_url) {
        queued_command_(std::make_shared<msgs::command>(std::make_pair("open", stream_url)));
    }

protected:
    void on_setup() override {
        open_url(prop_stream_url_);
    };

    std::thread demux_thread_;
    bool stop_signal_{};

    void _process_command(const std::shared_ptr<msgs::command> &command) {
        const auto &pair = command->message;
        if (pair.first == "open") {
            run_new_demux(pair.second);
        }
    }

    void _stop_old_demux() {
        if (demux_thread_.joinable()) {
            stop_signal_ = true;
            demux_thread_.join();
            stop_signal_ = false;
        }
    }

    void run_new_demux(const std::string &stream_url) {
        _stop_old_demux();
        // create new demux thread
        demux_thread_ = std::thread([this, stream_url] {
            av_wrapper::DemuxStream(
                stop_signal_,
                stream_url,
                [this](AVStream *p) { return _on_video_open(p); },
                [this](AVPacket *p) { _on_packet(p); });
        });
    }

    bool _on_video_open(AVStream *av_stream) {
        auto codec_parameters = av_stream->codecpar;
        std::cout << "profile    " << codec_parameters->profile << "." << codec_parameters->level << std::endl;
        std::cout << "width      " << codec_parameters->width << std::endl;
        std::cout << "height     " << codec_parameters->height << std::endl;
        std::cout << "fps: " << av_q2d(av_stream->r_frame_rate)
                  << ", den: " << av_stream->r_frame_rate.den
                  << ", num: " << av_stream->r_frame_rate.num;
        std::cout << std::endl;

        raise_open_(std::make_shared<msgs::video_codecpar>(codec_parameters));
        return true;
    }

    void _on_packet(AVPacket *packet) {
        auto msg = std::make_shared<msgs::video_packet>(packet);
        msg->timestamp.ref.id = packet->pts;
        msg->timestamp.ref.created = msg->timestamp.created;
        raise_packet_(msg);
    }
};
}
}
}

#endif //FFMPEG_WRAPPER_SRC_NODES_DEMUXER_NODE_V1_HPP_
