//
// Created by cc on 2021/11/3.
//

#ifndef FFMPEG_WRAPPER_SRC_NODES_DEMUXER_NODE_V1_1_HPP_
#define FFMPEG_WRAPPER_SRC_NODES_DEMUXER_NODE_V1_1_HPP_
#include "utils.hpp"
#include "message_templates.hpp"
#include "node_any_basic.hpp"
#include "modules/endecode/demux_stream.hpp"
#include "demuxer_node_v1.hpp"

namespace gddi {
namespace nodes {
namespace lib_av {

namespace msgs {
class av_stream_open : public ngraph::Message {
public:
    explicit av_stream_open(const AVCodecParameters *codec_parameters,
                            double fps,
                            AVRational time_base,
                            AVRational r_frame_rate)
        : codec_parameters_(avcodec_parameters_clone_(codec_parameters)),
          fps_(fps),
          time_base_(time_base),
          r_frame_rate_(r_frame_rate) {}
    ~av_stream_open() override { avcodec_parameters_free(&codec_parameters_); }
    AVCodecParameters *codec_parameters_;
    double fps_;
    AVRational time_base_;
    AVRational r_frame_rate_;

    std::string name() const override { return utils::get_class_name(this); }
    std::string to_string() const override { return utils::fmts(codec_parameters_); }
};
}

class Demuxer_v1_1 : public node_any_basic<Demuxer_v1_1> {
protected:
    message_pipe<msgs::av_stream_open> raise_open_;
    message_pipe<msgs::video_packet> raise_packet_;
    std::string stream_url_;
public:
    explicit Demuxer_v1_1(std::string name)
        : node_any_basic(std::move(name)) {
        bind_simple_property("stream_url", stream_url_, "视频流地址");
        raise_packet_ = register_output_message_<msgs::video_packet>();
        raise_open_ = register_output_message_<msgs::av_stream_open>();
    }

    ~Demuxer_v1_1()
    override {
        _stop_old_demux();
    }

protected:
    std::thread demux_thread_;
    bool stop_signal_{};

    void on_setup() override {
        run_new_demux();
    }

    void _stop_old_demux() {
        if (demux_thread_.joinable()) {
            stop_signal_ = true;
            demux_thread_.join();
            stop_signal_ = false;
        }
    }

    void run_new_demux() {
        _stop_old_demux();
        // create new demux thread
        demux_thread_ = std::thread([this] {
            av_wrapper::DemuxStream(
                stop_signal_,
                stream_url_,
                [this](AVStream *p) { return _on_video_open(p); },
                [this](AVPacket *p) { _on_packet(p); });
        });
    }

    bool _on_video_open(AVStream *av_stream) {
        auto codec_parameters = av_stream->codecpar;
        auto fps = av_q2d(av_stream->r_frame_rate);
        std::cout << "profile    " << codec_parameters->profile << "." << codec_parameters->level << std::endl;
        std::cout << "width      " << codec_parameters->width << std::endl;
        std::cout << "height     " << codec_parameters->height << std::endl;

        raise_open_(std::make_shared<msgs::av_stream_open>(codec_parameters,
                                                           fps,
                                                           av_stream->time_base,
                                                           av_stream->r_frame_rate));
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

#endif //FFMPEG_WRAPPER_SRC_NODES_DEMUXER_NODE_V1_1_HPP_
