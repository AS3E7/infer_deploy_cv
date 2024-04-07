//
// Created by cc on 2021/10/28.
//

#include "demux_stream.hpp"
#include "utils.hpp"
#include "debug_tools.hpp"

using gddi::utils::TextWriter;
using gddi::utils::object_release_manager;

namespace av_wrapper {

template<class AvPtr>
class av_ptr_wrapper {
public:
    std::function<void()> release_;
    explicit av_ptr_wrapper(std::function<void()> release)
        : release_(std::move(release)), object(nullptr) {}
    ~av_ptr_wrapper() { release_(); }
    AvPtr *object;
};

class LiveStreamDemuxer {

};

bool call_on_open(const std::function<bool(AVStream *)> &on_open, AVStream *stream) {
    if (on_open) {
        return on_open(stream);
    }
    return false;
}

/**
 * @brief
 * @param stream_url
 * @param on_open
 * @param on_packet
 */
ProcessResult DemuxStream(
    bool &stop_signal,
    const std::string &stream_url,
    const std::function<bool(AVStream *)> &on_open,
    const std::function<void(AVPacket *)> &on_packet
) {
    int64_t demux_packet_count = 0;
    int error_step = 0;
    std::string error_string;
    TextWriter text_writer(error_string);
    auto &err_ss = text_writer.text_stream_;
    object_release_manager release_manager;

    // 1. setup stream option
    AVDictionary *opts = nullptr;
    release_manager.bind("dict options", [&opts] {
        DEBUG_release_view(opts);
        av_dict_free(&opts);
    });
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    error_step++;

    // 2. create stream demux context
    AVFormatContext *fmt_ctx = avformat_alloc_context();
    release_manager.bind("fmt ctx", [&fmt_ctx] {
        DEBUG_release_view(fmt_ctx);
        avformat_close_input(&fmt_ctx);
    });

    // 3. try open
    if (avformat_open_input(&fmt_ctx, stream_url.c_str(), nullptr, &opts) != 0) {
        err_ss << "couldn't open input stream: " << stream_url << std::endl;
        return {false, error_step++, error_string};
    }

    // 4. find stream info
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        err_ss << "couldn't find stream information. " << stream_url << std::endl;
        return {false, error_step++, error_string};
    }

    // 5. find the best stream in file
    int real_video_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (real_video_index < 0) {
        err_ss << "didn't find a video stream." << stream_url << std::endl;
        return {false, error_step++, error_string};
    }

    // 6. call on_open
    auto video_stream = fmt_ctx->streams[real_video_index];
    if (!call_on_open(on_open, video_stream)) {
        return {false, error_step++, "open call failed!"};
    }

    // 8. loop read the packet, until reach end
    auto packet = av_packet_alloc();
    release_manager.bind("packet", [&packet] {
        DEBUG_release_view(packet);
        av_packet_free(&packet);
    });
    if (on_packet) {
        while (true) {
            int read_ret_val = av_read_frame(fmt_ctx, packet);
            if (read_ret_val >= 0) {
                demux_packet_count++;

                if (packet->stream_index == real_video_index) {
                    on_packet(packet);
                }
                av_packet_unref(packet);
            } else {
                return {false, error_step++, "no more packet or EOF!"};
            }

            if (stop_signal) {
                return {true, error_step++, "stop by user control"};
            }
        }
    }
    return {false, error_step++, "no packet handler!"};
}

}