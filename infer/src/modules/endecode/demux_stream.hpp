//
// Created by cc on 2021/10/28.
//

#ifndef FFMPEG_WRAPPER_SRC_DEMUX_STREAM_HPP_
#define FFMPEG_WRAPPER_SRC_DEMUX_STREAM_HPP_

#include "av_lib.h"

namespace av_wrapper {

struct ProcessResult {
    bool success;
    int error_code;
    std::string error_message;
};

ProcessResult DemuxStream(
    bool &stop_signal,
    const std::string &stream_url,
    const std::function<bool(AVStream *)> &on_open,
    const std::function<void(AVPacket *)> &on_packet
);

}

#endif //FFMPEG_WRAPPER_SRC_DEMUX_STREAM_HPP_
