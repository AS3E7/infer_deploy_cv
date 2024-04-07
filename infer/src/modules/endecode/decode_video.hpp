//
// Created by cc on 2021/10/29.
//

#ifndef FFMPEG_WRAPPER_SRC_DECODE_VIDEO_HPP_
#define FFMPEG_WRAPPER_SRC_DECODE_VIDEO_HPP_
#include "demux_stream.hpp"

namespace av_wrapper
{
    struct DecodeUrlInfo
    {
        AVCodecParameters *codec_par;
        AVHWDeviceType hw_type;
        AVPixelFormat hw_pixfmt;
        bool open_success;
    };

    struct DecodeUrlOptions
    {
        bool disable_hw_acc;
        AVHWDeviceType prefer_hw;
    };

    ProcessResult DecodeUrl(
        bool &stop_signal,
        const DecodeUrlOptions &options,
        const std::string &stream_url,
        const std::function<void(const DecodeUrlInfo &)> &on_ready,
        const std::function<void(AVFrame *, int64_t)> &on_frame);

}

#endif //FFMPEG_WRAPPER_SRC_DECODE_VIDEO_HPP_
