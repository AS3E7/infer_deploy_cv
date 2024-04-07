//
// Created by cc on 2021/11/09.
//

#ifndef __ENCODE_VIDEO_HPP__
#define __ENCODE_VIDEO_HPP__

#include "av_lib.h"

namespace av_wrapper
{
    struct EcnodeOptions
    {
        int width = 1920;
        int height = 1080;
        int bitrate = 4000000;
        int framerate = 25;
        int gop_size = 32;
        AVPixelFormat pix_fmt;

#ifdef WITH_BM1684
        std::string codec = "h264_bm";
#else
        std::string codec = "h264_nvenc";
#endif
    };

    class VideoEncoder
    {
        using on_open_t = std::function<void(const AVCodecParameters *codec_parameters, const AVRational &time_base, const AVRational &framerate)>;

    public:
        VideoEncoder();
        ~VideoEncoder();

        bool open_enc(const EcnodeOptions &options, on_open_t on_open);

        bool enc_frame(AVFrame *frame, std::function<void(const AVPacket *packet)> on_packet);

        void close_enc();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}

namespace utils
{
    inline std::ostream &operator<<(std::ostream &oss, const av_wrapper::EcnodeOptions &val)
    {
        oss << " width: " << val.width << ", height: " << val.height << ", bitrate: " << val.bitrate
            << ", framerate: " << val.framerate << ", gop_size: " << val.gop_size;
        return oss;
    }
}

#endif // __ENCODE_VIDEO_HPP__
