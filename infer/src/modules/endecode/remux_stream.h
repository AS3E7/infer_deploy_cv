#ifndef __REMUXE_STREAM_H__
#define __REMUXE_STREAM_H__

#include "av_lib.h"
#include <map>

namespace av_wrapper
{
    class Remuxer
    {
    public:
        Remuxer();
        ~Remuxer();

        void init(const AVCodecParameters *codecpar, const AVRational &timebase, const AVRational &framerate);
        bool open(const std::string &stream_url);
        bool write(AVPacket *packet);
        void close();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}

#endif