//
// Created by cc on 2021/10/28.
//

#ifndef FFMPEG_WRAPPER_SRC_AV_LIB_H_
#define FFMPEG_WRAPPER_SRC_AV_LIB_H_

#ifdef WITH_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libavutil/timestamp.h>
#include <libswscale/swscale.h>
#if LIBAVCODEC_VERSION_MAJOR == 60
#include <libavcodec/bsf.h>
#endif
}
#endif

#include <memory>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <memory>
#include <utility>
#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <chrono>
#include <string>
#include <map>

inline std::ostream &operator<<(std::ostream &oss, const AVCodecParameters *codec_parameters) {
    if (codec_parameters) {
        auto codec_name = avcodec_get_name(codec_parameters->codec_id);
        oss << codec_parameters->width << "x" << codec_parameters->height
            << " @ " << codec_parameters->bit_rate << " bps, "
            << (codec_name ? codec_name : "");
    }
    return oss;
}

inline std::ostream &operator<<(std::ostream &oss, const AVPacket *packet) {
    if (packet) {
        oss << packet->flags
            << std::setw(7) << packet->size
            << ", " << packet->pts;
    }
    return oss;
}

#endif //FFMPEG_WRAPPER_SRC_AV_LIB_H_
