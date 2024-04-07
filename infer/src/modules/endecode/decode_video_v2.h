#ifndef __DECODE_VIDEO_H__
#define __DECODE_VIDEO_H__

#include "av_lib.h"

namespace av_wrapper {

using on_open_t = std::function<void(const AVCodecParameters *)>;
using on_frame_t = std::function<void(const uint64_t, const std::shared_ptr<AVFrame> &)>;

class VideoDecoder_v2 {
public:
    VideoDecoder_v2();
    ~VideoDecoder_v2();

    /**
         * @brief 打开解码器，通过指定的codec参数
         * 
         * @param codecpar 
         * @param type 硬件加速器类型
         * @return true 
         * @return false 
         */
    bool open_decoder(const AVCodecParameters *codecpar,
                      const AVHWDeviceType type = AV_HWDEVICE_TYPE_NONE);

    void close_decode();

    void set_decoder_on_open(on_open_t on_open);

    /**
         * @brief Set the decoder callback object
         * 
         * @param on_frame 
         */
    void set_decoder_callback(on_frame_t on_frame);

    int filter_packet(const AVPacket *packetx);

    int decode_one_packet(const AVPacket *packet);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}// namespace av_wrapper

#endif