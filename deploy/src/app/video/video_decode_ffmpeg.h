#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
// #include <libavfilter/avfilter.h>
// #include <libavfilter/buffersink.h>
// #include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/time.h>
#include <libavutil/timestamp.h>
// #include <libswscale/swscale.h>
}

#include <memory>
#include <functional>

namespace gddeploy{

using DecodeCallback = std::function<void(const int64_t, const std::shared_ptr<AVFrame> &)>;

class DecoderPrivate;
class Decoder {
public:

public:
    Decoder();
    ~Decoder(){}

    int open_stream(const std::string &stream_url, DecodeCallback decode_cb = NULL);
    /**
     * @brief 注册解码回调
     * 
     * @param decode_cb 
     */
    void register_deocde_callback(const DecodeCallback &decode_cb){
        decode_cb_ = decode_cb;
    }

private:
    std::shared_ptr<DecoderPrivate> decoder_priv_;

    DecodeCallback decode_cb_;
};

}   // namespace gddeploy