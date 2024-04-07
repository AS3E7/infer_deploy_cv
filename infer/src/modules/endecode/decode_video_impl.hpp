//
// Created by cc on 2021/11/4.
//

#ifndef FFMPEG_WRAPPER_SRC_DECODE_VIDEO_IMPL_HPP_
#define FFMPEG_WRAPPER_SRC_DECODE_VIDEO_IMPL_HPP_
#include <utility>

#include "av_lib.h"
#include "decode_video.hpp"

#define AV_RUNTIME_ERROR(err, what)     \
    if ((err) < 0)                      \
    {                                   \
        throw std::runtime_error(what); \
    }

namespace av_wrapper {

std::pair<AVHWDeviceType, AVPixelFormat> find_decoder_hw_config2(AVCodec *decoder, AVHWDeviceType prefer_hw_type);

inline
bool casecmp(const std::string &a, const std::string &b) {
    if (a.length() == b.length()) {
        return std::equal(a.begin(), a.end(), b.begin(), [](char a, char b) {
            return tolower(a) == tolower(b);
        });
    }
    return false;
}

class BitStreamFilter {
public:
    explicit BitStreamFilter(std::function<int(AVPacket *, int64_t)> on_packet) : on_packet_(std::move(on_packet)) {
        av_bit_stream_filter_ = nullptr;
        avbsf_context_ = nullptr;
    }

    ~BitStreamFilter() {
        if (avbsf_context_) {
            av_bsf_free(&avbsf_context_);
        }
        av_bit_stream_filter_ = nullptr;
    }

    bool setup_filter_by_decoder_name(const std::string &name, const AVCodecParameters *codecpar) {
        if (casecmp(name, "h264") || casecmp(name, "h264_bm")) {
            av_bit_stream_filter_ = av_bsf_get_by_name("h264_mp4toannexb");
        } else if (casecmp(name, "hevc")) {
            av_bit_stream_filter_ = av_bsf_get_by_name("hevc_mp4toannexb");
        }

        if (av_bit_stream_filter_) {
            init_bsf_alloc(codecpar);
            return true;
        }
        return false;
    }

    int send_packet(AVPacket *packet, int64_t packet_index) {
        int ret = 0;

        if (av_bit_stream_filter_) {
            // 1. push packet to filter
            auto packet_to_filter = av_packet_clone(packet);
            ret = av_bsf_send_packet(avbsf_context_, packet_to_filter);
            if (ret < 0) {
                av_packet_free(&packet_to_filter);
                fprintf(stderr, "error during decoding filter\n");
                return ret;
            }

            // 2. read packet from filter
            auto packet_result = av_packet_alloc();
            while (av_bsf_receive_packet(avbsf_context_, packet_result) == 0) {
                // packet ready
                ret = on_packet_(packet_result, packet_index);
                // unref the
                av_packet_unref(packet_result);
            }

            // 3. free data
            av_packet_free(&packet_result);
            av_packet_free(&packet_to_filter);
        } else {
            ret = on_packet_(packet, packet_index);
        }
        return ret;
    }

protected:
    void init_bsf_alloc(const AVCodecParameters *codecpar) {
        int ret = 0;
        ret = av_bsf_alloc(av_bit_stream_filter_, &avbsf_context_);
        if (ret < 0) {
            throw std::runtime_error("fail to alloc av_bsf_alloc");
        }
        ret = avcodec_parameters_copy(avbsf_context_->par_in, codecpar);
        if (ret < 0) {
            throw std::runtime_error("fail to alloc av bsf avcodec_parameters_copy");
        }
        av_bsf_init(avbsf_context_);
    }

protected:
    const AVBitStreamFilter *av_bit_stream_filter_;
    AVBSFContext *avbsf_context_;
    std::function<int(AVPacket *, int64_t)> on_packet_;
};

class VideoDecoder {
public:
    std::shared_ptr<BitStreamFilter> bit_stream_filter;
    const AVCodec *decoder;
    AVPixelFormat hw_pixfmt;
    AVCodecContext *decoder_ctx;
    AVBufferRef *hw_device_ctx;
    AVHWDeviceType hw_type;
    std::function<void(AVFrame *, int64_t)> on_frame_;
    std::function<void(const DecodeUrlInfo &)> on_ready_;
    bool hardware_decode_device_available;
    DecodeUrlOptions options_;

    explicit VideoDecoder(
        std::function<void(const DecodeUrlInfo &)> on_ready,
        std::function<void(AVFrame *, int64_t)> on_frame,
        const DecodeUrlOptions &options)
        : decoder(nullptr),
          hw_pixfmt(AV_PIX_FMT_NONE),
          decoder_ctx(nullptr),
          hw_device_ctx(nullptr),
          hw_type(AV_HWDEVICE_TYPE_NONE),
          on_frame_(std::move(on_frame)),
          on_ready_(std::move(on_ready)),
          hardware_decode_device_available(false),
          options_(options) {
        // setup bit stream filter callback
        bit_stream_filter = std::make_shared<BitStreamFilter>([this](AVPacket *packet, int64_t packet_index) {
            return decode_packet(packet, packet_index);
        });
    }
    ~VideoDecoder() {
        DEBUG_release_view(this);
        if (decoder_ctx) {
            DEBUG_release_view(decoder_ctx);
            avcodec_free_context(&decoder_ctx);
        }
        if (hw_device_ctx) {
            DEBUG_release_view(hw_device_ctx);
            av_buffer_unref(&hw_device_ctx);
        }
        decoder = nullptr;
        hw_type = AV_HWDEVICE_TYPE_NONE;
        hw_pixfmt = AV_PIX_FMT_NONE;
    }

    static AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
        auto _this = reinterpret_cast<VideoDecoder *>(ctx->opaque);
        if (_this) {
            const enum AVPixelFormat *p;
            auto hw_pixfmt = _this->hw_pixfmt;
            for (p = pix_fmts; *p != -1; p++) {
                if (*p == hw_pixfmt)
                    return *p;
            }

            fprintf(stderr, "Failed to get HW surface format.\n");
        }
        return AV_PIX_FMT_NONE;
    }

    bool init_pixfmt(AVCodecParameters *stream, AVHWDeviceType prefer) {
        decoder = avcodec_find_decoder(stream->codec_id);
        if (decoder) {
            // here try to decode by hardware!!!
            if (options_.disable_hw_acc) {
                hardware_decode_device_available = false;
                std::cout << "force use software decoder: " << avcodec_get_name(decoder->id) << std::endl;
            } else {
                auto device_pixfmt = find_decoder_hw_config2(const_cast<AVCodec *>(decoder), prefer);
                hw_type = device_pixfmt.first;
                hw_pixfmt = device_pixfmt.second;
                hardware_decode_device_available = hw_pixfmt != AV_PIX_FMT_NONE;
            }

            if (!hardware_decode_device_available) {
                if (!options_.disable_hw_acc) {
                    std::cout << "fail to setup hardware decode for: " << decoder->name << std::endl;
                }
            }
            return bit_stream_filter->setup_filter_by_decoder_name(decoder->name, stream);
        }
        std::cout << "no decoder for given " << stream->codec_id << std::endl;
        return false;
    }

    bool init_context(AVCodecParameters *stream) {
        decoder_ctx = avcodec_alloc_context3(decoder);

        if (decoder_ctx != nullptr) {
            AV_RUNTIME_ERROR(avcodec_parameters_to_context(decoder_ctx, stream),
                             "fail in init_decoder_context");
            if (hardware_decode_device_available) {
                decoder_ctx->opaque = this;
                decoder_ctx->get_format = get_hw_format;
            }
            return true;
        }
        std::cerr << "alloc decoder ctx fail!" << std::endl;
        return false;
    }

    bool init_hardware() {
        int err = 0;
        if ((err = av_hwdevice_ctx_create(&hw_device_ctx, hw_type, nullptr, nullptr, 0)) < 0) {
            fprintf(stderr, "failed to create specified HW device.\n");
            return false;
        }

        // add once more ref
        decoder_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

        // open the codec
        if ((err = avcodec_open2(decoder_ctx, decoder, nullptr)) < 0) {
            fprintf(stderr, "failed to open codec for stream\n");
            return false;
        }
        return true;
    }

    bool open_software_coder() const {
        int ret = 0;
        ret = avcodec_open2(decoder_ctx, decoder, nullptr);
        if (ret < 0) {
            fprintf(stderr, "could not open codec %d\n", ret);
            return false;
        }
        return true;
    }

    bool open_decoder(AVCodecParameters *stream) {
        auto notify_opened = [this](bool open_success) {

            auto codec_par = avcodec_parameters_alloc();
            avcodec_parameters_from_context(codec_par, decoder_ctx);

            DecodeUrlInfo decode_url_info{};
            decode_url_info.hw_type = hw_type;
            decode_url_info.hw_pixfmt = hw_pixfmt;
            decode_url_info.open_success = open_success;
            decode_url_info.codec_par = std::move(codec_par);
            on_ready_(decode_url_info);
            
            return open_success;
        };
        if (init_pixfmt(stream, options_.prefer_hw)) {
            if (init_context(stream)) {
                if (hardware_decode_device_available) {
                    if (init_hardware()) {
                        return notify_opened(true);
                    }
                } else {
                    if (open_software_coder()) {
                        return notify_opened(true);
                    }
                }
            }
        }
        return notify_opened(false);
    }

    int filter_packet(AVPacket *packet, int64_t packet_index) const {
        return bit_stream_filter->send_packet(packet, packet_index);
    }

    int decode_packet(AVPacket *packet, int64_t packet_index) {
        int ret = 0;
        ret = avcodec_send_packet(decoder_ctx, packet);
        if (ret < 0) {
            fprintf(stderr, "error during decoding %d\n", ret);
            return ret;
        }

        AVFrame *frame = nullptr;
        AVFrame *sw_frame = nullptr;
        AVFrame *tmp_frame = nullptr;
        while (true) {
            frame = av_frame_alloc();
            sw_frame = av_frame_alloc();
            if (!frame || !sw_frame) {
                fprintf(stderr, "can not alloc frame\n");
                return AVERROR(ENOMEM);
            }

            ret = avcodec_receive_frame(decoder_ctx, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                ret = 0;
                break;
            } else if (ret < 0) {
                fprintf(stderr, "error while decoding===========\n");
                break;
            }

            if (frame->format == hw_pixfmt) {
                /* retrieve data from GPU to CPU */
                ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                if (ret < 0) {
                    fprintf(stderr, "Error transferring the data to system memory\n");
                    break;
                }
                // Need copy same basic info
                av_frame_copy_props(sw_frame, frame);
                tmp_frame = sw_frame;
            } else {
                tmp_frame = frame;
            }

            on_frame_(tmp_frame, packet_index);
            av_frame_free(&frame);
            av_frame_free(&sw_frame);
        }
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        return ret;
    }
};
}

#endif //FFMPEG_WRAPPER_SRC_DECODE_VIDEO_IMPL_HPP_
