#include "decode_video_v3.h"
#include "basic_logs.hpp"
#include "bitstream_filter_v3.hpp"
#include <atomic>

namespace av_wrapper {

static void dump_codec_info(const AVCodec *codec) {
    spdlog::info("---------------- Codec Info -----------------");
    spdlog::info("Long Name: {}", codec->long_name);
    spdlog::info("Short Name: {}", codec->name);
    spdlog::info("Codec Name: {}", avcodec_get_name(codec->id));
    spdlog::info("-------------------------------------------------");
}

static AVPixelFormat find_decoder_hw_config(AVCodec *decoder, AVHWDeviceType type) {
    for (int i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
        if (!config) {
            spdlog::error("Decoder {} does not support device type {}", decoder->name, av_hwdevice_get_type_name(type));
            return AV_PIX_FMT_NONE;
        }

        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == type) {
            return config->pix_fmt;
        }
    }
}

class DecoderPrivate {
public:
    DecoderPrivate(const Decoder_v3::OpenCallback &open_cb, const Decoder_v3::DecodeCallback &decode_cb)
        : bs_filter(std::make_unique<BitStreamFilter_v3>()), open_cb_(open_cb), decode_cb_(decode_cb) {}

    ~DecoderPrivate() {
        open_cb_ = nullptr;
        decode_cb_ = nullptr;
        av_dict_free(&dicts_);
    }

    bool open_decoder_impl(const std::shared_ptr<AVCodecParameters> &codecpar, const AVHWDeviceType type);

    bool filter_packet(const std::shared_ptr<AVPacket> &packet);

private:
    Decoder_v3::OpenCallback open_cb_;
    Decoder_v3::DecodeCallback decode_cb_;

    int64_t frame_idx = 1;

    AVPixelFormat hw_pixfmt{AV_PIX_FMT_NONE};
    std::unique_ptr<BitStreamFilter_v3> bs_filter{nullptr};

    AVDictionary *dicts_ = nullptr;
    std::unique_ptr<AVCodecContext, void (*)(AVCodecContext *)> codec_ctx{nullptr, nullptr};

    static AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
        auto _this = reinterpret_cast<DecoderPrivate *>(ctx->opaque);
        if (_this) {
            auto hw_pix_fmt = _this->hw_pixfmt;
            for (const enum AVPixelFormat *p = pix_fmts; *p != -1; p++) {
                if (*p == hw_pix_fmt) return *p;
            }
            spdlog::error("Failed to get HW surface format");
        }
        return AV_PIX_FMT_NONE;
    }
};

bool DecoderPrivate::open_decoder_impl(const std::shared_ptr<AVCodecParameters> &codecpar, const AVHWDeviceType type) {
    codec_ctx.reset();
    av_dict_free(&dicts_);

    const AVCodec *decoder = nullptr;
#ifdef WITH_BM1684
    if (codecpar->codec_id == AV_CODEC_ID_MJPEG && codecpar->profile == FF_PROFILE_MJPEG_HUFFMAN_BASELINE_DCT) {
        decoder = avcodec_find_decoder_by_name("jpeg_bm");
    } else {
        decoder = avcodec_find_decoder(codecpar->codec_id);
    }
#elif defined(WITH_MLU220) || defined(WITH_MLU270) || defined(WITH_MLU370)
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264: decoder = avcodec_find_decoder_by_name("h264_mludec"); break;
        case AV_CODEC_ID_HEVC: decoder = avcodec_find_decoder_by_name("hevc_mludec"); break;
        case AV_CODEC_ID_VP8: decoder = avcodec_find_decoder_by_name("vp8_mludec"); break;
        case AV_CODEC_ID_VP9: decoder = avcodec_find_decoder_by_name("vp9_mludec"); break;
        case AV_CODEC_ID_MJPEG: decoder = avcodec_find_decoder_by_name("mjpeg_mludec"); break;
        default: decoder = avcodec_find_decoder(codecpar->codec_id); break;
    }
#elif defined(WITH_JETSON)
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264: decoder = avcodec_find_decoder_by_name("h264_nvv4l2dec"); break;
        case AV_CODEC_ID_HEVC: decoder = avcodec_find_decoder_by_name("hevc_nvv4l2dec"); break;
        case AV_CODEC_ID_VP8: decoder = avcodec_find_decoder_by_name("vp8_nvv4l2dec"); break;
        case AV_CODEC_ID_VP9: decoder = avcodec_find_decoder_by_name("vp9_nvv4l2dec"); break;
        case AV_CODEC_ID_MPEG2TS: decoder = avcodec_find_decoder_by_name("mpeg2_nvv4l2dec"); break;
        case AV_CODEC_ID_MPEG4: decoder = avcodec_find_decoder_by_name("mpeg4_nvv4l2dec"); break;
        default: decoder = avcodec_find_decoder(codecpar->codec_id); break;
    }
#elif defined(WITH_RV1126)
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264: decoder = avcodec_find_decoder_by_name("h264_rkmpp"); break;
        case AV_CODEC_ID_HEVC: decoder = avcodec_find_decoder_by_name("hevc_rkmpp"); break;
        case AV_CODEC_ID_VP8: decoder = avcodec_find_decoder_by_name("vp8_rkmpp"); break;
        case AV_CODEC_ID_VP9: decoder = avcodec_find_decoder_by_name("vp9_rkmpp"); break;
        default: decoder = avcodec_find_decoder(codecpar->codec_id); break;
    }
#elif defined(WITH_TX5368)
    switch (codecpar->codec_id) {
        case AV_CODEC_ID_H264: decoder = avcodec_find_decoder_by_name("h264_tsmpp"); break;
        case AV_CODEC_ID_HEVC: decoder = avcodec_find_decoder_by_name("hevc_tsmpp"); break;
        case AV_CODEC_ID_VP8: decoder = avcodec_find_decoder_by_name("vp8_tsmpp"); break;
        case AV_CODEC_ID_VP9: decoder = avcodec_find_decoder_by_name("vp9_tsmpp"); break;
        default: decoder = avcodec_find_decoder(codecpar->codec_id); break;
    }
#else
    decoder = avcodec_find_decoder(codecpar->codec_id);
#endif

    if (decoder == nullptr) {
        spdlog::error("Failed to find decoder: {}", avcodec_get_name(codecpar->codec_id));
        return false;
    }

    bs_filter->register_packet_callback([this](const std::shared_ptr<AVPacket> &packet) {
        if (avcodec_send_packet(codec_ctx.get(), packet.get()) < 0) {
            throw std::runtime_error("Error during decoding");
        }

        while (true) {
            auto avframe = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame *ptr) { av_frame_free(&ptr); });

            int ret = avcodec_receive_frame(codec_ctx.get(), avframe.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                throw std::runtime_error("Error during receive frame, error code: " + std::to_string(ret));
            }
            if (decode_cb_) { decode_cb_(frame_idx++, avframe); }
        }
    });

    bs_filter->init_filter_by_codec_name(decoder->name, codecpar);

    codec_ctx = std::unique_ptr<AVCodecContext, void (*)(AVCodecContext *)>(
        avcodec_alloc_context3(decoder), [](AVCodecContext *ptr) { avcodec_free_context(&ptr); });
    if (codec_ctx.get() == nullptr) {
        spdlog::error("Failed to alloc context of avcodec");
        return false;
    }

    if (avcodec_parameters_to_context(codec_ctx.get(), codecpar.get()) < 0) {
        spdlog::error("Failed in init_decoder_context");
        return false;
    }

    hw_pixfmt = find_decoder_hw_config(const_cast<AVCodec *>(decoder), type);
    if (hw_pixfmt != AV_PIX_FMT_NONE) {
        if (av_hwdevice_ctx_create(&codec_ctx->hw_device_ctx, type, 0, NULL, 0) < 0) {
            spdlog::error("Failed to create specified HW device: {}", type);
            return false;
        }
        codec_ctx->opaque = this;
        codec_ctx->get_format = DecoderPrivate::get_hw_format;
    }

#if defined(WITH_BM1684)
    av_dict_set_int(&dicts_, "extra_frame_buffer_num", 1, 0);
    av_dict_set_int(&dicts_, "chroma_interleave", 1, 0);
    av_dict_set_int(&dicts_, "bs_buffer_size", 20480, 0);
#elif defined(WITH_TX5368)
    av_dict_set_int(&dicts_, "height", codecpar->height, 0);
    av_dict_set_int(&dicts_, "width", codecpar->width, 0);
    av_dict_set_int(&dicts_, "framebuf_cnt", 25, 0);
    av_dict_set_int(&dicts_, "refframebuf_num", 25, 0);
#endif

    if (avcodec_open2(codec_ctx.get(), decoder, &dicts_) < 0) {
        spdlog::error("Failed to open codec for stream");
        return false;
    }

    if (open_cb_) {
        auto codec_parameters = std::shared_ptr<AVCodecParameters>(
            avcodec_parameters_alloc(), [](AVCodecParameters *ptr) { avcodec_parameters_free(&ptr); });
        avcodec_parameters_from_context(codec_parameters.get(), codec_ctx.get());
        open_cb_(codec_parameters);
    }

    dump_codec_info(decoder);

    return true;
}

bool DecoderPrivate::filter_packet(const std::shared_ptr<AVPacket> &packet) {
    try {
        bs_filter->send_packet(packet);
    } catch (const std::exception &e) {
        spdlog::error("{}, stream_index: {}, pts: {}", e.what(), packet->stream_index, packet->pts);
        return false;
    }
    return true;
}

Decoder_v3::Decoder_v3() {}
Decoder_v3::~Decoder_v3() {}

bool Decoder_v3::open_decoder(const std::shared_ptr<AVCodecParameters> &codecpar, const AVHWDeviceType type) {
    impl_ = std::make_unique<DecoderPrivate>(open_cb_, decode_cb_);
    return impl_->open_decoder_impl(codecpar, type);
}

bool Decoder_v3::decode_packet(const std::shared_ptr<AVPacket> &packet) {
    if (impl_) { return impl_->filter_packet(packet); }
    return false;
}

void Decoder_v3::register_open_callback(const OpenCallback &open_cb) { open_cb_ = open_cb; }

void Decoder_v3::register_deocde_callback(const DecodeCallback &decode_cb) { decode_cb_ = decode_cb; }

}// namespace av_wrapper