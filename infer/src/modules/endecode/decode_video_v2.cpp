#include "decode_video_v2.h"
#include "basic_logs.hpp"
#include "bitstream_filter.hpp"

namespace av_wrapper {
std::ostream &dump_codec_info(std::ostream &oss, const AVCodec *codec) {
    auto flags = oss.flags();
    oss << "---------------- Codec Info -----------------" << '\n'
        << std::setw(16) << std::setiosflags(std::ios::left) << "Long Name" << std::setw(20)
        << std::setiosflags(std::ios::left) << codec->long_name << '\n'
        << std::setw(16) << std::setiosflags(std::ios::left) << "Short Name" << std::setw(20)
        << std::setiosflags(std::ios::left) << codec->name << '\n'
        << std::setw(16) << std::setiosflags(std::ios::left) << "Codec Name" << std::setw(20)
        << std::setiosflags(std::ios::left) << avcodec_get_name(codec->id) << '\n'
        << "-------------------------------------------------" << std::endl;

    oss.flags(flags);
    return oss;
}

void dump_codec_info(const AVCodec *codec) {
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
            fprintf(stderr, "Decoder %s does not support device type %s.\n", decoder->name,
                    av_hwdevice_get_type_name(type));
            return AV_PIX_FMT_NONE;
        }

        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX
            && config->device_type == type) {
            return config->pix_fmt;
        }
    }
}

struct VideoDecoder_v2::Impl {
    AVPixelFormat hw_pixfmt = AV_PIX_FMT_NONE;

    AVCodecContext *decoder_ctx = nullptr;// free required
    int64_t frame_idx = 0;

    on_frame_t on_frame;
    on_open_t on_open;

    std::unique_ptr<BitStreamFilter> bsf_ptr{};

    static AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
        auto _this = reinterpret_cast<VideoDecoder_v2 *>(ctx->opaque);
        if (_this) {
            auto hw_pix_fmt = _this->impl_->hw_pixfmt;
            for (const enum AVPixelFormat *p = pix_fmts; *p != -1; p++) {
                if (*p == hw_pix_fmt) return *p;
            }
            fprintf(stderr, "Failed to get HW surface format.\n");
        }
        return AV_PIX_FMT_NONE;
    }
};

VideoDecoder_v2::VideoDecoder_v2() : impl_(std::make_unique<VideoDecoder_v2::Impl>()) {}

VideoDecoder_v2::~VideoDecoder_v2() { close_decode(); }

void VideoDecoder_v2::close_decode() {
    if (impl_->decoder_ctx) { avcodec_free_context(&impl_->decoder_ctx); }
    impl_->decoder_ctx = nullptr;
    impl_->hw_pixfmt = AV_PIX_FMT_NONE;
    impl_->frame_idx = 1;
}

void VideoDecoder_v2::set_decoder_on_open(on_open_t on_open) {
    impl_->on_open = std::move(on_open);
}

void VideoDecoder_v2::set_decoder_callback(on_frame_t on_frame) {
    impl_->on_frame = std::move(on_frame);
}

bool VideoDecoder_v2::open_decoder(const AVCodecParameters *codecpar, const AVHWDeviceType type) {
    close_decode();

    try {

        const AVCodec *decoder = nullptr;
#ifdef WITH_BM1684
        if (codecpar->codec_id == AV_CODEC_ID_MJPEG) {
            decoder = avcodec_find_decoder_by_name("jpeg_bm");
        } else {
            decoder = avcodec_find_decoder(codecpar->codec_id);
        }
#else
        decoder = avcodec_find_decoder(codecpar->codec_id);
#endif

        if (decoder) {
            // dump_codec_info(std::cout, decoder) << std::endl;
            dump_codec_info(decoder);

            impl_->bsf_ptr = std::make_unique<BitStreamFilter>(
                [this](const AVPacket *packet) { return decode_one_packet(packet); });

            if (!impl_->bsf_ptr->setup_filter_by_decoder_name(decoder->name, codecpar)) {
                throw std::runtime_error("Fail to setup filter");
            }

            if ((impl_->decoder_ctx = avcodec_alloc_context3(decoder)) == nullptr) {
                throw std::runtime_error("Fail to alloc context of avcodec");
            }

            if (avcodec_parameters_to_context(impl_->decoder_ctx, codecpar) < 0) {
                throw std::runtime_error("Fail in init_decoder_context");
            }

            impl_->decoder_ctx->opaque = this;

            if (type != AV_HWDEVICE_TYPE_NONE) {
                impl_->hw_pixfmt = find_decoder_hw_config(const_cast<AVCodec *>(decoder), type);
                if (impl_->hw_pixfmt != AV_PIX_FMT_NONE) {
                    impl_->decoder_ctx->get_format = Impl::get_hw_format;
                    if (av_hwdevice_ctx_create(&impl_->decoder_ctx->hw_device_ctx, type, NULL, NULL,
                                               0)
                        < 0) {
                        throw std::runtime_error("Failed to create specified HW device");
                    }
                }
            }

            AVDictionary *opts = nullptr;

#ifdef WITH_BM1684
            // av_dict_set_int(&opts, "output_format", 101, 0);
            av_dict_set_int(&opts, "extra_frame_buffer_num", 18, 0);
            // av_dict_set_int(&opts, "enable_cache", 1, 0);
            av_dict_set_int(&opts, "chroma_interleave", 1, 0);
#endif

            if (avcodec_open2(impl_->decoder_ctx, decoder, &opts) < 0) {
                throw std::runtime_error("Failed to open codec for stream");
            }

            av_dict_free(&opts);

            if (impl_->on_open) {
                auto av_param = avcodec_parameters_alloc();
                avcodec_parameters_from_context(av_param, impl_->decoder_ctx);
                impl_->on_open(av_param);
                avcodec_parameters_free(&av_param);
            }

            return true;
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; }

    return false;
}

int VideoDecoder_v2::filter_packet(const AVPacket *packet) {
    return impl_->bsf_ptr->send_packet(packet);
}

int VideoDecoder_v2::decode_one_packet(const AVPacket *packet) {
    int ret = avcodec_send_packet(impl_->decoder_ctx, packet);
    if (ret < 0) {
        fprintf(stderr, "error during decoding %d\n", ret);
        return ret;
    }

    auto avframe =
        std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame *ptr) { av_frame_free(&ptr); });

    while (true) {
        ret = avcodec_receive_frame(impl_->decoder_ctx, avframe.get());
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            ret = 0;
            break;
        } else if (ret < 0) {
            fprintf(stderr, "Error while decoding");
            break;
        }

        if (impl_->on_frame) { impl_->on_frame(impl_->frame_idx++, avframe); }
    }

    return ret;
}
}// namespace av_wrapper