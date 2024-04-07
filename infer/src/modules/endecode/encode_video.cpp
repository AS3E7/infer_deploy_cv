//
// Created by cc on 2021/11/09.
//

#include "encode_video.h"

namespace av_wrapper
{
    static AVPixelFormat _convert_deprecated_format(enum AVPixelFormat format)
    {
        switch (format)
        {
        case AV_PIX_FMT_YUVJ420P:
            return AV_PIX_FMT_YUV420P;
            break;
        case AV_PIX_FMT_YUVJ422P:
            return AV_PIX_FMT_YUV422P;
            break;
        case AV_PIX_FMT_YUVJ444P:
            return AV_PIX_FMT_YUV444P;
            break;
        case AV_PIX_FMT_YUVJ440P:
            return AV_PIX_FMT_YUV440P;
            break;
        default:
            return format;
            break;
        }
    }
    
    static AVPixelFormat find_decoder_hw_config(AVCodec *decoder, AVHWDeviceType type)
    {
        for (int i = 0;; i++)
        {
            const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
            if (!config)
            {
                fprintf(stderr, "Decoder %s does not support device type %s.\n",
                        decoder->name, av_hwdevice_get_type_name(type));
                return AV_PIX_FMT_NONE;
            }

            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                config->device_type == type)
            {
                return config->pix_fmt;
            }
        }
    }

    struct VideoEncoder::Impl
    {
        const AVCodec       *encoder = nullptr;
        AVCodecContext      *encoder_ctx = nullptr;

        uint64_t            frame_count = 0;

        AVBufferRef         *hw_device_ctx = nullptr;
        AVHWFramesContext   *frames_ctx = nullptr;
        enum AVPixelFormat  hw_pix_fmt;

        AVPacket            *packet = nullptr;
    };

    VideoEncoder::VideoEncoder() : impl_(std::make_unique<VideoEncoder::Impl>())
    {
        impl_->packet = av_packet_alloc();
    }

    VideoEncoder::~VideoEncoder()
    {
        close_enc();
    }

    bool VideoEncoder::open_enc(const EcnodeOptions &options, on_open_t on_open)
    {
        try
        {
            impl_->encoder = avcodec_find_encoder_by_name(options.codec.c_str());

            if (!impl_->encoder)
            {
                throw std::runtime_error("Necessary VideoEncoder not found");
            }

            impl_->encoder_ctx = avcodec_alloc_context3(impl_->encoder);
            if (!impl_->encoder_ctx)
            {
                throw std::runtime_error("Failed to allocate the encoder context");
            }

            impl_->encoder_ctx->codec_id = impl_->encoder->id;
            impl_->encoder_ctx->width = options.width;
            impl_->encoder_ctx->height = options.height;

#ifdef WITH_BM1684
            impl_->encoder_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
#else
            impl_->encoder_ctx->pix_fmt = options.pix_fmt;
#endif
            impl_->encoder_ctx->bit_rate_tolerance = options.bitrate;
            impl_->encoder_ctx->bit_rate = options.bitrate;
            impl_->encoder_ctx->time_base = {1, options.framerate};
            impl_->encoder_ctx->framerate = {options.framerate, 1};
            impl_->encoder_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER | AV_CODEC_FLAG_LOW_DELAY;
            impl_->encoder_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
            impl_->encoder_ctx->gop_size = options.gop_size;

#ifdef __x86_64
            impl_->encoder_ctx->max_b_frames = 0;
#endif
            impl_->encoder_ctx->profile = FF_PROFILE_HEVC_MAIN;

            if (impl_->encoder_ctx->pix_fmt == AV_PIX_FMT_CUDA)
            {
                if (av_hwdevice_ctx_create(&impl_->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL, NULL, 0) < 0)
                {
                    throw std::runtime_error("Failed to create specified HW device");
                }

                if (!(impl_->encoder_ctx->hw_frames_ctx = av_hwframe_ctx_alloc(impl_->hw_device_ctx)))
                {
                    throw std::runtime_error("Failed to create cuda frame context");
                }

                impl_->frames_ctx = (AVHWFramesContext *)(impl_->encoder_ctx->hw_frames_ctx->data);
                impl_->frames_ctx->format = AV_PIX_FMT_CUDA;
                impl_->frames_ctx->sw_format = AV_PIX_FMT_NV12;
                impl_->frames_ctx->width = options.width;
                impl_->frames_ctx->height = options.height;
                impl_->frames_ctx->initial_pool_size = 20;

                if (av_hwframe_ctx_init(impl_->encoder_ctx->hw_frames_ctx) < 0)
                {
                    av_buffer_unref(&impl_->encoder_ctx->hw_frames_ctx);
                    throw std::runtime_error("Failed to initialize CUDA frame context");
                }
            }
            
            AVDictionary *dict = nullptr;
#ifdef WITH_BM1684
            av_dict_set_int(&dict, "sophon_idx", 0, 0);
            av_dict_set_int(&dict, "gop_preset", 8, 0);
            av_dict_set_int(&dict, "is_dma_buffer", 1 , 0);
#else
            // av_dict_set(&opts, "preset", "superfast", 0);
            av_dict_set(&dict, "flags2", "+export_mvs", 0);
#endif

            if (avcodec_open2(impl_->encoder_ctx, impl_->encoder, &dict) < 0)
            {
                av_dict_free(&dict);
                throw std::runtime_error("Cannot open video encoder codec");
            }

            av_dict_free(&dict);

            if (on_open)
            {
                auto codec_parameters = avcodec_parameters_alloc();
                avcodec_parameters_from_context(codec_parameters, impl_->encoder_ctx);
                on_open(codec_parameters, impl_->encoder_ctx->time_base, impl_->encoder_ctx->framerate);
            }
        }
        catch(std::exception &e)
        {
            printf("%s\n", e.what());
            return false;
        }

        return true;
    }

    bool VideoEncoder::enc_frame(AVFrame *src_frame, std::function<void(const AVPacket *packet)> on_packet)
    {
        if (impl_->encoder_ctx == nullptr)
            return false;

        bool res_code = true;
        AVFrame *enc_frame = src_frame;

        try
        {    
            std::shared_ptr<uint8_t> buffer;
            if (src_frame->width != impl_->encoder_ctx->width || src_frame->height != impl_->encoder_ctx->height)
            {
                enc_frame = av_frame_alloc();
                enc_frame->width = impl_->encoder_ctx->width;
                enc_frame->height = impl_->encoder_ctx->height;
                enc_frame->format = _convert_deprecated_format((AVPixelFormat)src_frame->format);
                buffer = std::shared_ptr<uint8_t>((uint8_t *)av_malloc(
                    av_image_get_buffer_size((AVPixelFormat)enc_frame->format, impl_->encoder_ctx->width, impl_->encoder_ctx->height, 1)), [](uint8_t *buf) {
                        av_free(buf);
                });
                av_image_fill_arrays(enc_frame->data, enc_frame->linesize, buffer.get(), (AVPixelFormat)enc_frame->format, enc_frame->width, enc_frame->height, 1);

                SwsContext *conversion = sws_getContext(
                    src_frame->width, src_frame->height, (AVPixelFormat)enc_frame->format,
                    enc_frame->width, enc_frame->height, (AVPixelFormat)enc_frame->format,
                    SWS_FAST_BILINEAR, NULL, NULL, NULL);
                sws_scale(conversion, src_frame->data, src_frame->linesize, 0, src_frame->height, enc_frame->data, enc_frame->linesize);
                sws_freeContext(conversion);
            }

            if (impl_->hw_device_ctx)
            {
                AVFrame *hw_frame = av_frame_alloc();
                if (!hw_frame)
                {
                    throw std::runtime_error("Can not alloc frame");
                }

                if (av_hwframe_get_buffer(impl_->encoder_ctx->hw_frames_ctx, hw_frame, 0) < 0)
                {
                    throw std::runtime_error("Failed to convert hwframe");
                }

                if (!hw_frame->hw_frames_ctx)
                {
                     throw std::runtime_error("Frame is not hwaccel pixel format");
                }

                if (av_hwframe_transfer_data(hw_frame, enc_frame, 0) < 0)
                {
                    throw std::runtime_error("Error while transferring frame data to surface");
                }

                av_frame_copy_props(hw_frame, enc_frame);

                hw_frame->pts = impl_->frame_count++;
                if (avcodec_send_frame(impl_->encoder_ctx, hw_frame) < 0)
                {
                    throw std::runtime_error("Error sending a frame for encoding");
                }

                av_frame_free(&hw_frame);
            }
            else
            {
                enc_frame->pts = impl_->frame_count++;
                if (avcodec_send_frame(impl_->encoder_ctx, enc_frame) < 0)
                {
                    throw std::runtime_error("Error sending a frame for encoding");
                }
            }

            while (true)
            {
                int ret = avcodec_receive_packet(impl_->encoder_ctx, impl_->packet);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                else if (ret < 0)
                    throw std::runtime_error("Error during encoding");

                impl_->packet->stream_index = 0;

                if (on_packet)
                    on_packet(impl_->packet);
                
                av_packet_unref(impl_->packet);
            }
        }
        catch(const std::exception& e)
        {
            av_packet_unref(impl_->packet);
            std::cerr << e.what() << '\n';
            res_code = false;
        }

        if (enc_frame != src_frame)
        {
            av_frame_free(&enc_frame);
        }

        return res_code;
    }

    void VideoEncoder::close_enc()
    {
        if (impl_->hw_device_ctx)
        {
            av_buffer_unref(&impl_->hw_device_ctx);
            impl_->hw_device_ctx = nullptr;
        }

        if (impl_->encoder_ctx)
        {
            avcodec_free_context(&impl_->encoder_ctx);
            impl_->encoder_ctx = nullptr;
        }

        av_packet_free(&impl_->packet);
        impl_->packet = nullptr;
    }
}