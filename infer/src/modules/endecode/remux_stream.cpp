#include "remux_stream.h"

namespace av_wrapper
{
    struct Remuxer::Impl
    {
        AVFormatContext     *out_fmt_ctx = nullptr;
        AVStream            *out_stream = nullptr;
        int64_t             encoded_frames = 0;
        AVDictionary        *format_opts = nullptr;

        std::shared_ptr<AVCodecParameters> codec_par;
        AVRational          codec_time_base;
        AVRational          codec_framerate;

        bool                is_exit = false;
    };

    Remuxer::Remuxer() : impl_(std::make_unique<Impl>())
    {
    }

    Remuxer::~Remuxer()
    {
        close();
    }

    void Remuxer::init(const AVCodecParameters *codecpar, const AVRational &timebase, const AVRational &framerate)
    {
        impl_->codec_time_base = timebase;
        impl_->codec_framerate = framerate;
        impl_->codec_par = std::shared_ptr<AVCodecParameters>(avcodec_parameters_alloc(), [] (AVCodecParameters *ptr) {
            avcodec_parameters_free(&ptr);
        });
        avcodec_parameters_copy(impl_->codec_par.get(), codecpar);
    }

    bool Remuxer::open(const std::string &stream_url)
    {
        try
        {
            if (stream_url.substr(0, 4) == "rtsp")
            {
                avformat_alloc_output_context2(&impl_->out_fmt_ctx, NULL, "rtsp", stream_url.c_str());
                av_dict_set(&impl_->format_opts, "rtsp_transport", "tcp", 0);
            }
            else if (stream_url.substr(0, 4) == "rtmp")
            {
                avformat_alloc_output_context2(&impl_->out_fmt_ctx, NULL, "flv", stream_url.c_str());
            }
            else
            {
                avformat_alloc_output_context2(&impl_->out_fmt_ctx, NULL, NULL, stream_url.c_str());
            }

            /*-------------- set callback, avoid blocking --------------*/
            // impl_->out_fmt_ctx->interrupt_callback.callback = [] (void *opaque) {
            //     bool *is_exit = (bool*)opaque;
            //     if (*is_exit) {
            //         return 1;
            //     }
            //     return 0;
            // };
            // impl_->out_fmt_ctx->interrupt_callback.opaque = &impl_->is_exit;
            
            if (!impl_->out_fmt_ctx)
            {
                throw std::runtime_error("FCould not create output context");
            }

            if (!(impl_->out_stream = avformat_new_stream(impl_->out_fmt_ctx, nullptr)))
            {
                throw std::runtime_error("Failed allocating output stream");
            }

            impl_->out_stream->codecpar->codec_tag = 0;
            impl_->out_stream->time_base       = impl_->codec_time_base;
            impl_->out_stream->avg_frame_rate  = impl_->codec_framerate;
            impl_->out_stream->r_frame_rate    = impl_->out_stream->avg_frame_rate;
            avcodec_parameters_copy(impl_->out_stream->codecpar, impl_->codec_par.get());

            av_dump_format(impl_->out_fmt_ctx, 0, stream_url.c_str(), 1);

#if !defined(WITH_TX5368)
            if (stream_url.substr(0, 4) == "rtsp")
            {
                impl_->out_fmt_ctx->oformat->flags |= AVFMT_NOFILE;
            }

            if (impl_->out_fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
                impl_->out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#endif

            if (!(impl_->out_fmt_ctx->oformat->flags & AVFMT_NOFILE))
            {
                if (avio_open(&impl_->out_fmt_ctx->pb, stream_url.c_str(), AVIO_FLAG_WRITE) < 0)
                {
                    throw std::runtime_error("Could not open output file " + stream_url);
                }
            }

            if (avformat_write_header(impl_->out_fmt_ctx, &impl_->format_opts) < 0)
            {
                throw std::runtime_error("Occurred when opening url: " + stream_url);
            }
        }
        catch(const std::exception& e)
        {
            printf("%s\n", e.what());
            avformat_free_context(impl_->out_fmt_ctx);
            impl_->out_fmt_ctx = nullptr;
            return false;
        }

        return true;
    }

    bool Remuxer::write(AVPacket *packet)
    {
        if (impl_->out_fmt_ctx)
        {
            packet->stream_index = impl_->out_stream->index;
            av_packet_rescale_ts(packet, impl_->codec_time_base, impl_->out_stream->time_base);
            return av_interleaved_write_frame(impl_->out_fmt_ctx, packet) == 0;
        }
        return false;
    }

    void Remuxer::close()
    {
        if(impl_->format_opts)
        {
            av_dict_free(&impl_->format_opts);
            impl_->format_opts = nullptr;
        }

        if (impl_->out_fmt_ctx)
        {
            if (!(impl_->out_fmt_ctx->oformat->flags & AVFMT_NOFILE))
            {
                av_write_trailer(impl_->out_fmt_ctx);
                avio_closep(&impl_->out_fmt_ctx->pb);
            }
            avformat_free_context(impl_->out_fmt_ctx);
            impl_->out_fmt_ctx = nullptr;
        }
    }
}