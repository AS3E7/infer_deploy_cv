#ifndef __BITSTRAM_FILTER_HPP_
#define __BITSTRAM_FILTER_HPP_

#include "av_lib.h"
#include <string.h>

class BitStreamFilter
{
public:
    explicit BitStreamFilter(std::function<int(const AVPacket *)> on_packet) : on_packet_(std::move(on_packet))
    {
        av_bit_stream_filter_ = nullptr;
        avbsf_context_ = nullptr;
    }

    ~BitStreamFilter()
    {
        if (avbsf_context_)
        {
            av_bsf_free(&avbsf_context_);
        }
        av_bit_stream_filter_ = nullptr;
    }

    bool setup_filter_by_decoder_name(const std::string &name, const AVCodecParameters *codecpar)
    {
        if (!strcmp(name.c_str(), "h264") || !strcmp(name.c_str(), "h264_bm"))
        {
            av_bit_stream_filter_ = av_bsf_get_by_name("h264_mp4toannexb");
        }
        else if (!strcmp(name.c_str(), "hevc") == 0 || !strcmp(name.c_str(), "hevc_bm") == 0)
        {
            // av_bit_stream_filter_ = av_bsf_get_by_name("hevc_mp4toannexb");
        }

        if (av_bit_stream_filter_)
        {
            init_bsf_alloc(codecpar);    
        }

        return true;
    }

    int send_packet(const AVPacket *packet)
    {
        int ret = 0;

        if (av_bit_stream_filter_)
        {
            // 1. push packet to filter
            auto packet_to_filter = av_packet_clone(packet);
            ret = av_bsf_send_packet(avbsf_context_, packet_to_filter);
            if (ret < 0)
            {
                av_packet_free(&packet_to_filter);
                fprintf(stderr, "error during decoding filter\n");
                return ret;
            }

            // 2. read packet from filter
            auto packet_result = av_packet_alloc();
            while (av_bsf_receive_packet(avbsf_context_, packet_result) == 0)
            {
                // packet ready
                ret = on_packet_(packet_result);
                // unref the
                av_packet_unref(packet_result);
            }

            // 3. free data
            av_packet_free(&packet_result);
            av_packet_free(&packet_to_filter);
        }
        else
        {
            ret = on_packet_(packet);
        }
        return ret;
    }

protected:
    void init_bsf_alloc(const AVCodecParameters *codecpar)
    {
        int ret = av_bsf_alloc(av_bit_stream_filter_, &avbsf_context_);
        if (ret < 0)
        {
            throw std::runtime_error("fail to alloc av_bsf_alloc");
        }
        ret = avcodec_parameters_copy(avbsf_context_->par_in, codecpar);
        if (ret < 0)
        {
            throw std::runtime_error("fail to alloc av bsf avcodec_parameters_copy");
        }
        av_bsf_init(avbsf_context_);
    }

protected:
    const AVBitStreamFilter *av_bit_stream_filter_;
    AVBSFContext *avbsf_context_;
    std::function<int(const AVPacket *)> on_packet_;
};

#endif