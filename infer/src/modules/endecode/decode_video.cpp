//
// Created by cc on 2021/10/29.
//

#include "decode_video.hpp"
#include "debug_tools.hpp"
#include "decode_video_impl.hpp"

namespace av_wrapper {
std::pair<AVHWDeviceType, AVPixelFormat> find_decoder_hw_config2(AVCodec *decoder, AVHWDeviceType prefer_hw_type) {
    AVHWDeviceType pre_hw_device_type = AV_HWDEVICE_TYPE_NONE;
    std::vector<std::pair<AVHWDeviceType, AVPixelFormat>> available_pixfmts;

    // list all supported hw for codec
    while ((pre_hw_device_type = av_hwdevice_iterate_types(pre_hw_device_type)) != AV_HWDEVICE_TYPE_NONE) {
        std::cout << "Hardware: " << av_hwdevice_get_type_name(pre_hw_device_type) << std::endl;
        for (int i = 0;; i++) {
            const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
            if (!config) {
                fprintf(stderr, "\tdecoder <%s> does not support device type %s.\n",
                        decoder->name, av_hwdevice_get_type_name(pre_hw_device_type));
                break;
            }

            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                config->device_type == pre_hw_device_type) {
                std::cout << "    " << std::setw(2) << i
                          << ", PixFmt: " << std::setw(16) << av_get_pix_fmt_name(config->pix_fmt)
                          << ", " << config->methods
                          << std::endl;
                available_pixfmts.emplace_back(config->device_type, config->pix_fmt);
                break;
            }
        }
    }

    if (available_pixfmts.empty()) { return {AV_HWDEVICE_TYPE_NONE, AV_PIX_FMT_NONE}; }

    // find if prefer hw type is available
    for (const auto &pixfmt: available_pixfmts) {
        if (pixfmt.first == prefer_hw_type) {
            return pixfmt;
        }
    }
    return available_pixfmts.front();
}
}

namespace av_wrapper {
ProcessResult DecodeUrl(
    bool &stop_signal,
    const DecodeUrlOptions &options,
    const std::string &stream_url,
    const std::function<void(const DecodeUrlInfo &)> &on_ready,
    const std::function<void(AVFrame *, int64_t)> &on_frame
) {
    int64_t demux_packet_count = 0;
    VideoDecoder video_decoder(on_ready, on_frame, options);
    std::string error_on_packet_process;
    stop_signal = false;

    // demux stream, video only
    auto &&result = DemuxStream(stop_signal, stream_url, [&](AVStream *stream) {
        return video_decoder.open_decoder(stream->codecpar);
    }, [&](AVPacket *packet) {
        demux_packet_count++;
        if (video_decoder.filter_packet(packet, demux_packet_count) < 0) {
            stop_signal = true;
            error_on_packet_process = "fail to filter packet";
        }
    });
    return result;
}

}