#include "mpp_jpeg_codec.h"
#include "spdlog/spdlog.h"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <rockchip/mpp_err.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_mpi.h>
#include <stdexcept>
#include <vector>

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))

namespace gddi {
namespace codec {

class MppJpegCodecPrivate {
public:
    MppJpegCodecPrivate() {}
    ~MppJpegCodecPrivate() {
        if (mpi_) { mpi_->reset(ctx_); }
        if (ctx_) { mpp_destroy(ctx_); }
        if (cfg_) { mpp_enc_cfg_deinit(cfg_); }
        if (buf_grp_) { mpp_buffer_group_put(buf_grp_); }
    }

    bool init_encoder(const size_t width, const size_t height, const int quality) {
        width_ = width;
        height_ = height;

        try {
            if ((ret_ = mpp_buffer_group_get_internal(&buf_grp_, MPP_BUFFER_TYPE_DRM)) != MPP_OK) {
                throw std::runtime_error("failed to get mpp buffer group");
            }

            if ((ret_ = mpp_create(&ctx_, &mpi_)) != MPP_OK) { throw std::runtime_error("mpp_create failed"); }

            MppPollType timeout = MPP_POLL_BLOCK;
            if ((ret_ = mpi_->control(ctx_, MPP_SET_OUTPUT_TIMEOUT, &timeout)) != MPP_OK) {
                throw std::runtime_error("mpi control set output timeout " + std::to_string(timeout));
            }

            if ((ret_ = mpp_init(ctx_, MPP_CTX_ENC, MPP_VIDEO_CodingMJPEG)) != MPP_OK) {
                throw std::runtime_error("mpp_init failed");
            }

            if ((ret_ = mpp_enc_cfg_init(&cfg_)) != MPP_OK) { throw std::runtime_error("mpp_enc_cfg_init failed"); }

            mpp_enc_cfg_set_s32(cfg_, "prep:width", width);
            mpp_enc_cfg_set_s32(cfg_, "prep:height", height);
            mpp_enc_cfg_set_s32(cfg_, "prep:hor_stride", ALIGN(width, 16));
            mpp_enc_cfg_set_s32(cfg_, "prep:ver_stride", ALIGN(height, 16));
            mpp_enc_cfg_set_s32(cfg_, "prep:format", format_);
            mpp_enc_cfg_set_s32(cfg_, "codec:type", MPP_VIDEO_CodingMJPEG);
            mpp_enc_cfg_set_s32(cfg_, "rc:mode", MPP_ENC_RC_MODE_FIXQP);
            mpp_enc_cfg_set_s32(cfg_, "jpeg:quant", quality / 10); /* range 0 - 10, worst -> best */

            if ((ret_ = mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg_)) != MPP_OK) {
                throw std::runtime_error("mpi control enc set cfg failed");
            }
        } catch (std::exception &ex) {
            spdlog::error("MppJpegCodecInit: {}", ex.what());
            return false;
        }

        return true;
    }

    bool encode(const u_char *image_data, const JpegCodecCallback &callback) {
        size_t buf_size = 0;
        switch (format_) {
            case MPP_FMT_YUV420P:
            case MPP_FMT_YUV420SP: buf_size = ALIGN(width_, 16) * ALIGN(height_, 16) * 3 / 2; break;
            case MPP_FMT_BGR888: buf_size = ALIGN(width_, 16) * ALIGN(height_, 16) * 3; break;
            default: return false;
        }

        try {
            MppBuffer frame_buf{nullptr};
            if ((ret_ = mpp_buffer_get(buf_grp_, &frame_buf, buf_size)) != MPP_OK) {
                throw std::runtime_error("buffer_get failed");
            }

            MppFrame frame{nullptr};
            mpp_frame_init(&frame);
            mpp_frame_set_width(frame, width_);
            mpp_frame_set_height(frame, height_);
            mpp_frame_set_hor_stride(frame, ALIGN(width_, 16));
            mpp_frame_set_ver_stride(frame, ALIGN(height_, 16));
            mpp_frame_set_fmt(frame, format_);
            mpp_frame_set_buffer(frame, frame_buf);
            mpp_frame_set_eos(frame, 0);

            if (format_ == MPP_FMT_YUV420P) {
                memcpy(mpp_buffer_get_ptr(frame_buf), image_data, width_ * height_);
                memcpy((char *)mpp_buffer_get_ptr(frame_buf) + ALIGN(width_, 16) * ALIGN(height_, 16),
                       image_data + width_ * height_, width_ * height_ / 4);
                memcpy((char *)mpp_buffer_get_ptr(frame_buf) + ALIGN(width_, 16) * ALIGN(height_, 16) * 5 / 4,
                       image_data + width_ * height_ * 5 / 4, width_ * height_ / 4);
            } else if (format_ == MPP_FMT_YUV420SP) {
                memcpy(mpp_buffer_get_ptr(frame_buf), image_data, width_ * height_ * 3 / 2);
            } else if (format_ == MPP_FMT_BGR888) {
                memcpy(mpp_buffer_get_ptr(frame_buf), image_data, width_ * height_ * 3);
            }

            if ((ret_ = mpi_->encode_put_frame(ctx_, frame)) != MPP_OK) {
                throw std::runtime_error("put_frame failed");
            }
            mpp_frame_deinit(&frame);
            mpp_buffer_put(frame_buf);

            MppPacket packet{nullptr};
            if ((ret_ = mpi_->encode_get_packet(ctx_, &packet)) != MPP_OK) {
                throw std::runtime_error("get_packet failed");
            }

            if (packet) {
                callback((uint8_t *)mpp_packet_get_pos(packet), mpp_packet_get_length(packet));
                // auto pkt_eos = mpp_packet_get_eos(packet);
                mpp_packet_deinit(&packet);
            }
        } catch (std::exception &ex) {
            spdlog::error("MppJpegEnCodec: {}", ex.what());
            return false;
        }

        return true;
    }

private:
    MppCtx ctx_{nullptr};
    MppApi *mpi_{nullptr};
    MppEncCfg cfg_{nullptr};

    size_t width_;
    size_t height_;

    MppBufferGroup buf_grp_{nullptr};

    MppFrameFormat format_ = MPP_FMT_YUV420SP;

    MPP_RET ret_ = MPP_NOK;
};

MppJpegCodec::MppJpegCodec(const CodecType type) : codec_type_(type), impl_(std::make_unique<MppJpegCodecPrivate>()) {}

MppJpegCodec::~MppJpegCodec() {}

bool MppJpegCodec::init_codecer(const size_t width, const size_t height, const int quality) {
    if (codec_type_ == CodecType::kEncoder) { return impl_->init_encoder(width, height, quality); }
    return false;
}

bool MppJpegCodec::codec_image(const u_char *input, const JpegCodecCallback &callback) {
    return impl_->encode(input, callback);
}

const bool MppJpegCodec::save_image(const u_char *input, const char *path) {
    return impl_->encode(input, [&path](const uint8_t *data, const size_t dlen) {
        FILE *file = fopen(path, "wb");
        fwrite(data, sizeof(char), dlen, file);
        fclose(file);
    });
}

}// namespace codec
}// namespace gddi