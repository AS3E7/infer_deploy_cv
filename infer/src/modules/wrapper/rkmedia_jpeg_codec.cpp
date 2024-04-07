#include "rkmedia_jpeg_codec.h"
#include "spdlog/spdlog.h"
#include <exception>
#include <memory>
#include <rkmedia/rkmedia_api.h>
#include <thread>
#include <utility>
#include <vector>

namespace gddi {
namespace codec {

class RkMediaJpegCodecPrivate {
public:
    RkMediaJpegCodecPrivate() {}
    ~RkMediaJpegCodecPrivate() {
        RK_MPI_VENC_DestroyChn(enc_chn_.s32ChnId);
        RK_MPI_MB_ReleaseBuffer(mb_);
    }

    bool init_encoder(const size_t width, const size_t height, const int quality) {
        enc_chn_.enModId = RK_ID_VENC;
        enc_chn_.s32ChnId = channal_id_++;
        channal_id_ %= VENC_MAX_CHN_NUM;

        memset(&venc_chn_attr_, 0, sizeof(venc_chn_attr_));
        venc_chn_attr_.stVencAttr.enType = RK_CODEC_TYPE_JPEG;
        venc_chn_attr_.stVencAttr.imageType = IMAGE_TYPE_NV12;
        venc_chn_attr_.stVencAttr.u32PicWidth = width;
        venc_chn_attr_.stVencAttr.u32PicHeight = height;
        venc_chn_attr_.stVencAttr.u32VirWidth = width;
        venc_chn_attr_.stVencAttr.u32VirHeight = height;

        if (RK_MPI_VENC_CreateChn(enc_chn_.s32ChnId, &venc_chn_attr_) != 0) {
            spdlog::error("Create Venc failed!");
            return false;
        }

        VENC_RECV_PIC_PARAM_S stRecvParam;
        stRecvParam.s32RecvPicNum = 0;
        RK_MPI_VENC_StartRecvFrame(enc_chn_.s32ChnId, &stRecvParam);

        jpeg_param_.u32Qfactor = quality;
        if (RK_MPI_VENC_SetJpegParam(enc_chn_.s32ChnId, &jpeg_param_)) {
            spdlog::error("Set Jpeg  param failed!");
            return false;
        }

        image_info_ = {width, height, width, height, IMAGE_TYPE_NV12};
        if ((mb_ = RK_MPI_MB_CreateImageBuffer(&image_info_, RK_TRUE, MB_FLAG_NOCACHED)) == nullptr) {
            spdlog::error("Create image buffer!");
            return false;
        }

        if (RK_MPI_MB_SetSize(mb_, width * height * 3 / 2) != 0) {
            spdlog::error("No space left for image buffer!");
            return false;
        }

        return true;
    }

    bool encode(const u_char *image_data, const JpegCodecCallback &callback) {
        auto buffer = std::pair<bool, const JpegCodecCallback &>{false, callback};
        if (RK_MPI_SYS_RegisterOutCbEx(&enc_chn_, rk_jpeg_cb, (void *)&buffer) != 0) {
            spdlog::error("Register Output callback failed!");
            return false;
        }

        VENC_RECV_PIC_PARAM_S stRecvParam;
        stRecvParam.s32RecvPicNum = 1;
        if (RK_MPI_VENC_StartRecvFrame(enc_chn_.s32ChnId, &stRecvParam) != 0) {
            spdlog::error("RK_MPI_VENC_StartRecvFrame failed!");
            return false;
        }

        memcpy(RK_MPI_MB_GetPtr(mb_), image_data, image_info_.u32Width * image_info_.u32Height * 3 / 2);
        RK_MPI_SYS_SendMediaBuffer(RK_ID_VENC, enc_chn_.s32ChnId, mb_);

        while (!buffer.first) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }

        return true;
    }

    bool encode(const u_char *image_data, std::vector<uint8_t> &jepg_data) {
        auto buffer = std::pair<bool, std::vector<uint8_t> &>{false, jepg_data};
        if (RK_MPI_SYS_RegisterOutCbEx(&enc_chn_, rk_jpeg_sync_cb, (void *)&buffer) != 0) {
            spdlog::error("Register Output callback failed!");
            return false;
        }

        VENC_RECV_PIC_PARAM_S stRecvParam;
        stRecvParam.s32RecvPicNum = 1;
        if (RK_MPI_VENC_StartRecvFrame(enc_chn_.s32ChnId, &stRecvParam) != 0) {
            spdlog::error("RK_MPI_VENC_StartRecvFrame failed!");
            return false;
        }

        memcpy(RK_MPI_MB_GetPtr(mb_), image_data, image_info_.u32Width * image_info_.u32Height * 3 / 2);
        RK_MPI_SYS_SendMediaBuffer(RK_ID_VENC, enc_chn_.s32ChnId, mb_);

        while (!buffer.first) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }

        return true;
    }

protected:
    static void rk_jpeg_cb(MEDIA_BUFFER mb, void *data) {
        auto &buffer = *(std::pair<bool, const JpegCodecCallback &> *)data;
        buffer.second((uint8_t *)RK_MPI_MB_GetPtr(mb), RK_MPI_MB_GetSize(mb));
        RK_MPI_MB_ReleaseBuffer(mb);
        buffer.first = true;
    }

    static void rk_jpeg_sync_cb(MEDIA_BUFFER mb, void *data) {
        auto &buffer = *(std::pair<bool, std::vector<uint8_t> &> *)data;
        buffer.second.resize(RK_MPI_MB_GetSize(mb));
        memcpy(buffer.second.data(), RK_MPI_MB_GetPtr(mb), RK_MPI_MB_GetSize(mb));
        RK_MPI_MB_ReleaseBuffer(mb);
        buffer.first = true;
    }

private:
    size_t width_;
    size_t height_;

    MPP_CHN_S enc_chn_;
    VENC_JPEG_PARAM_S jpeg_param_;
    VENC_CHN_ATTR_S venc_chn_attr_;
    MB_IMAGE_INFO_S image_info_;
    MEDIA_BUFFER mb_;

    static int channal_id_;
};

int RkMediaJpegCodecPrivate::channal_id_{0};

RkMediaJpegCodec::RkMediaJpegCodec() : impl_(std::make_unique<RkMediaJpegCodecPrivate>()) {}

RkMediaJpegCodec::~RkMediaJpegCodec() {}

bool RkMediaJpegCodec::init_codecer(const size_t width, const size_t height, const int quality) {
    return impl_->init_encoder(width, height, quality);
}

bool RkMediaJpegCodec::codec_image(const u_char *input, const JpegCodecCallback &callback) {
    return impl_->encode(input, callback);
}

bool RkMediaJpegCodec::codec_image(const uint8_t *input, std::vector<uint8_t> &jepg_data) {
    return impl_->encode(input, jepg_data);
}

const bool RkMediaJpegCodec::save_image(const u_char *input, const char *path) {
    return impl_->encode(input, [&path](const uint8_t *data, const size_t dlen) {
        FILE *file = fopen(path, "wb");
        fwrite(data, sizeof(char), dlen, file);
        fclose(file);
    });
}

}// namespace codec
}// namespace gddi