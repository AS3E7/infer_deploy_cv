#include "cn_jpeg_codec.h"
#include "mlu220_wrapper.hpp"
#include "spdlog/spdlog.h"

#include <cstdint>
#include <memory.h>
#include <memory>

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))

namespace gddi {
namespace codec {

class CNJpegCodecPrivate {
public:
    CNJpegCodecPrivate() {}
    ~CNJpegCodecPrivate() {}

    bool init_encoder(const size_t width, const size_t height, const PixelFormat format, const size_t quality) {
        width_ = width;
        height_ = height;
        quality_ = quality;

        // 1. 创建JPEG编码器
        encoder_ = std::shared_ptr<cnjpegEncoder>(new cnjpegEncoder, [](cnjpegEncoder ptr) { cnjpegEncDestroy(ptr); });

        cnjpegEncCreateInfo enc_info;
        memset(&enc_info, 0, sizeof(cnjpegEncCreateInfo));
        enc_info.deviceId = 0;
        enc_info.instance = 6;
        switch (format) {
            case PixelFormat::kYUV420P: enc_info.pixelFmt = CNCODEC_PIX_FMT_I420; break;
            case PixelFormat::KNV12: enc_info.pixelFmt = CNCODEC_PIX_FMT_NV12; break;
            case PixelFormat::kBGR888: enc_info.pixelFmt = CNCODEC_PIX_FMT_BGRA; break;
        }
        enc_info.width = width;
        enc_info.height = height;
        enc_info.colorSpace = CNCODEC_COLOR_SPACE_BT_709;
        enc_info.inputBufNum = 0;//同步模式不生效
        enc_info.outputBufNum = 0;
        enc_info.allocType = CNCODEC_BUF_ALLOC_LIB;

        if (cnjpegEncCreate(encoder_.get(), CNJPEGENC_RUN_MODE_SYNC, NULL, &enc_info) != CNCODEC_SUCCESS) {
            spdlog::error("cnjpegEncCreate failed!");
            return false;
        }

        return true;
    }

    bool encode(const std::shared_ptr<cncodecFrame> &input, const JpegCodecCallback &callback) {
        int stride_width = ALIGN(width_, 128);
        int stride_height = ALIGN(height_, 16);
        auto enc_image = input;

        if (stride_width != input->stride[0]) {
            enc_image = image_wrapper::alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, input->width, input->height);
            for (int i = 0; i < input->height; i++) {
                cnrtMemcpy((void *)(enc_image->plane[0].addr + stride_width * i),
                           (void *)(input->plane[0].addr + input->width * i), input->width, CNRT_MEM_TRANS_DIR_DEV2DEV);
            }

            for (int i = 0; i < input->height / 2; i++) {
                cnrtMemcpy((void *)(enc_image->plane[1].addr + stride_width * i),
                           (void *)(input->plane[1].addr + input->width * i), input->width, CNRT_MEM_TRANS_DIR_DEV2DEV);
            }
        } else if (stride_height != input->height) {
            enc_image = image_wrapper::alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, input->width, input->height);
            cnrtMemcpy((void *)enc_image->plane[0].addr, (void *)input->plane[0].addr, input->plane[0].size,
                       CNRT_MEM_TRANS_DIR_DEV2DEV);
            cnrtMemcpy((void *)enc_image->plane[1].addr, (void *)input->plane[1].addr, input->plane[1].size,
                       CNRT_MEM_TRANS_DIR_DEV2DEV);
        }

        // 2. 创建输入输出上下文
        cnjpegEncInput enc_input{*enc_image, CNJPEGENC_FLAG_TIMESTAMP};
        auto enc_output = std::unique_ptr<cnjpegEncOutput, void (*)(cnjpegEncOutput *)>(
            new cnjpegEncOutput, [](cnjpegEncOutput *ptr) {
                if (ptr->streamBuffer.addr) { cnrtFree((void *)ptr->streamBuffer.addr); }
            });

        // 3. 分配 Buffer 空间
        cnjpegEncGetSuggestBitStreamBufSize(enc_input.frame.width, enc_input.frame.height, quality_,
                                            &enc_output->streamBuffer.size);
        // enc_info.suggestedLibAllocBitStrmBufSize = enc_output->streamBuffer.size;
        if (cnrtMallocFrameBuffer((void **)(&enc_output->streamBuffer.addr), enc_output->streamBuffer.size)
            != CNRT_RET_SUCCESS) {
            throw std::runtime_error("cnrtMallocFrameBuffer failed!");
        }

        // 4. 编码
        cnjpegEncParameters enc_frame_param{quality_, 0, 0};
        if (cnjpegEncSyncEncode(*encoder_, enc_output.get(), &enc_input, &enc_frame_param, 4000) != CNCODEC_SUCCESS) {
            std::runtime_error("cnjpegEncSyncEncode failed!");
        }

        // 5. 内存拷贝 device -> host
        auto jpeg_data = new uint8_t[enc_output->streamLength + 1];
        if (cnrtMemcpy((void *)jpeg_data, (void *)(enc_output->streamBuffer.addr + enc_output->dataOffset),
                       enc_output->streamLength, CNRT_MEM_TRANS_DIR_DEV2HOST)
            != CNRT_RET_SUCCESS) {
            throw std::runtime_error("cnrtMemcpy failed!");
        }

        callback(jpeg_data, enc_output->streamLength);
        delete[] jpeg_data;

        return true;
    }

private:
    uint32_t width_;
    uint32_t height_;
    uint32_t quality_;

    std::shared_ptr<cnjpegEncoder> encoder_;
};

CNJpegCodec::CNJpegCodec() : impl_(std::make_unique<CNJpegCodecPrivate>()) {}

CNJpegCodec::~CNJpegCodec() {}

bool CNJpegCodec::init_codecer(const JpegCodecParam &option) {
    if (option.type == CodecType::kEncoder) {
        return impl_->init_encoder(option.width, option.height, option.format, option.quality);
    }
    return false;
}

bool CNJpegCodec::codec_image(const std::shared_ptr<cncodecFrame> &input, const JpegCodecCallback &callback) {
    return impl_->encode(input, callback);
}

bool CNJpegCodec::save_image(const std::shared_ptr<cncodecFrame> &input, const char *path) {
    return impl_->encode(input, [&path](const uint8_t *data, const size_t dlen) {
        FILE *file = fopen(path, "wb");
        fwrite(data, sizeof(char), dlen, file);
        fclose(file);
    });
}

}// namespace codec
}// namespace gddi