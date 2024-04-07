#ifndef __MLU220_WRAPPER_HPP__
#define __MLU220_WRAPPER_HPP__

#if defined(WITH_MLU220) || defined(WITH_MLU270)

#include <cn_codec_common.h>
#include <cn_codec_define.h>
#include <cstdint>
#include <libavutil/macros.h>
#include <memory>
#include <stdexcept>
#include <sys/types.h>

#ifdef WITH_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

#include "../mem_pool.hpp"
#include "types.hpp"

#include "cn_jpeg_dec.h"
#include "cn_jpeg_enc.h"
#include "cncv.h"
#include "cnis/contrib/video_helper.h"
#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include "cnrt.h"
#include <cn_codec_common.h>
#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

namespace gddi {

using ImagePool = gddi::MemPool<cncodecFrame, cncodecPixelFormat, int, int>;

namespace image_wrapper {

static void get_stride(const cncodecPixelFormat format, const size_t width, u32_t *stride) {
    if (format == CNCODEC_PIX_FMT_NV12) {
        stride[0] = stride[1] = width;
    } else if (format == CNCODEC_PIX_FMT_I420) {
        stride[0] = width;
        stride[1] = stride[2] = width / 2;
    }
}

static size_t get_plane_num(const cncodecPixelFormat format) {
    size_t plane_num = 1;
    if (format == CNCODEC_PIX_FMT_NV12 || format == CNCODEC_PIX_FMT_NV21) {
        plane_num = 2;
    } else if (format == CNCODEC_PIX_FMT_I420) {
        plane_num = 3;
    }
    return plane_num;
}

static std::shared_ptr<cncvHandle_t> get_handle(const int dev_id = 0) {
    cnrtInit(dev_id);
    return std::shared_ptr<cncvHandle_t>(new cncvHandle_t, [](cncvHandle_t *ptr) { cncvDestroy(*ptr); });
}
const auto cn_handle = get_handle(0);

static std::shared_ptr<cncodecFrame> alloc_cncodec_frame(const cncodecPixelFormat format, const int width,
                                                         const int height) {
    auto frame = std::shared_ptr<cncodecFrame>(new cncodecFrame, [](cncodecFrame *ptr) {
        for (int i = 0; i < ptr->planeNum; i++) { cnrtFree((void *)ptr->plane[i].addr); }
    });
    frame->pixelFmt = format;
    frame->colorSpace = CNCODEC_COLOR_SPACE_BT_709;
    frame->height = height;
    frame->width = width;
    frame->deviceId = 0;

    int stride_width = FFALIGN(frame->width, 128);
    int stride_height = FFALIGN(frame->height, 16);
    if (format == CNCODEC_PIX_FMT_NV12 || format == CNCODEC_PIX_FMT_NV21) {
        frame->channel = 2;
        frame->planeNum = 2;
        frame->stride[0] = stride_width;
        frame->stride[1] = stride_width;
        frame->plane[0].size = stride_width * stride_height;
        frame->plane[1].size = stride_width * stride_height / 2;
        cnrtMalloc((void **)&frame->plane[0].addr, frame->plane[0].size);
        cnrtMalloc((void **)&frame->plane[1].addr, frame->plane[1].size);
    } else if (format == CNCODEC_PIX_FMT_I420) {
        frame->channel = 3;
        frame->planeNum = 3;
        frame->stride[0] = stride_width;
        frame->stride[1] = stride_width / 2;
        frame->stride[2] = stride_width / 2;
        frame->plane[0].size = stride_width * stride_height;
        frame->plane[1].size = stride_width * stride_height / 4;
        frame->plane[2].size = stride_width * stride_height / 4;
        cnrtMalloc((void **)&frame->plane[0].addr, frame->plane[0].size);
        cnrtMalloc((void **)&frame->plane[1].addr, frame->plane[1].size);
        cnrtMalloc((void **)&frame->plane[2].addr, frame->plane[2].size);
    } else {
        frame->channel = 1;
        frame->planeNum = 1;
        frame->stride[0] = stride_width * 4;
        frame->plane[0].size = stride_width * stride_height * 4;
        cnrtMalloc((void **)&frame->plane[0].addr, frame->plane[0].size);
    }

    return frame;
}

static void image_jpeg_enc(const std::shared_ptr<cncodecFrame> &image, std::vector<uchar> &vec_jpeg_data,
                           int &jpeg_data_size, const uint32_t quality = 85) {
    auto src_frame = image;

    int stride_width = FFALIGN(image->width, 128);
    int stride_height = FFALIGN(image->height, 16);
    if (stride_width != image->stride[0]) {
        src_frame = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, image->width, image->height);
        for (int i = 0; i < image->height; i++) {
            cnrtMemcpy((void *)(src_frame->plane[0].addr + stride_width * i),
                       (void *)(image->plane[0].addr + image->width * i), image->width, CNRT_MEM_TRANS_DIR_DEV2DEV);
        }

        for (int i = 0; i < image->height / 2; i++) {
            cnrtMemcpy((void *)(src_frame->plane[1].addr + stride_width * i),
                       (void *)(image->plane[1].addr + image->width * i), image->width, CNRT_MEM_TRANS_DIR_DEV2DEV);
        }
    } else if (stride_height != image->height) {
        src_frame = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, image->width, image->height);
        cnrtMemcpy((void *)src_frame->plane[0].addr, (void *)image->plane[0].addr, image->plane[0].size,
                   CNRT_MEM_TRANS_DIR_DEV2DEV);
        cnrtMemcpy((void *)src_frame->plane[1].addr, (void *)image->plane[1].addr, image->plane[1].size,
                   CNRT_MEM_TRANS_DIR_DEV2DEV);
    }

    // 1. 创建JPEG编码器
    auto handle = std::unique_ptr<cnjpegEncoder, void (*)(cnjpegEncoder *)>(
        new cnjpegEncoder, [](cnjpegEncoder *ptr) { cnjpegEncDestroy(*ptr); });

    cnjpegEncCreateInfo enc_info;
    memset(&enc_info, 0, sizeof(cnjpegEncCreateInfo));
    enc_info.deviceId = 0;
    enc_info.instance = 6;
    enc_info.pixelFmt = src_frame->pixelFmt;
    enc_info.width = src_frame->width;
    enc_info.height = src_frame->height;
    enc_info.colorSpace = src_frame->colorSpace;
    enc_info.inputBufNum = 0;//同步模式不生效
    enc_info.outputBufNum = 0;
    enc_info.allocType = CNCODEC_BUF_ALLOC_LIB;
    if (cnjpegEncCreate(handle.get(), CNJPEGENC_RUN_MODE_SYNC, NULL, &enc_info) != CNCODEC_SUCCESS) {
        throw std::runtime_error("cnjpegEncCreate failed!");
    }

    // 2. 创建输入输出上下文
    cnjpegEncInput enc_input{*src_frame, CNJPEGENC_FLAG_TIMESTAMP};
    auto enc_output =
        std::unique_ptr<cnjpegEncOutput, void (*)(cnjpegEncOutput *)>(new cnjpegEncOutput, [](cnjpegEncOutput *ptr) {
            if (ptr->streamBuffer.addr) { cnrtFree((void *)ptr->streamBuffer.addr); }
        });

    // 3. 分配 Buffer 空间
    cnjpegEncGetSuggestBitStreamBufSize(enc_input.frame.width, enc_input.frame.height, quality,
                                        &enc_output->streamBuffer.size);
    enc_info.suggestedLibAllocBitStrmBufSize = enc_output->streamBuffer.size;
    if (cnrtMallocFrameBuffer((void **)(&enc_output->streamBuffer.addr), enc_output->streamBuffer.size)
        != CNRT_RET_SUCCESS) {
        throw std::runtime_error("cnrtMallocFrameBuffer failed!");
    }

    // 4. 编码
    cnjpegEncParameters enc_frame_param{quality, 0, 0};
    if (cnjpegEncSyncEncode(*handle, enc_output.get(), &enc_input, &enc_frame_param, 4000) != CNCODEC_SUCCESS) {
        std::runtime_error("cnjpegEncSyncEncode failed!");
    }

    // 5. 内存拷贝 device -> host
    jpeg_data_size = enc_output->streamLength;
    if (cnrtMemcpy((void *)vec_jpeg_data.data(), (void *)(enc_output->streamBuffer.addr + enc_output->dataOffset),
                   enc_output->streamLength, CNRT_MEM_TRANS_DIR_DEV2HOST)
        != CNRT_RET_SUCCESS) {
        throw std::runtime_error("cnrtMemcpy failed!");
    }
}

static auto image_resize(const std::shared_ptr<cncodecFrame> &image, int width, int height) {
    // 无需 Resize
    if (image->width == width && image->height == height) { return image; }

    auto src_frame = image;
    auto dst_frame = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, width, height);

    int stride_width = FFALIGN(image->width, 128);
    if (stride_width != image->stride[0]) {
        src_frame = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, image->width, image->height);
        for (int i = 0; i < image->height; i++) {
            cnrtMemcpy((void *)(src_frame->plane[0].addr + stride_width * i),
                       (void *)(image->plane[0].addr + image->width * i), image->width, CNRT_MEM_TRANS_DIR_DEV2DEV);
        }

        for (int i = 0; i < image->height / 2; i++) {
            cnrtMemcpy((void *)(src_frame->plane[1].addr + stride_width * i),
                       (void *)(image->plane[1].addr + image->width * i), image->width, CNRT_MEM_TRANS_DIR_DEV2DEV);
        }
    }

    cncodecWorkInfo work_info{.inMsg = {0, 0, 0, 0, 0}, .outMsg = {0.0, 0, 0}};
    if (cncodecImageTransform(dst_frame.get(), nullptr, src_frame.get(), nullptr, cncodecFilter::CNCODEC_NumFilters,
                              &work_info)
        != CNCODEC_SUCCESS) {
        throw std::runtime_error("cncodecImageTransform failed!");
    }

    return dst_frame;
}

static std::shared_ptr<cncodecFrame> image_png_dec(const unsigned char *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    cv::Mat mat_image = cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR);
    auto cn_frame = alloc_cncodec_frame(CNCODEC_PIX_FMT_BGRA, mat_image.cols, mat_image.rows);
    for (int i = 0; i < mat_image.cols * mat_image.rows; i++) {
        cnrtMemcpy(mat_image.data + i * 4, mat_image.data + i * 3, 3, CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    return cn_frame;
}

static void image_save_as_jpeg(const std::shared_ptr<cncodecFrame> &src_frame, const std::string &path,
                               const int quality = 85) {
    int jpeg_data_size = 0;
    auto vec_jpeg_data = std::vector<uchar>(src_frame->width * src_frame->height);
    image_jpeg_enc(src_frame, vec_jpeg_data, jpeg_data_size, quality);
    std::ofstream file(path);
    file.write((char *)vec_jpeg_data.data(), jpeg_data_size);
    file.close();
}

static void image_save_as_jpeg(const cv::Mat &image, const std::string &path, const int quality = 85) {
    cv::imwrite(path, image, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, quality});
}

static cv::Mat image_to_mat(const std::shared_ptr<cncodecFrame> &image) {
    cv::Mat mat_image;
    if (image->pixelFmt == CNCODEC_PIX_FMT_NV12) {
        mat_image = cv::Mat(image->height * 3 / 2, image->width, CV_8UC1);
        cnrtMemcpy(mat_image.data, (void *)image->plane[0].addr, image->stride[0] * image->height,
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
        cnrtMemcpy(mat_image.data + image->stride[0] * image->height, (void *)image->plane[1].addr,
                   image->stride[1] * image->height / 2, CNRT_MEM_TRANS_DIR_DEV2HOST);
        cv::cvtColor(mat_image, mat_image, cv::COLOR_YUV2BGRA_NV12);
    } else if (image->pixelFmt == CNCODEC_PIX_FMT_BGRA) {
        mat_image = cv::Mat(image->height, image->width, CV_8UC4);
        cnrtMemcpy(mat_image.data, (void *)image->plane[0].addr, image->width * image->height * 4,
                   CNRT_MEM_TRANS_DIR_DEV2HOST);
    }
    return mat_image;
}

static std::shared_ptr<cncodecFrame> mat_to_image(const cv::Mat &image) {
    std::shared_ptr<cncodecFrame> cn_image;
    if (image.channels() == 4) {
        cn_image = alloc_cncodec_frame(CNCODEC_PIX_FMT_BGRA, image.cols, image.rows);
        cnrtMemcpy((void *)cn_image->plane[0].addr, image.data, image.cols * image.rows * 4,
                   CNRT_MEM_TRANS_DIR_HOST2DEV);
    }
    return cn_image;
}

static std::shared_ptr<cncodecFrame> image_threshold(const std::shared_ptr<cncodecFrame> &src_image) {
    auto mask_image =
        gddi::image_wrapper::alloc_cncodec_frame(CNCODEC_PIX_FMT_BGRA, src_image->width, src_image->height);
    cncvImageDescriptor src_desc{src_image->width,
                                 src_image->height,
                                 {src_image->width * 4},
                                 CNCV_PIX_FMT_BGRA,
                                 CNCV_COLOR_SPACE_BT_709};
    cncvImageDescriptor mask_desc{src_image->width,
                                  src_image->height,
                                  {src_image->width * 4},
                                  CNCV_PIX_FMT_BGRA,
                                  CNCV_COLOR_SPACE_BT_709};
    void *src_mlu_ptr;
    void *mask_mlu_ptr;
    cnrtMalloc(&src_mlu_ptr, sizeof(void *));
    cnrtMalloc(&mask_mlu_ptr, sizeof(void *));
    cnrtMemcpy(src_mlu_ptr, &src_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(mask_mlu_ptr, &mask_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    if (cncvThreshold_Gray(*cn_handle, 1, 255, CNCV_THRESH_BINARY, src_desc, src_mlu_ptr, mask_desc, mask_mlu_ptr)
        != CNCV_STATUS_SUCCESS) {
        throw std::runtime_error("cncvThreshold_Gray failed!");
    }

    return mask_image;
}

static void copy_to_image(const std::shared_ptr<cncodecFrame> &src_image,
                          const std::shared_ptr<cncodecFrame> &mask_image,
                          const std::shared_ptr<cncodecFrame> &out_image) {
    cncvImageDescriptor desc{src_image->width,
                             src_image->height,
                             {src_image->width * 4},
                             CNCV_PIX_FMT_BGRA,
                             CNCV_COLOR_SPACE_BT_709};
    cncvRect src_roi{0, 0, src_image->width, src_image->height};
    cncvRect dst_roi{0, 0, out_image->width, out_image->height};

    void *src_mlu_ptr;
    void *mask_mlu_ptr;
    void *output_mlu_ptr;
    cnrtMalloc(&src_mlu_ptr, sizeof(void *));
    cnrtMalloc(&mask_mlu_ptr, sizeof(void *));
    cnrtMalloc(&output_mlu_ptr, sizeof(void *));
    cnrtMemcpy(src_mlu_ptr, &src_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(mask_mlu_ptr, &mask_image->plane[1].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(output_mlu_ptr, &out_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);

    if (cncvCopyTo(*cn_handle, 1, desc, src_roi, &src_mlu_ptr, desc, &mask_mlu_ptr, desc, dst_roi, &output_mlu_ptr)
        != CNCV_STATUS_SUCCESS) {
        throw std::runtime_error("cncvCopyTo failed!");
    }
}

static std::shared_ptr<cncodecFrame> image_jpeg_dec(const unsigned char *jpeg_data, size_t dsize) {
    // 1. 创建JPEG解码器
    auto handle = std::unique_ptr<cnjpegDecoder, void (*)(cnjpegDecoder *)>(
        new cnjpegDecoder, [](cnjpegDecoder *ptr) { cnjpegDecDestroy(*ptr); });

    cnjpegDecCreateInfo dec_info;
    dec_info.deviceId = 0;
    dec_info.instance = CNJPEGDEC_INSTANCE_0;
    dec_info.pixelFmt = CNCODEC_PIX_FMT_NV12;
    dec_info.colorSpace = CNCODEC_COLOR_SPACE_BT_601;
    dec_info.width = 16;
    dec_info.height = 16;
    dec_info.inputBufNum = 0;
    dec_info.outputBufNum = 0;
    dec_info.bitDepthMinus8 = 0;
    dec_info.allocType = CNCODEC_BUF_ALLOC_LIB;
    dec_info.suggestedLibAllocBitStrmBufSize = (4U << 20);
    dec_info.enablePreparse = 0;

    if (cnjpegDecCreate(handle.get(), CNJPEGDEC_RUN_MODE_SYNC, nullptr, &dec_info) != CNCODEC_SUCCESS) {
        throw std::runtime_error("cnjpegDecCreate failed!");
    }

    cnjpegDecImageInfo image_info;
    if (cnjpegDecGetImageInfo(*handle, &image_info, (void *)jpeg_data, dsize) != CNCODEC_SUCCESS) {
        throw std::runtime_error("cnjpegDecGetImageInfo failed!");
    }

    // 2. 创建输入输出上下文
    cnjpegDecInput dec_input;
    dec_input.streamLength = dsize;
    dec_input.streamBuffer = (uint8_t *)jpeg_data;
    cnjpegDecOutput dec_output;
    auto cn_frame = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, image_info.width, image_info.height);
    dec_output.frame = *cn_frame;

    // 3. 解码
    if (cnjpegDecSyncDecode(*handle, &dec_output, &dec_input, 4000) != CNCODEC_SUCCESS) {
        throw std::runtime_error("cnjpegDecSyncDecode failed!");
    }

    if (FFALIGN(cn_frame->width, 128) != cn_frame->width || FFALIGN(cn_frame->height, 2) != cn_frame->height) {
        cn_frame = image_resize(cn_frame, FFALIGN(cn_frame->width, 128), FFALIGN(cn_frame->height, 2));
    }

    return cn_frame;
}

static auto image_raw2yuv(unsigned char *raw_image, size_t dsize, int width, int height) {
    auto dst_image = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, width, height);
    cnrtMemcpy((void *)dst_image->plane[0].addr, raw_image, width * height, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy((void *)dst_image->plane[1].addr, raw_image + width * height, width * height / 2,
               CNRT_MEM_TRANS_DIR_HOST2DEV);
    return dst_image;
}

static auto yuv_to_bgra_1(const std::shared_ptr<cncodecFrame> &src_image) {
    auto dst_image = alloc_cncodec_frame(CNCODEC_PIX_FMT_BGRA, src_image->width, src_image->height);
    cncvImageDescriptor src_desc{src_image->width,
                                 src_image->height,
                                 {src_image->width, src_image->width},
                                 CNCV_PIX_FMT_NV12,
                                 CNCV_COLOR_SPACE_BT_709};
    cncvImageDescriptor dst_desc{src_image->width,
                                 src_image->height,
                                 {src_image->width * 4},
                                 CNCV_PIX_FMT_BGRA,
                                 CNCV_COLOR_SPACE_BT_709};
    cncvRect src_roi{0, 0, src_image->width, src_image->height};
    void *buffer[2];
    void *output_mlu_ptr;
    cnrtMalloc(&buffer[0], sizeof(void *));
    cnrtMalloc(&buffer[1], sizeof(void *));
    cnrtMalloc(&output_mlu_ptr, sizeof(void *));
    cnrtMemcpy(buffer[0], &src_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(buffer[1], &src_image->plane[1].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(output_mlu_ptr, &dst_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);

    size_t workspace_size{0};
    void *workspace{nullptr};
    cncvGetYuvToRgbxWorkspaceSize(CNCV_PIX_FMT_NV12, CNCV_PIX_FMT_BGRA, &workspace_size);
    cnrtMalloc(&workspace, workspace_size);
    if (cncvYuvToRgbx(*cn_handle, 1, src_desc, buffer, dst_desc, &output_mlu_ptr, workspace_size, workspace)
        != CNCV_STATUS_SUCCESS) {
        throw std::runtime_error("cncvYuvToRgbx failed!");
    }
    cnrtFree(buffer[0]);
    cnrtFree(buffer[1]);
    cnrtFree(output_mlu_ptr);
    cnrtFree(workspace);

    return dst_image;
}

static auto yuv_to_bgra(const std::shared_ptr<cncodecFrame> &src_image) {
    auto dst_image = alloc_cncodec_frame(CNCODEC_PIX_FMT_BGRA, src_image->width, src_image->height);
    cncvImageDescriptor src_desc{src_image->width,
                                 src_image->height,
                                 {src_image->width, src_image->width},
                                 CNCV_PIX_FMT_NV12,
                                 CNCV_COLOR_SPACE_BT_709};
    cncvImageDescriptor dst_desc{src_image->width,
                                 src_image->height,
                                 {src_image->width * 4},
                                 CNCV_PIX_FMT_BGRA,
                                 CNCV_COLOR_SPACE_BT_709};
    cncvRect src_roi{0, 0, src_image->width, src_image->height};
    cncvRect dst_roi{0, 0, src_image->width, src_image->height};
    void *y_mlu_ptr;
    void *uv_mlu_ptr;
    void *output_mlu_ptr;
    cnrtMalloc(&y_mlu_ptr, sizeof(void *));
    cnrtMalloc(&uv_mlu_ptr, sizeof(void *));
    cnrtMalloc(&output_mlu_ptr, sizeof(void *));
    cnrtMemcpy(y_mlu_ptr, &src_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(uv_mlu_ptr, &src_image->plane[1].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(output_mlu_ptr, &dst_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    size_t workspace_size{0};
    void *workspace{nullptr};
    cncvGetResizeConvertWorkspaceSize(1, &src_desc, &src_roi, &dst_desc, &dst_roi, &workspace_size);
    cnrtMalloc(&workspace, workspace_size);
    if (cncvResizeConvert(*cn_handle, 1, &src_desc, &src_roi, (void **)y_mlu_ptr, (void **)uv_mlu_ptr, &dst_desc,
                          &dst_roi, (void **)output_mlu_ptr, workspace_size, workspace,
                          cncvInterpolation::CNCV_INTER_BILINEAR)
        != CNCV_STATUS_SUCCESS) {
        throw std::runtime_error("cncvResizeConvert failed!");
    }
    cnrtFree(y_mlu_ptr);
    cnrtFree(uv_mlu_ptr);
    cnrtFree(output_mlu_ptr);
    cnrtFree(workspace);

    return dst_image;
}

static auto bgr_to_yuv(const std::shared_ptr<cncodecFrame> &src_image) {
    auto dst_image = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, src_image->width, src_image->height);
    cncvImageDescriptor src_desc{src_image->width,
                                 src_image->height,
                                 {src_image->width * 4},
                                 CNCV_PIX_FMT_BGRA,
                                 CNCV_COLOR_SPACE_BT_709};
    cncvImageDescriptor dst_desc{src_image->width,
                                 src_image->height,
                                 {src_image->width, src_image->width},
                                 CNCV_PIX_FMT_NV12,
                                 CNCV_COLOR_SPACE_BT_709};
    cncvRect src_roi{0, 0, src_image->width, src_image->height};
    void *src_mlu_ptr;
    void *buffer[2];
    cnrtMalloc(&src_mlu_ptr, sizeof(void *));
    cnrtMalloc(&buffer[0], sizeof(void *));
    cnrtMalloc(&buffer[1], sizeof(void *));
    cnrtMemcpy(src_mlu_ptr, &src_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(buffer[0], &dst_image->plane[0].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpy(buffer[1], &dst_image->plane[1].addr, sizeof(void *), CNRT_MEM_TRANS_DIR_HOST2DEV);
    if (cncvRgbxToYuv(*cn_handle, src_desc, src_roi, src_mlu_ptr, dst_desc, buffer) != CNCV_STATUS_SUCCESS) {
        throw std::runtime_error("cncvRgbxToYuv failed");
    }

    return dst_image;
}

static auto image_crop(const std::shared_ptr<cncodecFrame> &src_frame, const std::map<int, Rect2f> &crop_rects) {
    auto crop_num = crop_rects.size();
    std::map<int, std::shared_ptr<cncodecFrame>> map_crop_images;

    for (const auto &[idx, rect] : crop_rects) {
        map_crop_images[idx] = alloc_cncodec_frame(CNCODEC_PIX_FMT_NV12, rect.width, rect.height);

        cncodecRectangle rectangle{(uint32_t)rect.x, (uint32_t)rect.y, (uint32_t)(rect.x + rect.width),
                                   (uint32_t)(rect.y + rect.height)};
        cncodecWorkInfo work_info{.inMsg = {0, 0, 0, 0, 0}, .outMsg = {0.0, 0, 0}};
        if (cncodecImageTransform(map_crop_images[idx].get(), nullptr, src_frame.get(), &rectangle,
                                  cncodecFilter::CNCODEC_NumFilters, &work_info)
            != CNCODEC_SUCCESS) {
            throw std::runtime_error("crop cncodecImageTransform failed");
        }
    }

    return map_crop_images;
}

static std::shared_ptr<MemObject<cncodecFrame, std::shared_ptr<AVFrame>>>
image_from_avframe(ImagePool &mem_pool, const std::shared_ptr<AVFrame> &frame) {
    auto mem_obj =
        mem_pool.alloc_mem_detach<std::shared_ptr<AVFrame>>(frame, CNCODEC_PIX_FMT_NV12, frame->width, frame->height);
    mem_obj->data->plane[0].addr = (uint64_t)frame->data[0];
    mem_obj->data->plane[0].size = frame->linesize[0] * frame->height;
    mem_obj->data->plane[1].addr = (uint64_t)frame->data[1];
    mem_obj->data->plane[1].size = frame->linesize[1] * frame->height / 2;
    return mem_obj;
}

}// namespace image_wrapper
}// namespace gddi

#endif

#endif