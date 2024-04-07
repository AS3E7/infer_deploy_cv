#ifndef __MLU370_WRAPPER_HPP__
#define __MLU370_WRAPPER_HPP__

#if defined(WITH_MLU370)

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
#include "cnis/infer_server.h"
#include "cnis/processor.h"
#include "cnrt.h"
#include <cn_codec_common.h>
#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <cnedk_buf_surface.h>

namespace gddi {

using ImagePool = gddi::MemPool<CnedkBufSurface>;

namespace image_wrapper {

static void get_stride(const CnedkBufSurfaceColorFormat format, const size_t width, uint32_t *stride) {
    if (format == CNEDK_BUF_COLOR_FORMAT_NV12 || format == CNEDK_BUF_COLOR_FORMAT_NV21) {
        stride[0] = stride[1] = width;
    } else if (format == CNEDK_BUF_COLOR_FORMAT_YUV420) {
        stride[0] = width;
        stride[1] = stride[2] = width / 2;
    }
}

static size_t get_plane_num(const CnedkBufSurfaceColorFormat format) {
    size_t plane_num = 1;
    if (format == CNEDK_BUF_COLOR_FORMAT_NV12 || format == CNEDK_BUF_COLOR_FORMAT_NV21) {
        plane_num = 2;
    } else if (format == CNEDK_BUF_COLOR_FORMAT_YUV420) {
        plane_num = 3;
    }
    return plane_num;
}

static std::shared_ptr<CnedkBufSurface> alloc_cnedk_frame(const CnedkBufSurfaceColorFormat format, const int width,
                                                          const int height) {
    auto frame = std::shared_ptr<CnedkBufSurface>(new CnedkBufSurface, [](CnedkBufSurface *ptr) {
        if (ptr->surface_list) {
            cnrtFree((void *)ptr->surface_list->data_ptr);
            delete ptr->surface_list;
        }
    });
    frame->surface_list = new CnedkBufSurfaceParams();
    frame->surface_list->color_format = format;
    frame->surface_list->height = height;
    frame->surface_list->width = width;
    frame->device_id = 0;
    frame->batch_size = 1;
    frame->num_filled = 1;
    frame->device_id = 0;
    frame->mem_type = CNEDK_BUF_MEM_DEVICE;

    uint32_t stride_width = width;
    uint32_t stride_height = height;
    if (format == CNEDK_BUF_COLOR_FORMAT_NV12 || format == CNEDK_BUF_COLOR_FORMAT_NV21) {
        frame->surface_list->plane_params.num_planes = 2;
        frame->surface_list->plane_params.width[0] = stride_width;
        frame->surface_list->plane_params.height[0] = stride_width;
        frame->surface_list->plane_params.pitch[0] = stride_width;
        frame->surface_list->plane_params.pitch[1] = stride_width;
        frame->surface_list->plane_params.offset[0] = 0;
        frame->surface_list->plane_params.offset[1] = stride_width * stride_height;
        frame->surface_list->data_size = stride_width * stride_height * 3 / 2;
    } else if (format == CNEDK_BUF_COLOR_FORMAT_YUV420) {
        frame->surface_list->plane_params.num_planes = 3;
        frame->surface_list->plane_params.width[0] = stride_width;
        frame->surface_list->plane_params.height[0] = stride_width;
        frame->surface_list->plane_params.pitch[0] = stride_width;
        frame->surface_list->plane_params.pitch[1] = stride_width / 2;
        frame->surface_list->plane_params.pitch[2] = stride_width / 2;
        frame->surface_list->plane_params.offset[0] = 0;
        frame->surface_list->plane_params.offset[1] = stride_width * stride_height;
        frame->surface_list->plane_params.offset[1] = stride_width * stride_height * 5 / 4;
        frame->surface_list->data_size = stride_width * stride_height * 3 / 2;
    } else {
        frame->surface_list->plane_params.num_planes = 1;
        frame->surface_list->plane_params.width[0] = stride_width;
        frame->surface_list->plane_params.height[0] = stride_width;
        frame->surface_list->plane_params.pitch[0] = stride_width * 3;
        frame->surface_list->plane_params.offset[0] = 0;
        frame->surface_list->data_size = stride_width * stride_height * 3;
    }
    cnrtMalloc(&frame->surface_list->data_ptr, frame->surface_list->data_size);

    return frame;
}

static void image_jpeg_enc(const std::shared_ptr<CnedkBufSurface> &image, std::vector<uchar> &vec_jpeg_data,
                           const uint32_t quality = 85) {}

static auto image_resize(const std::shared_ptr<CnedkBufSurface> &image, uint32_t width, uint32_t height) {
    // 无需 Resize
    if (image->surface_list->width == width && image->surface_list->height == height) { return image; }

    auto dst_frame = alloc_cnedk_frame(CNEDK_BUF_COLOR_FORMAT_NV12, width, height);

    // cncodecWorkInfo work_info{.inMsg = {0, 0, 0, 0, 0}, .outMsg = {0.0, 0, 0}};
    // if (cncodecImageTransform(dst_frame.get(), nullptr, image.get(), nullptr, cncodecFilter::CNCODEC_NumFilters,
    //                           &work_info)
    //     != CNCODEC_SUCCESS) {
    //     throw std::runtime_error("cncodecImageTransform failed!");
    // }

    return dst_frame;
}

static std::shared_ptr<CnedkBufSurface> image_png_dec(const unsigned char *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    cv::Mat mat_image = cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR);
    auto cndek_frame = alloc_cnedk_frame(CNEDK_BUF_COLOR_FORMAT_BGR, mat_image.cols, mat_image.rows);
    for (int i = 0; i < mat_image.cols * mat_image.rows; i++) {
        cnrtMemcpy(mat_image.data + i * 4, mat_image.data + i * 3, 3, CNRT_MEM_TRANS_DIR_HOST2DEV);
    }

    return cndek_frame;
}

static void image_save_as_jpeg(const std::shared_ptr<CnedkBufSurface> &src_frame, const std::string &path,
                               const int quality = 85) {
    std::vector<uchar> vec_jpeg_data;
    image_jpeg_enc(src_frame, vec_jpeg_data, quality);
    std::ofstream file(path);
    file.write((char *)vec_jpeg_data.data(), vec_jpeg_data.size());
    file.close();
}

static void image_save_as_jpeg(const cv::Mat &image, const std::string &path) { cv::imwrite(path, image); }

static cv::Mat image_to_mat(const std::shared_ptr<CnedkBufSurface> &image) {
    cv::Mat mat_image(image.surface_list->height * 3 / 2, image.surface_list->width, CV_8UC1);
    cnrtMemcpy(mat_image.data, image.surface_list->data_ptr, image.surface_list->width * image.surface_list->height,
               CNRT_MEM_TRANS_DIR_DEV2HOST);
    cnrtMemcpy(mat_image.data + image.surface_list->width * image.surface_list->height,
               (char *)image.surface_list->data_ptr
                   + image.surface_list->plane_params.pitch[0] * image.surface_list->height,
               image.surface_list->width * image.surface_list->height / 2, CNRT_MEM_TRANS_DIR_DEV2HOST);
               
    cv::cvtColor(mat_image, mat_image, cv::COLOR_YUV2BGRA_NV12);
    return mat_image;
}

static std::shared_ptr<CnedkBufSurface> image_jpeg_dec(const unsigned char *jpeg_data, size_t dsize) {
    // auto cndek_frame = alloc_cnedk_frame(CNEDK_BUF_COLOR_FORMAT_NV12, image_info.width, image_info.height);

    return {};
}

static auto image_crop(const std::shared_ptr<CnedkBufSurface> &src_frame, const std::map<int, Rect2f> &crop_rects) {
    auto crop_num = crop_rects.size();
    std::map<int, std::shared_ptr<CnedkBufSurface>> map_crop_images;

    for (const auto &[idx, rect] : crop_rects) {
        map_crop_images[idx] = alloc_cnedk_frame(CNEDK_BUF_COLOR_FORMAT_NV12, rect.width, rect.height);

        // cncodecRectangle rectangle{(uint32_t)rect.x, (uint32_t)rect.y, (uint32_t)(rect.x + rect.width),
        //                            (uint32_t)(rect.y + rect.height)};
        // cncodecWorkInfo work_info{.inMsg = {0, 0, 0, 0, 0}, .outMsg = {0.0, 0, 0}};
        // if (cncodecImageTransform(map_crop_images[idx].get(), nullptr, src_frame.get(), &rectangle,
        //                           cncodecFilter::CNCODEC_NumFilters, &work_info)
        //     != CNCODEC_SUCCESS) {
        //     throw std::runtime_error("crop cncodecImageTransform failed");
        // }
    }

    return map_crop_images;
}

static std::shared_ptr<MemObject<CnedkBufSurface>> image_from_avframe(ImagePool &mem_pool,
                                                                      const std::shared_ptr<AVFrame> &frame) {
    auto mem_obj = mem_pool.alloc_mem_detach();
    mem_obj->data = alloc_cnedk_frame(CNEDK_BUF_COLOR_FORMAT_NV12, frame->width, frame->height);
    cnrtMemcpy(mem_obj->data->surface_list->data_ptr, frame->data[0], frame->width * frame->height, cnrtMemcpyDevToDev);
    cnrtMemcpy(reinterpret_cast<char *>(mem_obj->data->surface_list->data_ptr) + frame->width * frame->height,
               frame->data[1], frame->width * frame->height / 2, cnrtMemcpyDevToDev);
    return mem_obj;
}

}// namespace image_wrapper
}// namespace gddi

#endif

#endif