#ifndef __NVIDIA_WRAPPER_HPP__
#define __NVIDIA_WRAPPER_HPP__

#if defined(WITH_NVIDIA)

#ifdef WITH_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

#include "../mem_pool.hpp"
#include "../types.hpp"
#include <cuda_runtime_api.h>
#include <fstream>
#include <map>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

namespace gddi {

using ImagePool = gddi::MemPool<cv::cuda::GpuMat, int, int>;

namespace image_wrapper {

inline AVPixelFormat convert_deprecated_format(enum AVPixelFormat format) {
    switch (format) {
        case AV_PIX_FMT_YUVJ420P: return AV_PIX_FMT_YUV420P; break;
        case AV_PIX_FMT_YUVJ422P: return AV_PIX_FMT_YUV422P; break;
        case AV_PIX_FMT_YUVJ444P: return AV_PIX_FMT_YUV444P; break;
        case AV_PIX_FMT_YUVJ440P: return AV_PIX_FMT_YUV440P; break;
        default: return format; break;
    }
}

static std::shared_ptr<cv::cuda::GpuMat> image_jpeg_dec(const uint8_t *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    auto gpu_image = std::make_shared<cv::cuda::GpuMat>();
    gpu_image->upload(cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR));

    return gpu_image;
}

static void image_jpeg_enc(const cv::cuda::GpuMat &image, std::vector<uchar> &vec_jpeg_data, int &jpeg_data_size,
                           const int quality = 85) {
    cv::Mat mat_image;
    image.download(mat_image);
    cv::imencode(".jpg", mat_image, vec_jpeg_data, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, quality});
    jpeg_data_size = vec_jpeg_data.size();
}

static void image_jpeg_enc(const std::shared_ptr<cv::cuda::GpuMat> &image, std::vector<uchar> &vec_jpeg_data,
                           int &jpeg_data_size, const int quality = 85) {
    cv::Mat mat_image;
    image->download(mat_image);
    cv::imencode(".jpg", mat_image, vec_jpeg_data, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, quality});
    jpeg_data_size = vec_jpeg_data.size();
}

static std::shared_ptr<cv::cuda::GpuMat> image_png_dec(const unsigned char *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    auto mat_image = cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR);

    void *p_dst_data;
    cudaMalloc(&p_dst_data, mat_image.rows * mat_image.cols * 3);
    auto gpu_image =
        std::make_shared<cv::cuda::GpuMat>(mat_image.rows, mat_image.cols, CV_8UC3, p_dst_data, mat_image.cols * 3);
    cudaMemcpy(gpu_image->data, mat_image.data, mat_image.rows * mat_image.cols * 3, cudaMemcpyHostToDevice);

    return gpu_image;
}

static auto image_resize(const std::shared_ptr<cv::cuda::GpuMat> &image, int width, int height) {
    if (image->cols == width && image->rows == height) { return *image; }

    cv::cuda::GpuMat resize_image;
    cv::cuda::resize(*image, resize_image, cv::Size(width, height));
    return resize_image;
}

static auto image_crop(const std::shared_ptr<cv::cuda::GpuMat> &image, std::map<int, Rect2f> &crop_rects,
                       std::vector<std::shared_ptr<uint8_t>> &crop_data) {
    std::map<int, cv::cuda::GpuMat> map_crop_images;

    // TODO: 需手动申请释放, cv::cuda::GpuMat 默认申请会对齐内存
    for (const auto &[idx, rect] : crop_rects) {
        uint8_t *p_dst_data;
        cudaMalloc(&p_dst_data, rect.width * rect.height * 3);
        map_crop_images[idx] = cv::cuda::GpuMat(rect.height, rect.width, CV_8UC3, p_dst_data, rect.width * 3);
        (*image)(cv::Rect2f{rect.x, rect.y, rect.width, rect.height}).copyTo(map_crop_images[idx]);
        crop_data.emplace_back(std::shared_ptr<uint8_t>(p_dst_data, cudaFree));
    }

    return map_crop_images;
}

static std::shared_ptr<cv::cuda::GpuMat> mat_to_image(const cv::Mat &image) {
    void *p_dst_data;
    cudaMalloc(&p_dst_data, image.rows * image.cols * 3);
    auto gpu_image = std::make_shared<cv::cuda::GpuMat>(image.rows, image.cols, CV_8UC3, p_dst_data, image.cols * 3);
    gpu_image->upload(image);
    return gpu_image;
}

static void image_save_as_jpeg(const std::shared_ptr<cv::cuda::GpuMat> &image, const std::string &path,
                               const int quality = 85) {
    cv::Mat mat_Image;
    image->download(mat_Image);
    cv::imwrite(path, mat_Image);
}

static void image_save_as_jpeg(const cv::Mat &image, const std::string &path, const int quality = 85) {
    cv::imwrite(path, image);
}

static cv::Mat image_to_mat(const std::shared_ptr<cv::cuda::GpuMat> &image) {
    cv::Mat mat_image;
    image->download(mat_image);
    return mat_image;
}

static cv::Mat avframe_to_cvmat(const AVFrame *frame) {
    cv::Mat cv_frame;

    int width = frame->width;
    int height = frame->height;

    auto dst_frame = av_frame_alloc();
    dst_frame->format = AV_PIX_FMT_BGR24;
    dst_frame->width = frame->width;
    dst_frame->height = frame->height;
    int buf_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
    if (buf_size < 0) { throw std::runtime_error("fail to calc frame buffer size"); }
    if (av_frame_get_buffer(dst_frame, 1) != 0) { throw std::runtime_error("fail to alloc buffer"); }

    SwsContext *conversion =
        sws_getContext(width, height, convert_deprecated_format((AVPixelFormat)frame->format), width, height,
                       AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, frame->data, frame->linesize, 0, height, dst_frame->data, dst_frame->linesize);
    sws_freeContext(conversion);
    cv_frame = cv::Mat(height, width, CV_8UC3, dst_frame->data[0], dst_frame->linesize[0]).clone();
    av_frame_free(&dst_frame);

    return cv_frame;
}

static std::shared_ptr<MemObject<cv::cuda::GpuMat>> image_from_avframe(ImagePool &mem_pool,
                                                                       const std::shared_ptr<AVFrame> &frame) {
    auto mem_obj = mem_pool.alloc_mem_detach(frame->width, frame->height);
    mem_obj->data->create(frame->height, frame->width, CV_8UC3);
    mem_obj->data->step = frame->width * 3;
    nppiNV12ToBGR_8u_P2C3R(frame->data, frame->linesize[0], mem_obj->data->data, mem_obj->data->step1(),
                           NppiSize{frame->width, frame->height});

    return mem_obj;
}

}// namespace image_wrapper
}// namespace gddi

#endif

#endif