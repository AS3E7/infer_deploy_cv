#ifndef __INTEL_WRAPPER_HPP__
#define __INTEL_WRAPPER_HPP__

#if defined(WITH_INTEL)

#ifdef WITH_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

#include "../mem_pool.hpp"
#include "types.hpp"
#include <fstream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

namespace gddi {

using ImagePool = gddi::MemPool<cv::Mat, int, int>;

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

static cv::Mat image_to_mat(const std::shared_ptr<cv::Mat> &image) {
    return *image;
}

static std::shared_ptr<cv::Mat> mat_to_image(const cv::Mat &image) {
    auto mat_image = std::make_shared<cv::Mat>();
    *mat_image = image;
    return mat_image;
}

static std::shared_ptr<cv::Mat> image_jpeg_dec(const unsigned char *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    auto mat_image = std::make_shared<cv::Mat>();
    *mat_image = cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR);
    return mat_image;
}

static void image_jpeg_enc(const cv::Mat &image, std::vector<uchar> &vec_jpeg_data, int &jpeg_data_size, const int quality = 85) {
    cv::imencode(".jpg", image, vec_jpeg_data, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, quality});
    jpeg_data_size = vec_jpeg_data.size();
}

static void image_jpeg_enc(const std::shared_ptr<cv::Mat> &image, std::vector<uchar> &vec_jpeg_data, int &jpeg_data_size, const int quality = 85) {
    cv::imencode(".jpg", *image, vec_jpeg_data, std::vector<int>{cv::IMWRITE_JPEG_QUALITY, quality});
    jpeg_data_size = vec_jpeg_data.size();
}

static std::shared_ptr<cv::Mat> image_png_dec(const unsigned char *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    auto mat_image = std::make_shared<cv::Mat>();
    *mat_image = cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR);
    return mat_image;
}

static auto image_resize(const std::shared_ptr<cv::Mat> &image, int width, int height) {
    cv::Mat resize_image;
    cv::resize(*image, resize_image, cv::Size(width, height));
    return resize_image;
}

static auto image_crop(const std::shared_ptr<cv::Mat> &image, std::map<int, Rect2f> &crop_rects) {
    auto crop_num = crop_rects.size();
    std::map<int, cv::Mat> map_crop_images;

    for (const auto &[idx, rect] : crop_rects) {
        (*image)(cv::Rect2f{rect.x, rect.y, rect.width, rect.height}).copyTo(map_crop_images[idx]);
    }

    return map_crop_images;
}

static void image_save_as_jpeg(const std::shared_ptr<cv::Mat> &image, const std::string &path, const int quality = 85) {
    cv::imwrite(path, *image);
}

static void image_save_as_jpeg(const cv::Mat &image, const std::string &path, const int quality = 85) {
    cv::imwrite(path, image);
}

static cv::Mat avframe_to_cvmat(const AVFrame *frame) {
    cv::Mat cv_frame;

    int width = frame->width;
    int height = frame->height;

    auto dst_frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [] (AVFrame *p) { av_frame_free(&p); });
    dst_frame->format = AV_PIX_FMT_BGR24;
    dst_frame->width = frame->width;
    dst_frame->height = frame->height;
    int buf_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, frame->width, frame->height, 1);
    if (buf_size < 0) { throw std::runtime_error("fail to calc frame buffer size"); }
    if (av_frame_get_buffer(dst_frame.get(), 1) != 0) { throw std::runtime_error("fail to alloc buffer"); }

    SwsContext *conversion =
        sws_getContext(width, height, convert_deprecated_format((AVPixelFormat)frame->format), width, height,
                       AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, frame->data, frame->linesize, 0, height, dst_frame->data, dst_frame->linesize);
    sws_freeContext(conversion);
    cv_frame = cv::Mat(height, width, CV_8UC3, dst_frame->data[0], dst_frame->linesize[0]).clone();

    return cv_frame;
}

static std::shared_ptr<MemObject<cv::Mat>> image_from_avframe(ImagePool &mem_pool,
                                                              const std::shared_ptr<AVFrame> &frame) {
    auto mem_obj = mem_pool.alloc_mem_detach(frame->width, frame->height);
    if (frame->format == AV_PIX_FMT_QSV || frame->format == AV_PIX_FMT_VAAPI) {
        auto av_frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [] (AVFrame *p) { av_frame_free(&p); });
        if (av_hwframe_transfer_data(av_frame.get(), frame.get(), 0) < 0) {
            throw std::runtime_error("Error transferring the data to system memory");
        }
        av_frame_copy_props(av_frame.get(), frame.get());
        *mem_obj->data = avframe_to_cvmat(av_frame.get());
    } else {
        *mem_obj->data = avframe_to_cvmat(frame.get());
    }
    return mem_obj;
}

}// namespace image_wrapper
}// namespace gddi

#endif

#endif