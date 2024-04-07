#ifndef __RV1126_WRAPPER_HPP__
#define __RV1126_WRAPPER_HPP__

#if defined(WITH_RV1126)

#ifdef WITH_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

#include "../mem_pool.hpp"
#include "spdlog/spdlog.h"
#include "types.hpp"
#include <fstream>
#include <map>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <rga/RgaUtils.h>
#include <rga/im2d.h>
#include <rga/rga.h>
#include <rkmedia/rkmedia_api.h>

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))

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

class DevHandle {
public:
    DevHandle(int dev_id = 0) {
        dev_id_ = dev_id;
        RK_MPI_SYS_Init();
    }
    ~DevHandle() = default;

    int dev_id() { return dev_id_; }

private:
    DevHandle(const DevHandle &) = delete;
    DevHandle &operator=(const DevHandle &) = delete;

private:
    int dev_id_;
};

static auto s_handle = std::make_shared<DevHandle>();

static cv::Mat image_to_mat(const std::shared_ptr<cv::Mat> &image) { return *image; }

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

static std::shared_ptr<cv::Mat> image_png_dec(const unsigned char *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    auto mat_image = std::make_shared<cv::Mat>();
    *mat_image = cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR);
    return mat_image;
}

static auto image_resize(const std::shared_ptr<cv::Mat> &image, int width, int height) {
    cv::Mat resize_image;
    if (image->cols == width && image->rows * 2 / 3 == height) {
        return *image;
    } else {
        resize_image = cv::Mat(height * 2 / 3, width, CV_8UC1);
        auto src = wrapbuffer_virtualaddr(image->data, image->cols, image->rows * 2 / 3, RK_FORMAT_YCbCr_420_P);
        auto dst = wrapbuffer_virtualaddr(resize_image.data, width, height, RK_FORMAT_YCbCr_420_P);
        imresize(src, dst);
    }

    return resize_image;
}

static auto image_crop(const std::shared_ptr<cv::Mat> &image, std::map<int, Rect2f> &crop_rects) {
    std::map<int, cv::Mat> map_crop_images;

    for (const auto &[idx, rect] : crop_rects) {
        auto crop_image = cv::Mat((int)rect.height, (int)rect.width, CV_8UC3);

        im_rect src_rect{ALIGN((int)rect.x, 2), ALIGN((int)rect.y, 2), (int)rect.width, (int)rect.height};
        auto src = wrapbuffer_virtualaddr(image->data, image->cols, image->rows * 2 / 3, RK_FORMAT_YCbCr_420_P);
        auto dst = wrapbuffer_virtualaddr(crop_image.data, (int)rect.width, (int)rect.height, RK_FORMAT_BGR_888);
        if (imcrop(src, dst, src_rect) == IM_STATUS_SUCCESS) {
            map_crop_images.insert(std::make_pair(idx, std::move(crop_image)));
        } else {
            printf("Failed to crop image, x: %.2f, y: %.2f, w: %.2f, h: %.2f\n", rect.x, rect.y, rect.width,
                   rect.height);
        }

        // cv::imwrite(std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        //                                std::chrono::system_clock::now().time_since_epoch())
        //                                .count())
        //                 + ".jpg",
        //             map_crop_images.at(idx));
    }

    return map_crop_images;
}

static cv::Mat yuv420p_to_bgr888(const cv::Mat &src_image) {
    auto dst_image = cv::Mat(src_image.rows * 2 / 3, src_image.cols, CV_8UC3);
    auto src = wrapbuffer_virtualaddr(src_image.data, src_image.cols, src_image.rows * 2 / 3, RK_FORMAT_YCbCr_420_P);
    auto dst = wrapbuffer_virtualaddr(dst_image.data, src_image.cols, src_image.rows * 2 / 3, RK_FORMAT_BGR_888);
    imcvtcolor(src, dst, src.format, dst.format);
    return dst_image;
}

static std::shared_ptr<MemObject<cv::Mat>> image_from_avframe(ImagePool &mem_pool,
                                                              const std::shared_ptr<AVFrame> &frame) {
    auto mem_obj = mem_pool.alloc_mem_detach(frame->width, frame->height);
    *mem_obj->data = cv::Mat(frame->height * 3 / 2, frame->width, CV_8UC1);
    memcpy(mem_obj->data->data, frame->data[0], frame->width * frame->height);
    memcpy(mem_obj->data->data + frame->width * frame->height, frame->data[1], frame->height * frame->width / 4);
    memcpy(mem_obj->data->data + frame->width * frame->height * 5 / 4, frame->data[2],
           frame->height * frame->width / 4);
    return mem_obj;
}

}// namespace image_wrapper
}// namespace gddi

#endif

#endif