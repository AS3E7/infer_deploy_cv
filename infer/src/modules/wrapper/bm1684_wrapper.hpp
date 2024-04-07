#ifndef __BM1684_WRAPPER_HPP__
#define __BM1684_WRAPPER_HPP__

#ifdef WITH_BM1684

#include "../mem_pool.hpp"
#include "basic_logs.hpp"
#include "types.hpp"
#include <core/mem/buf_surface.h>
#include <core/mem/buf_surface_impl.h>
#include <core/mem/buf_surface_util.h>
#include <fstream>
#include <string>
#include <vector>

#ifdef WITH_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bm_wrapper.hpp"
#include <bmcv_api_ext.h>
#include <opencv2/core/bmcv.hpp>

namespace gddi {

using ImagePool = gddi::MemPool<bm_image, bm_image_format_ext, int, int, bool>;

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

inline bm_image_format_ext map_avformat_to_bmformat(int avformat) {
    bm_image_format_ext format;
    switch (avformat) {
        case AV_PIX_FMT_YUV420P: format = FORMAT_YUV420P; break;
        case AV_PIX_FMT_YUV422P: format = FORMAT_YUV422P; break;
        case AV_PIX_FMT_YUV444P: format = FORMAT_YUV444P; break;
        case AV_PIX_FMT_NV12: format = FORMAT_NV12; break;
        case AV_PIX_FMT_RGBA: format = FORMAT_RGB_PACKED; break;
    }
    return format;
}

static gddeploy::BufSurfWrapperPtr convert_bm_image_to_sufsurface(const bm_image &img) {
    auto buf_surf = std::shared_ptr<gddeploy::BufSurfaceWrapper>(
        new gddeploy::BufSurfaceWrapper(new BufSurface(), false), [](gddeploy::BufSurfaceWrapper *ptr) {
            delete ptr->GetBufSurface()->surface_list;
            delete ptr->GetBufSurface();
            delete ptr;
        });

    auto surface = buf_surf->GetBufSurface();
    surface->mem_type = GDDEPLOY_BUF_MEM_BMNN;
    surface->batch_size = 1;
    surface->num_filled = 1;

    surface->surface_list = new BufSurfaceParams();
    surface->surface_list->color_format = GDDEPLOY_BUF_COLOR_FORMAT_NV12;
    surface->surface_list->data_size = img.width * img.height * 3 / 2;
    surface->surface_list->width = img.width;
    surface->surface_list->height = img.height;

    auto &plane_params = surface->surface_list->plane_params;
    plane_params.num_planes = bm_image_get_plane_num(img);
    plane_params.width[0] = img.width;
    plane_params.height[0] = img.height;

    buf_surf->GetSurfaceParams()->data_ptr = reinterpret_cast<void *>(img.image_private);
    bm_device_mem_t dev_mem[plane_params.num_planes];
    bm_image_get_device_mem(img, dev_mem);
    for (uint32_t i = 0; i < plane_params.num_planes; i++) {
        plane_params.data_ptr[i] = (void *)bm_mem_get_device_addr(dev_mem[i]);
    }

    return buf_surf;
}

static std::shared_ptr<AVFrame> avframe_cvt_format(const AVFrame *src_frame, const AVPixelFormat format) {
    auto dst_frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [](AVFrame *ptr) { av_frame_free(&ptr); });

    dst_frame->width = src_frame->width;
    dst_frame->height = src_frame->height;
    dst_frame->format = format;
    if (av_image_alloc(dst_frame->data, dst_frame->linesize, src_frame->width, src_frame->height, format, 64) < 0) {
        throw std::runtime_error("Could not allocate image\n");
    }

    SwsContext *conversion =
        sws_getContext(src_frame->width, src_frame->height, (AVPixelFormat)src_frame->format, dst_frame->width,
                       dst_frame->height, format, SWS_BILINEAR, NULL, NULL, NULL);
    sws_scale(conversion, src_frame->data, src_frame->linesize, 0, dst_frame->height, dst_frame->data,
              dst_frame->linesize);
    sws_freeContext(conversion);

    return dst_frame;
}

static std::shared_ptr<bm_handle_t> get_bm_handle(const int dev_id = 0) {
    auto handle = std::shared_ptr<bm_handle_t>(new bm_handle_t, [](bm_handle_t *ptr) { bm_dev_free(*ptr); });
    if (bm_dev_request(handle.get(), dev_id) != 0) {
        spdlog::error("** failed to request device");
        exit(1);
    }
    return handle;
}

static std::array<int, 3> get_bm_stride(const bm_image_format_ext format, const int width) {
    std::array<int, 3> stride;
    auto align_width = FFALIGN(width, 64);
    switch (format) {
        case FORMAT_NV12: stride = {align_width, align_width}; break;
        case FORMAT_YUV420P: stride = {align_width, align_width / 2, align_width / 2}; break;
        case FORMAT_RGB_PACKED:
        case FORMAT_BGR_PACKED: stride = {align_width * 3}; break;
        case FORMAT_RGB_PLANAR:
        case FORMAT_BGR_PLANAR: stride = {align_width, align_width, align_width}; break;
        case FORMAT_ARGB_PACKED:
        case FORMAT_ABGR_PACKED: stride = {align_width * 4}; break;
        default: stride = {align_width};
    }
    return stride;
}

const auto bm_handle = get_bm_handle(0);

static auto image_cvt_format(const bm_image &image, bm_image_format_ext format = FORMAT_YUV420P,
                             bm_image_data_format_ext data_format = DATA_TYPE_EXT_1N_BYTE) {
    auto cvt_image = std::unique_ptr<bm_image, void (*)(bm_image *)>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    bmcv_rect_t rect{0, 0, image.width, image.height};
    bm_image_create(*bm_handle, image.height, image.width, format, data_format, cvt_image.get(),
                    get_bm_stride(format, image.width).data());

    bm_image_alloc_dev_mem_heap_mask(*cvt_image, 6);

    if (bmcv_image_vpp_convert(*bm_handle, 1, image, cvt_image.get(), &rect) != BM_SUCCESS) {
        throw std::runtime_error("Failed to convert image format " + std::to_string(image.image_format).append(" -> ")
                                 + std::to_string(format));
    }

    return cvt_image;
}

static auto image_cvt_format(const std::shared_ptr<bm_image> &image, bm_image_format_ext format = FORMAT_YUV420P,
                             bm_image_data_format_ext data_format = DATA_TYPE_EXT_1N_BYTE) {
    auto cvt_image = std::unique_ptr<bm_image, void (*)(bm_image *)>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    bmcv_rect_t rect{0, 0, image->width, image->height};
    bm_image_create(*bm_handle, image->height, image->width, format, data_format, cvt_image.get(),
                    get_bm_stride(format, image->width).data());

    bm_image_alloc_dev_mem_heap_mask(*cvt_image, 6);

    if (bmcv_image_vpp_convert(*bm_handle, 1, *image, cvt_image.get(), &rect) != BM_SUCCESS) {
        throw std::runtime_error("Failed to convert image format " + std::to_string(image->image_format).append(" -> ")
                                 + std::to_string(format));
    }

    return cvt_image;
}

static auto image_clone(bm_image *src_img) {
    auto dst_img = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    int stride[3];
    bm_image_get_stride(*src_img, stride);
    bm_image_create(*bm_handle, src_img->height, src_img->width, src_img->image_format, src_img->data_type,
                    dst_img.get(), stride);
    bm_image_alloc_dev_mem_heap_mask(*dst_img, 6);

    int plane_num = bm_image_get_plane_num(*src_img);
    bm_device_mem_t src_mem[plane_num];
    bm_device_mem_t dst_mem[plane_num];
    bm_image_get_device_mem(*src_img, src_mem);
    bm_image_get_device_mem(*dst_img, dst_mem);

    for (int i = 0; i < plane_num; i++) { bm_memcpy_c2c(*bm_handle, *bm_handle, src_mem[i], dst_mem[i], false); }

    return dst_img;
}

static auto image_resize(const std::shared_ptr<bm_image> &image, int width, int height) {
    if (image->width == width && image->height == height) { return image; }

    auto src_image = image;

    auto cvt_image = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    bm_image_create(*bm_handle, height, width, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, cvt_image.get(),
                    get_bm_stride(FORMAT_YUV420P, width).data());
    bm_image_alloc_dev_mem_heap_mask(*cvt_image, 6);

    if (image->image_format == bm_image_format_ext::FORMAT_YUV422P
        || image->image_format == bm_image_format_ext::FORMAT_YUV444P
        || image->image_format == bm_image_format_ext::FORMAT_NV16
        || image->image_format == bm_image_format_ext::FORMAT_NV24) {
        src_image = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
            bm_image_destroy(*ptr);
            delete ptr;
        });
        bm_image_create(*bm_handle, image->height, image->width, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, src_image.get(),
                        get_bm_stride(FORMAT_YUV420P, image->width).data());
        bm_image_alloc_dev_mem_heap_mask(*src_image, 6);

        if (bmcv_image_storage_convert(*bm_handle, 1, image.get(), src_image.get()) != BM_SUCCESS) {
            throw std::runtime_error("Failed to resize image, format " + std::to_string(image->image_format));
        }
    }

    bmcv_rect_t rect{0, 0, image->width, image->height};
    if (bmcv_image_vpp_convert(*bm_handle, 1, *src_image, cvt_image.get(), &rect) != BM_SUCCESS) {
        throw std::runtime_error("Failed to resize image, format " + std::to_string(src_image->image_format));
    }

    return cvt_image;
}

static auto image_crop(const std::shared_ptr<bm_image> &image, std::map<int, Rect2f> &crop_rects) {
    auto src_image = image;
    if (src_image->image_format != FORMAT_NV12) { src_image = image_cvt_format(image, FORMAT_YUV420P); }

    auto crop_images =
        std::shared_ptr<std::vector<bm_image>>(new std::vector<bm_image>(), [](std::vector<bm_image> *vec_images) {
            for (int i = 0; i < vec_images->size(); i++) { bm_image_destroy(*(vec_images->data() + i)); }
            delete vec_images;
        });

    int index = 0;
    auto crop_num = crop_rects.size();

    std::vector<int> crop_idxes;
    bmcv_rect_t bm_crop_rects[crop_num];
    for (const auto &[idx, rect] : crop_rects) {
        bm_crop_rects[index++] = {(int)rect.x, (int)rect.y, (int)rect.width, (int)rect.height};
        crop_idxes.push_back(idx);

        bm_image crop_image;
        bm_image_create(*bm_handle, rect.height, rect.width, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &crop_image,
                        get_bm_stride(FORMAT_YUV420P, rect.width).data());
        bm_image_alloc_dev_mem_heap_mask(crop_image, 6);
        crop_images->push_back(std::move(crop_image));
    }

    int status = bmcv_image_vpp_convert(*bm_handle, crop_num, *src_image, crop_images->data(), bm_crop_rects);
    if (status != BM_SUCCESS) {
        throw std::runtime_error("Failed to crop image format, status: " + std::to_string(status));
    }

    return std::make_pair(crop_idxes, crop_images);
}

static std::shared_ptr<bm_image> image_png_dec(const unsigned char *raw_image, size_t dsize) {
    auto vec_data = std::vector<uchar>(dsize);
    memcpy(vec_data.data(), raw_image, dsize);

    cv::Mat mat_image = cv::imdecode(vec_data, cv::ImreadModes::IMREAD_COLOR);
    auto dst_img = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    if (cv::bmcv::toBMI(mat_image, dst_img.get()) != BM_SUCCESS) {
        throw std::runtime_error("Failed to convert bm_image");
    }

    return dst_img;
}

static auto image_raw2yuv(unsigned char *raw_image, size_t dsize, int width, int height) {
    auto dst_image = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    bm_image_create(*bm_handle, height, width, FORMAT_NV12, DATA_TYPE_EXT_1N_BYTE, dst_image.get(),
                    get_bm_stride(FORMAT_NV12, width).data());
    bm_image_alloc_dev_mem_heap_mask(*dst_image, 6);

    bm_device_mem_t dst_mem[2];
    bm_image_get_device_mem(*dst_image, dst_mem);
    bm_memcpy_s2d(*bm_handle, dst_mem[0], raw_image);
    bm_memcpy_s2d(*bm_handle, dst_mem[1], raw_image + width * height);

    return dst_image;
}

static std::shared_ptr<bm_image> image_jpeg_dec(const unsigned char *data, size_t dsize) {
    auto dst_image = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    unsigned char *jpeg_data = const_cast<unsigned char *>(data);
    if (bmcv_image_jpeg_dec(*bm_handle, (void **)&jpeg_data, &dsize, 1, dst_image.get()) != BM_SUCCESS) {
        throw std::runtime_error("Failed to decode images");
    }

    if (FFALIGN(dst_image->width, 2) != dst_image->width || FFALIGN(dst_image->height, 2) != dst_image->height) {
        dst_image = image_resize(dst_image, FFALIGN(dst_image->width, 2), FFALIGN(dst_image->height, 2));
    }

    return dst_image;
}

static void image_jpeg_enc(const std::shared_ptr<bm_image> &image, std::vector<uchar> &vec_jpeg_data,
                           int &jpeg_data_size, const int quality = 85) {
    auto yuv_image = image;
    if (image->image_format != FORMAT_NV12 || image->image_format != FORMAT_YUV420P) {
        yuv_image = image_cvt_format(image);
    }

    void *jpeg_data = nullptr;
    if (bmcv_image_jpeg_enc(*bm_handle, 1, yuv_image.get(), &jpeg_data, (size_t *)&jpeg_data_size, quality)
        != BM_SUCCESS) {
        throw std::runtime_error("Failed to encode image");
    }

    memcpy(vec_jpeg_data.data(), jpeg_data, jpeg_data_size);
    free(jpeg_data);
}

static void image_jpeg_enc(std::shared_ptr<bm_image> &image, std::vector<uchar> &vec_jpeg_data,
                           const int quality = 85) {
    void *jpeg_data = nullptr;
    size_t size = 0;
    if (bmcv_image_jpeg_enc(*bm_handle, 1, image.get(), &jpeg_data, &size, quality) != BM_SUCCESS) {
        throw std::runtime_error("Failed to encode image");
    }

    vec_jpeg_data = std::vector<uchar>(size);
    memcpy(vec_jpeg_data.data(), jpeg_data, size);
    free(jpeg_data);
}

static void image_jpeg_enc(bm_image &image, std::vector<uchar> &vec_jpeg_data, const int quality = 85) {
    void *jpeg_data = nullptr;
    size_t size = 0;
    if (bmcv_image_jpeg_enc(*bm_handle, 1, &image, &jpeg_data, &size, quality) != BM_SUCCESS) {
        throw std::runtime_error("Failed to encode image");
    }

    vec_jpeg_data = std::vector<uchar>(size);
    memcpy(vec_jpeg_data.data(), jpeg_data, size);
    free(jpeg_data);
}

static int image_jpeg_enc(const std::shared_ptr<bm_image> &image, std::array<char, 49766400> &jpeg_data,
                          const int quality = 85) {
    auto yuv_image = image;
    if (image->image_format != FORMAT_NV12 || image->image_format != FORMAT_YUV420P) {
        yuv_image = image_cvt_format(image);
    }

    size_t size = 0;
    auto ptr = (void *)jpeg_data.data();
    if (bmcv_image_jpeg_enc(*bm_handle, 1, yuv_image.get(), &ptr, &size, quality) != BM_SUCCESS) {
        spdlog::error("Failed to encode image");
        return 0;
    }

    return size;
}

static void image_jpeg_enc(const std::shared_ptr<std::vector<bm_image>> &crop_images,
                           std::vector<std::vector<uchar>> &vec_jpeg_data, const int quality = 85) {
    int crop_num = crop_images->size();

    void *jpeg_data[crop_num]{nullptr};
    size_t jpeg_data_size[crop_num]{0};
    if (bmcv_image_jpeg_enc(*bm_handle, crop_num, crop_images->data(), jpeg_data, jpeg_data_size, quality)
        != BM_SUCCESS) {
        throw std::runtime_error("Failed to encode image");
    }

    for (int i = 0; i < crop_num; i++) {
        vec_jpeg_data.push_back(std::vector<uchar>(jpeg_data_size[i]));
        memcpy(vec_jpeg_data[i].data(), jpeg_data[i], jpeg_data_size[i]);
        free(jpeg_data[i]);
    }
}

static std::vector<char> base64_enc(std::vector<uchar> &raw_data) {
    long unsigned b64_len[] = {raw_data.size(), 0};
    auto b64_data = std::vector<char>((b64_len[0] + 2) / 3 * 4 + 3);
    if (bmcv_base64_enc(*bm_handle, bm_mem_from_system(raw_data.data()), bm_mem_from_system(b64_data.data()), b64_len)
        != BM_SUCCESS) {
        throw std::runtime_error("Failed to encode base64");
    }
    b64_data.resize(b64_len[1]);
    return b64_data;
}

static std::shared_ptr<bm_image> mat_to_image(const cv::Mat &image) {
    auto dst_image = std::shared_ptr<bm_image>(new bm_image, [](bm_image *ptr) {
        bm_image_destroy(*ptr);
        delete ptr;
    });

    cv::Mat yuv_image;
    cv::cvtColor(image, yuv_image, cv::COLOR_BGR2YUV_I420);

    bm_image_create(*bm_handle, image.rows, image.cols, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, dst_image.get(),
                    get_bm_stride(FORMAT_YUV420P, image.cols).data());
    bm_image_alloc_dev_mem_heap_mask(*dst_image, 6);

    bm_device_mem_t dst_mem[3];
    bm_image_get_device_mem(*dst_image, dst_mem);
    bm_memcpy_s2d(*bm_handle, dst_mem[0], yuv_image.data);
    bm_memcpy_s2d(*bm_handle, dst_mem[1], yuv_image.data + image.cols * image.rows);
    bm_memcpy_s2d(*bm_handle, dst_mem[2], yuv_image.data + image.cols * image.rows * 5 / 4);

    return dst_image;
}

static cv::Mat image_to_mat(const std::shared_ptr<bm_image> &image) {
    cv::Mat mat_image;
    cv::bmcv::toMAT(image.get(), mat_image);
    return mat_image.clone();
}

static void image_save_as_jpeg(const std::shared_ptr<bm_image> &image, const std::string &path,
                               const int quality = 85) {
    auto yuv_image = image;
    if (image->image_format != FORMAT_NV12 && image->image_format != FORMAT_YUV420P) {
        yuv_image = image_cvt_format(image);
    }

    void *p_jpeg_data = nullptr;
    size_t out_size = 0;
    if (bmcv_image_jpeg_enc(*bm_handle, 1, yuv_image.get(), &p_jpeg_data, &out_size, quality) != 0) {
        throw std::runtime_error("Failed to save as jpeg: " + path);
    }

    FILE *file = fopen(path.c_str(), "wb");
    if (file) { fwrite(p_jpeg_data, sizeof(uint8_t), out_size, file); }
    fclose(file);

    if (p_jpeg_data) { free(p_jpeg_data); }
}

static void image_save_as_jpeg(const cv::Mat &image, const std::string &path, const int quality = 85) {
    cv::imwrite(path, image);
}

static void image_save_as_yuv(const std::shared_ptr<bm_image> &image, const std::string &path) {
    FILE *file = fopen(path.c_str(), "wb");
    uint8_t *yuv_mem[2];
    yuv_mem[0] = new uint8_t[image->width * image->height];
    yuv_mem[1] = new uint8_t[image->width * image->height / 2];
    bm_image_copy_device_to_host(*image, (void **)yuv_mem);
    fwrite(yuv_mem[0], sizeof(uint8_t), image->width * image->height, file);
    fwrite(yuv_mem[1], sizeof(uint8_t), image->width * image->height / 2, file);
    delete yuv_mem[0];
    delete yuv_mem[1];
    fclose(file);
}

static void load_yuv_image(std::vector<uint8_t> &yuv_data, const int width, const int height,
                           std::shared_ptr<bm_image> &image) {
    bm_device_mem_t src_mem[2];
    bm_image_get_device_mem(*image, src_mem);
    bm_memcpy_s2d_partial(*image_wrapper::bm_handle, src_mem[0], yuv_data.data(), width * height);
    bm_memcpy_s2d_partial(*image_wrapper::bm_handle, src_mem[1], yuv_data.data() + width * height, width * height / 2);
}

static void save_as_png(bm_image *image, const std::string &path) {
    cv::Mat out;
    cv::bmcv::toMAT(image, out);
    cv::imwrite(path, out);
}

static bm_status_t bm_image_from_frame(const AVFrame &in, bm_image &out) {
    if (in.format == AV_PIX_FMT_NV12) {
        if (in.channel_layout == 101) { /* COMPRESSED NV12 FORMAT */
            /* sanity check */
            if ((0 == in.height) || (0 == in.width) || (0 == in.linesize[4]) || (0 == in.linesize[5])
                || (0 == in.linesize[6]) || (0 == in.linesize[7]) || (0 == in.data[4]) || (0 == in.data[5])
                || (0 == in.data[6]) || (0 == in.data[7])) {
                std::cout << "bm_image_from_frame: get yuv failed!!" << std::endl;
                return BM_ERR_PARAM;
            }
            bm_image cmp_bmimg;
            bm_image_create(*bm_handle, in.height, in.width, FORMAT_COMPRESSED, DATA_TYPE_EXT_1N_BYTE, &cmp_bmimg);

            /* calculate physical address of avframe */
            bm_device_mem_t input_addr[4];
            int size = in.height * in.linesize[4];
            input_addr[0] = bm_mem_from_device((unsigned long long)in.data[6], size);
            size = (in.height / 2) * in.linesize[5];
            input_addr[1] = bm_mem_from_device((unsigned long long)in.data[4], size);
            size = in.linesize[6];
            input_addr[2] = bm_mem_from_device((unsigned long long)in.data[7], size);
            size = in.linesize[7];
            input_addr[3] = bm_mem_from_device((unsigned long long)in.data[5], size);
            bm_image_attach(cmp_bmimg, input_addr);

            bmcv_rect_t crop_rect = {0, 0, in.width, in.height};
            bmcv_image_vpp_convert(*bm_handle, 1, cmp_bmimg, &out, &crop_rect);
            bm_image_destroy(cmp_bmimg);
        } else { /* UNCOMPRESSED NV12 FORMAT */
            /* sanity check */
            if ((0 == in.height) || (0 == in.width) || (0 == in.linesize[4]) || (0 == in.linesize[5])
                || (0 == in.data[4]) || (0 == in.data[5])) {
                std::cout << "bm_image_from_frame: get yuv failed!!" << std::endl;
                return BM_ERR_PARAM;
            }

            if (bm_image_is_attached(out)) {
                bm_device_mem_t out_mem[2];
                bm_image_get_device_mem(out, out_mem);

                /* calculate physical address of yuv mat */
                bm_device_mem_t input_addr[2];
                int size = in.height * in.linesize[4];
                input_addr[0] = bm_mem_from_device((unsigned long long)in.data[4], size);
                bm_memcpy_c2c(*bm_handle, *bm_handle, input_addr[0], out_mem[0], true);
                size = in.height * in.linesize[5] / 2;
                input_addr[1] = bm_mem_from_device((unsigned long long)in.data[5], size);
                bm_memcpy_c2c(*bm_handle, *bm_handle, input_addr[1], out_mem[1], true);
            } else {
                /* calculate physical address of yuv mat */
                bm_device_mem_t input_addr[2];
                int size = in.height * in.linesize[4];
                input_addr[0] = bm_mem_from_device((unsigned long long)in.data[4], size);
                size = in.height * in.linesize[5] / 2;
                input_addr[1] = bm_mem_from_device((unsigned long long)in.data[5], size);

                bm_image_attach(out, input_addr);
            }
        }
    } else {
        // host to device
        int plane_num = bm_image_get_plane_num(out);
        auto out_mem = std::vector<bm_device_mem_t>(plane_num);
        bm_image_get_device_mem(out, out_mem.data());
        for (auto i = 0; i < plane_num; i++) { bm_memcpy_s2d(*bm_handle, out_mem[i], in.data[i]); }
    }

    return BM_SUCCESS;
}

static std::shared_ptr<MemObject<bm_image, std::shared_ptr<AVFrame>>>
image_from_avframe(ImagePool &mem_pool, const std::shared_ptr<AVFrame> &frame) {
    auto cvt_frame = frame;
    std::shared_ptr<MemObject<bm_image, std::shared_ptr<AVFrame>>> mem_obj;
    if (frame->format == AV_PIX_FMT_NV12) {
        if (frame->channel_layout == 101) {
            mem_obj = mem_pool.alloc_mem_attach<std::shared_ptr<AVFrame>>(nullptr, FORMAT_YUV420P, frame->width,
                                                                          frame->height, true);
        } else {
            mem_obj = mem_pool.alloc_mem_attach<std::shared_ptr<AVFrame>>(nullptr, FORMAT_NV12, frame->width,
                                                                          frame->height, true);
        }
    } else {
        if (convert_deprecated_format((AVPixelFormat)frame->format) == AV_PIX_FMT_YUV420P) {
            mem_obj = mem_pool.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, FORMAT_YUV420P, frame->width,
                                                                          frame->height, true);
        } else {
            cvt_frame = avframe_cvt_format(frame.get(), AV_PIX_FMT_RGB24);
            mem_obj = mem_pool.alloc_mem_detach<std::shared_ptr<AVFrame>>(nullptr, FORMAT_RGB_PACKED, frame->width,
                                                                          frame->height, true);
        }
    }

    image_wrapper::bm_image_from_frame(*cvt_frame, *mem_obj->data);
    if (FFALIGN(mem_obj->data->width, 2) != mem_obj->data->width
        || FFALIGN(mem_obj->data->height, 2) != mem_obj->data->height) {
        mem_obj->data =
            image_resize(mem_obj->data, FFALIGN(mem_obj->data->width, 2), FFALIGN(mem_obj->data->height, 2));
    }

    return mem_obj;
}

}// namespace image_wrapper
}// namespace gddi
#endif

#endif