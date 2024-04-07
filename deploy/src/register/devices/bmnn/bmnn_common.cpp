#include "bmnn_common.h"
#include <memory>

bm_image_format_ext convertSurfFormat2BmFormat(BufSurfaceColorFormat &fmt)
{
    if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_NV12){
        return FORMAT_NV12;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_NV21){
        return FORMAT_NV21;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_YUV420){
        return FORMAT_YUV420P;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_RGB){
        return FORMAT_RGB_PACKED;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_BGR){
        return FORMAT_BGR_PACKED;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER){
        return FORMAT_RGB_PLANAR;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER){
        return FORMAT_BGR_PLANAR;
    } 
    return FORMAT_RGB_PLANAR;
}

BufSurfaceColorFormat convertBmFormat2SurfFormat(bm_image_format_ext fmt)
{
    if (fmt == FORMAT_NV12) {
        return GDDEPLOY_BUF_COLOR_FORMAT_NV12;
    } else if (fmt == FORMAT_NV21) {
        return GDDEPLOY_BUF_COLOR_FORMAT_NV21;
    } else if (fmt == FORMAT_YUV420P) {
        return GDDEPLOY_BUF_COLOR_FORMAT_YUV420;
    } else if (fmt == FORMAT_RGB_PACKED) {
        return GDDEPLOY_BUF_COLOR_FORMAT_RGB;
    } else if (fmt == FORMAT_BGR_PACKED) {
        return GDDEPLOY_BUF_COLOR_FORMAT_BGR;
    } else if (fmt == FORMAT_RGB_PLANAR) {
        return GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;
    } else if (fmt == FORMAT_BGR_PLANAR) {
        return GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER;
    }

    return GDDEPLOY_BUF_COLOR_FORMAT_INVALID;
}

int getPlaneNumByFormat(BufSurfaceColorFormat fmt)
{
    int num_planes = 0;
    switch (fmt) {
        case GDDEPLOY_BUF_COLOR_FORMAT_RGB:
        case GDDEPLOY_BUF_COLOR_FORMAT_BGR:
            num_planes = 1;
            break;
        case GDDEPLOY_BUF_COLOR_FORMAT_NV12:
        case GDDEPLOY_BUF_COLOR_FORMAT_NV21:
            num_planes = 2;
            break;
        case GDDEPLOY_BUF_COLOR_FORMAT_YUV420:    
        case GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER:
        case GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER:
            num_planes = 3;
            break;
        case GDDEPLOY_BUF_COLOR_FORMAT_INVALID:
        default: {
            return -1;
        }
    }
    return num_planes;
}

#define ALIGN(x, align) ((x % align == 0) ? x : (x / align + 1) * align)

int getStride(int *default_stride, bm_image_format_ext fmt, bm_image_data_format_ext data_type, int width)
{
    int data_size = 1;
    switch (data_type) {
        case DATA_TYPE_EXT_FLOAT32:
            data_size = 4;
            break;
        case DATA_TYPE_EXT_4N_BYTE:
        case DATA_TYPE_EXT_4N_BYTE_SIGNED:
            data_size = 4;
            break;
        default:
            data_size = 1;
            break;
    }
    default_stride[0] = 0;
    default_stride[1] = 0;
    default_stride[2] = 0;
    switch (fmt) {
        case FORMAT_YUV420P: {
            default_stride[0] = width * data_size;
            default_stride[1] = (ALIGN(width, 2) >> 1) * data_size;
            default_stride[2] = default_stride[1];
            break;
        }
        case FORMAT_YUV422P: {
            default_stride[0] = width * data_size;
            default_stride[1] = (ALIGN(width, 2) >> 1) * data_size;
            default_stride[2] = default_stride[1];
            break;
        }
        case FORMAT_YUV444P: {
            default_stride[0] = width * data_size;
            default_stride[1] = width * data_size;
            default_stride[2] = default_stride[1];
            break;
        }
        case FORMAT_NV12:
        case FORMAT_NV21: {
            default_stride[0] = width * data_size;
            default_stride[1] = ALIGN(width, 2) * data_size;
            break;
        }
        case FORMAT_NV16:
        case FORMAT_NV61: {
            default_stride[0] = width * data_size;
            default_stride[1] = ALIGN(width, 2) * data_size;
            break;
        }
        case FORMAT_GRAY: {
            default_stride[0] = width * data_size;
            break;
        }
        case FORMAT_COMPRESSED: {
            break;
        }
        case FORMAT_BGR_PACKED:
        case FORMAT_RGB_PACKED: {
            default_stride[0] = width * 3 * data_size;
            break;
        }
        case FORMAT_BGR_PLANAR:
        case FORMAT_RGB_PLANAR: {
            default_stride[0] = width * data_size;
            break;
        }
        case FORMAT_BGRP_SEPARATE:
        case FORMAT_RGBP_SEPARATE: {
            default_stride[0] = width * data_size;
            default_stride[1] = width * data_size;
            default_stride[2] = width * data_size;
            break;
        }
    }
    default_stride[0] = ((default_stride[0] % 64 == 0) ? default_stride[0] : (default_stride[0] / 64 + 1) * 64);
    default_stride[1] = ((default_stride[1] % 64 == 0) ? default_stride[1] : (default_stride[1] / 64 + 1) * 64);
    default_stride[2] = ((default_stride[2] % 64 == 0) ? default_stride[2] : (default_stride[2] / 64 + 1) * 64);

}

bm_image_data_format_ext convertPixelWidth(uint32_t bytes_per_pix)
{
    bm_image_data_format_ext data_type = DATA_TYPE_EXT_1N_BYTE;
    if (bytes_per_pix == 1) {
        data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    } else if (bytes_per_pix == 4) {
        data_type = DATA_TYPE_EXT_FLOAT32;
    }

    return data_type;
}