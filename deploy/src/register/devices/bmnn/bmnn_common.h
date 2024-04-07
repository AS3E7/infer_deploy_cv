#pragma once
#include "bmcv_api_ext.h"

#include "core/mem/buf_surface.h"

bm_image_format_ext convertSurfFormat2BmFormat(BufSurfaceColorFormat &format);
BufSurfaceColorFormat convertBmFormat2SurfFormat(bm_image_format_ext &format);
bm_image_data_format_ext convertPixelWidth(uint32_t bytes_per_pix);

int getPlaneNumByFormat(BufSurfaceColorFormat fmt);

int getStride(int *default_stride, bm_image_format_ext fmt, bm_image_data_format_ext data_type, int width);