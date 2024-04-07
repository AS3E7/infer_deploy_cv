#include "rk_common.h"
#include <memory>

#include "RgaUtils.h"
#include "im2d.hpp"
#include "rga.h"

int convertSurfFormat2RKFormat(BufSurfaceColorFormat &fmt)
{
    RgaSURF_FORMAT format = RK_FORMAT_RGB_888;
    if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_NV12){
        return RK_FORMAT_YVYU_420;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_NV21){
        return RK_FORMAT_YVYU_420;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_YUV420){
        return RK_FORMAT_YCbCr_420_P;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_RGB){
        return RK_FORMAT_RGB_888;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_BGR){
        return RK_FORMAT_BGR_888;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER){
        return RK_FORMAT_RGB_888;
    } else if (fmt == GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER){
        return RK_FORMAT_BGR_888;
    } 
    return (int)format;
}

BufSurfaceColorFormat convertRKFormat2SurfFormat(int &fmt)
{
    // if (fmt == FORMAT_NV12) {
    //     return GDDEPLOY_BUF_COLOR_FORMAT_NV12;
    // } else if (fmt == FORMAT_NV21) {
    //     return GDDEPLOY_BUF_COLOR_FORMAT_NV21;
    // } else if (fmt == FORMAT_YUV420P) {
    //     return GDDEPLOY_BUF_COLOR_FORMAT_YUV420;
    // } else if (fmt == FORMAT_RGB_PACKED) {
    //     return GDDEPLOY_BUF_COLOR_FORMAT_RGB;
    // } else if (fmt == FORMAT_BGR_PACKED) {
    //     return GDDEPLOY_BUF_COLOR_FORMAT_BGR;
    // } else if (fmt == FORMAT_RGB_PLANAR) {
    //     return GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;
    // } else if (fmt == FORMAT_BGR_PLANAR) {
    //     return GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER;
    // }

    return GDDEPLOY_BUF_COLOR_FORMAT_INVALID;
}