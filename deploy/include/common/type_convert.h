#pragma once

#include "core/infer_server.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_impl.h"

#include "opencv2/opencv.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
}

cv::Mat avframeToCvmat(const AVFrame * frame);
AVFrame* cvmatToAvframe(cv::Mat* image, AVFrame * frame);

int convertMat2BufSurface(cv::Mat &img, gddeploy::BufSurfWrapperPtr &surf, bool is_copy=true);
int convertBufSurface2Mat(cv::Mat &img, gddeploy::BufSurfWrapperPtr &surf, bool is_copy=true);

int convertAVFrame2BufSurface(AVFrame *in, gddeploy::BufSurfWrapperPtr &surf, bool is_copy=true);


#if WITH_BM1684
int convertBmImage2BufSurface(bm_image &img, gddeploy::BufSurfWrapperPtr &surf, bool is_copy);
int convertBufSurface2BmImage(bm_image &img, gddeploy::BufSurfWrapperPtr &surf);
#endif