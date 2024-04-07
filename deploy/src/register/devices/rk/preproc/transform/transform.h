// #pragma once

// #include "opencv2/opencv.hpp"
// #include <map>
// #include <string>
// #include "core/util/any.h"

// namespace gddeploy {
// namespace transform {

// /*TODO: 
//     输入参数：in_mat/out_mat/param
//     其中param支持：resize（直接resize、等比例resize、等比例居中resize），normalize(mean、std)
// */ 
// // class Transform{
// // public:

// // private:

// // };
// enum ResizeProcessType {
//     RESIZE_PT_DEFAULT = 0,  // 不保持比例，直接resize
//     RESIZE_PT_LEFT_TOP,     // 保持比例resize，并且对齐到左上角
//     RESIZE_PT_CENTER,         // 保持比例resize，并且对齐到中间位置
//     RESIZE_PT_CROP,        // 保持比例resize，短边对齐到目标长宽，超出的部分居中裁剪
// };

// typedef struct {
//     int in_w;
//     int in_h;
//     int out_w;
//     int out_h;
//     ResizeProcessType type;
//     int padding_num;
// }ComposeResizeParam;

// typedef struct {
//     float mean[4];
//     float std[4];
// }ComposeNormalizeParam;

// cv::Mat Normalize(const cv::Mat &mat, float mean, float scale);
// cv::Mat Normalize(const cv::Mat &mat, const float *mean, const float *scale);

// int Normalize(const cv::Mat &in_img, cv::Mat &out_img, const float *mean, const float *std);
// int Normalize(const cv::Mat &in_img, cv::Mat &out_img, ComposeNormalizeParam &param);


// int Resize(const cv::Mat &in_img, cv::Mat &out_img, int in_w, int in_h, int out_w, int out_h, ResizeProcessType type, int padding_num = 114);
// int Resize(const cv::Mat &in_img, cv::Mat &out_img, ComposeResizeParam param);


// int Compose(BufSurfaceParams *in_surf_param, BufSurfaceParams *out_surf_param, std::vector<std::pair<std::string, gddeploy::any>>ops);

// }   //namespace transform
// }   //namespace gddeploy