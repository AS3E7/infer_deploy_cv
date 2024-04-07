#pragma once

#include "preproc_opencv.h"
#include "opencv2/core.hpp"

#include "transform/transform.h"
#include <opencv2/imgproc.hpp>
#include "core/result_def.h"

namespace gddeploy {


int PreprocYolov5(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocYolox(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocClassify(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocYoloxNHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocRTMPoseNHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w, InferResult &results);
int PreprocClassifyNHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocSeg(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocAction(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocImageRetrieval(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocFaceRetrieval(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocOcrDet(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocOcrRec(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);

int PreprocYolov5Ts(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocYolov5NHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);
int PreprocYolov5Intel(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w);

}