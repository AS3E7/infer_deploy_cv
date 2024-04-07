#pragma once

#include "core/result_def.h"
#include "core/mem/buf_surface_util.h"
#include "opencv2/opencv.hpp"

int PrintResult(gddeploy::InferResult &result);

int DrawBbox(gddeploy::InferResult &result, std::vector<std::string> &pic_path, 
            std::vector<cv::Mat> &in_mats, 
            std::vector<gddeploy::BufSurfWrapperPtr> &surfs, std::string &save_path);