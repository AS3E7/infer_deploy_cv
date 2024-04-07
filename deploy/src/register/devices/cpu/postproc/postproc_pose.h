#pragma once

#include "core/result_def.h"
#include "core/mem/buf_surface_util.h"
#include "core/model.h"

int PoseDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data,
                         gddeploy::InferResult &result,
                         gddeploy::any param,
                         std::vector<gddeploy::FrameInfo> frame_info,
                         gddeploy::ModelPtr model_ptr);

int PoseDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data,
                         gddeploy::InferResult &result,
                         gddeploy::any param,
                         std::vector<gddeploy::FrameInfo> frame_info,
                         gddeploy::ModelPtr model_ptr);
int PoseDecodeOutput1NCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data,
                          gddeploy::InferResult &result,
                          gddeploy::any param,
                          std::vector<gddeploy::FrameInfo> frame_info,
                          gddeploy::ModelPtr model_ptr);
