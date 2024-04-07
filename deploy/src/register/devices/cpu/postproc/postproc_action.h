#pragma once

#include "core/result_def.h"
#include "core/mem/buf_surface_util.h"
#include "core/model.h"

int ActionDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr);

int ActionDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr);