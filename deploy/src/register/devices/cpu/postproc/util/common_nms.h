#pragma once
#include <vector>
#include "common_def.h"
#include "core/result_def.h"

namespace gddeploy {


void generate_grids_and_stride(const int target_size, std::vector<int> &strides,
                                      std::vector<GridAndStride> &grid_strides, float offset = 0.0f);
void get_rect(int img_w, int img_h,int model_w, int model_h, gddeploy::Bbox *bbox);

std::vector<DetectObject> nms(std::vector<DetectObject> objInfos, float conf_thresh);

} // namespace gddeploy