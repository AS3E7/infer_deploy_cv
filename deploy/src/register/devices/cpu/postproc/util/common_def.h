#pragma once 

#include <vector>
#include <memory>
#include <string>

namespace gddeploy{


struct GridAndStride
{
    float grid0;
    float grid1;
    int stride;
};


typedef struct {
    float iou_thresh;
    float conf_thresh;

    std::vector<float> output_scale;
    std::vector<int> output_zp;

    std::vector<std::string> labels;
}PostParam;


}