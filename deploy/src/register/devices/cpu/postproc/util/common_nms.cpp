#include "common_nms.h"


namespace gddeploy {

void generate_grids_and_stride(const int target_size, std::vector<int> &strides,
                                      std::vector<GridAndStride> &grid_strides, float offset)
{
    for (auto stride : strides)
    {
        int num_grid = target_size / stride;
        for (float g1 = 0; g1 < num_grid; g1++)
        {
            for (float g0 = 0; g0 < num_grid; g0++)
            {
                grid_strides.emplace_back((GridAndStride){g0+offset, g1+offset, stride});
            }
        }
    }
}

void get_rect(int img_w, int img_h,int model_w, int model_h, gddeploy::Bbox *bbox) {
    int w, h, x, y;
    float r_w = model_w / (img_w * 1.0);
    float r_h = model_h / (img_h * 1.0);

    if (r_h > r_w) 
    {
        bbox->x = bbox->x / r_w;
        bbox->w = bbox->w / r_w;
        bbox->h = bbox->h / r_w;

        h = r_w * img_h;
        y = (model_h - h) / 2;
        bbox->y = (bbox->y - y) / r_w;
    }else{
        bbox->y = bbox->y / r_h;
        bbox->w = bbox->w / r_h;
        bbox->h = bbox->h / r_h;

        w = r_h * img_w;
        x = (model_w - w) / 2;
        bbox->x = (bbox->x - x) / r_h;
    }

    bbox->x = std::max(0.0f, bbox->x);
    bbox->y = std::max(0.0f, bbox->y);

    bbox->w = std::min((float)bbox->x+img_w, bbox->x+bbox->w) - bbox->x;
    bbox->h = std::min((float)bbox->x+img_h, bbox->y+bbox->h) - bbox->y;
}

std::vector<DetectObject> nms(std::vector<DetectObject> objInfos, float conf_thresh)
{
    std::sort(objInfos.begin(), objInfos.end(), [](DetectObject lhs, DetectObject rhs)
              { return lhs.score > rhs.score; });
    if (objInfos.size() > 1000)
    {
        objInfos.erase(objInfos.begin() + 1000, objInfos.end());
    }

    std::vector<DetectObject> result;

    while (objInfos.size() > 0){
        result.push_back(objInfos[0]);
  
        for (auto it = objInfos.begin() + 1; it != objInfos.end();)
        {
            auto box1 = objInfos[0].bbox;
            auto box2 = (*it).bbox;

            float x1 = std::max(box1.x, box2.x);
            float y1 = std::max(box1.y, box2.y);
            float x2 = std::min(box1.x+box1.w, box2.x+box2.w);
            float y2 = std::min(box1.y+box1.h, box2.y+box2.h);
            float over_w = std::max(0.0f, x2 - x1);
            float over_h = std::max(0.0f, y2 - y1);
            float over_area = over_w * over_h;
            float iou_value = over_area / ((box1.w ) * (box1.h ) + (box2.w ) * (box2.h ) - over_area);

            if (iou_value > conf_thresh)
                it = objInfos.erase(it);
            else
                it++; 
        }
        objInfos.erase(objInfos.begin());
    }

    return result;
}


} // namespace gddeploy