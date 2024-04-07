#include "postproc_yolov6.h"

#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"
#include "core/model.h"

#include <math.h>
#include <vector>

#include "opencv2/opencv.hpp"

#include "util/common_nms.h"

using namespace gddeploy;

namespace gddeploy {

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

namespace yolov6{
int decodeNCHWInt8(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                int model_w, int channel, int num_class, float threshold,
                std::vector<int> feature_h_w, int b, DetectImg &detect_img, float *anchors, 
                std::vector<float> output_scales, std::vector<int> output_zp)
{
    std::vector<int> map_ww;
    for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
        int  ww = feature_h_w[a_idx];
        map_ww.emplace_back(ww);
    }
    // 对map_ww从大到小排序
    std::sort(map_ww.begin(), map_ww.end(), [](int lhs, int rhs)
              { return lhs > rhs; });

    std::map<int, int> anchors_map = {{map_ww[0], 0}, {map_ww[1], 1}, {map_ww[2], 2}};

    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    int feature_map_sum = 0;

    generate_grids_and_stride(model_w, strides, grid_strides, 0.5);

    for (size_t f_idx = 0; f_idx < 3; f_idx++){ //循环多少个输出
        BufSurfWrapperPtr surf_ptr = out_data[f_idx];
        int8_t *data_ptr = static_cast<int8_t*>(surf_ptr->GetHostData(0, 0));

        BufSurfWrapperPtr surf_bbox_ptr = out_data[f_idx+3];
        int8_t *data_bbox_ptr = static_cast<int8_t*>(surf_bbox_ptr->GetHostData(0, 0));
        
        int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx];
        int step = hh * ww;
        int ratio = model_w / ww;

        int8_t *group_addr = data_ptr + b * step * channel;

        for(int h = 0; h < hh; h++)
        {
            for(int w = 0; w < ww; w++)
            {
                int feature_map_index = h * ww + w;
                bool is_object = false;
                int max_score_index = -1;
                float max_score = -FLT_MAX;

                for (int c = 0; c < num_class; c++){
                    float class_score = deqnt_affine_to_f32(*(group_addr + step * c + h * ww + w), output_zp[f_idx] , output_scales[f_idx]);
                    if (class_score < threshold){
                        continue;
                    }
                    if (class_score > max_score)
                    {
                        max_score_index = c;
                        max_score = class_score;
                    }
                    is_object = true;
                } 

                if(is_object)
                {
                    const float grid0 = grid_strides[feature_map_sum + feature_map_index].grid0;
                    const float grid1 = grid_strides[feature_map_sum + feature_map_index].grid1;
                    const int stride = grid_strides[feature_map_sum + feature_map_index].stride;

                    // 取出后4个像素值，是bbox的坐标，需要转为x,y,w,h，再乘以stride
                    int8_t *box_src = static_cast<int8_t *>(data_bbox_ptr);
                    int8_t *bbox_pred = box_src + feature_map_index;
                    // int8_t *bbox_pred = src + num_class * step + feature_map_index;
                    float x0 = grid0 - deqnt_affine_to_f32(*(bbox_pred + 0 * step), output_zp[f_idx+3], output_scales[f_idx+3]);
                    float y0 = grid1 - deqnt_affine_to_f32(*(bbox_pred + 1 * step), output_zp[f_idx+3], output_scales[f_idx+3]);
                    float x1 = grid0 + deqnt_affine_to_f32(*(bbox_pred + 2 * step), output_zp[f_idx+3], output_scales[f_idx+3]);
                    float y1 = grid1 + deqnt_affine_to_f32(*(bbox_pred + 3 * step), output_zp[f_idx+3], output_scales[f_idx+3]);

                    float x_center = (x0 + x1) / 2;
                    float y_center = (y0 + y1) / 2;
                    float width = x1 - x0;
                    float height = y1 - y0;

                    x_center = x_center * stride;
                    y_center = y_center * stride;
                    width = width * stride;
                    height = height * stride;

                    DetectObject obj;
                    obj.bbox.x = x_center - width / 2;
                    obj.bbox.y = y_center - height / 2;
                    obj.bbox.w = width;
                    obj.bbox.h = height;  
                    obj.score = max_score;
                    obj.class_id = max_score_index;

                    detect_img.detect_objs.emplace_back(obj);  
                }
            }
        }
        feature_map_sum += step;
    }
    return 0;
}


int decodeNCHWFloat(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                int model_w, int channel, int num_class, float threshold,
                std::vector<int> feature_h_w, int b, DetectImg &detect_img, float *anchors)
{
    std::vector<int> map_ww;
    for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
        int  ww = feature_h_w[a_idx];
        map_ww.emplace_back(ww);
    }
    // 对map_ww从大到小排序
    std::sort(map_ww.begin(), map_ww.end(), [](int lhs, int rhs)
              { return lhs > rhs; });

    std::map<int, int> anchors_map = {{map_ww[0], 0}, {map_ww[1], 1}, {map_ww[2], 2}};

    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(model_w, strides, grid_strides, 0.5);

    for (size_t f_idx = 0; f_idx < 3; f_idx++){ //循环多少个输出
        BufSurfWrapperPtr surf_ptr = out_data[f_idx];
        int8_t *data_ptr = static_cast<int8_t*>(surf_ptr->GetHostData(0, 0));

        BufSurfWrapperPtr surf_bbox_ptr = out_data[f_idx+3];
        int8_t *data_bbox_ptr = static_cast<int8_t*>(surf_bbox_ptr->GetHostData(0, 0));
        
        int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx];
        int step = hh * ww;
        int ratio = model_w / ww;

        int8_t *group_addr = data_ptr + b * step * channel;
        int feature_map_sum = 0;
        
        for(int h = 0; h < hh; h++)
        {
            for(int w = 0; w < ww; w++)
            {
                int feature_map_index = h * ww + w;
                bool is_object = false;
                int max_score_index = -1;
                float max_score = -FLT_MAX;

                for (int c = 0; c < num_class; c++){
                    float class_score = *(group_addr + step * c + h * ww + w);
                    if (class_score < threshold){
                        continue;
                    }
                    if (class_score > max_score)
                    {
                        max_score_index = c;
                        max_score = class_score;
                    }
                    is_object = true;
                } 

                if(is_object)
                {
                    const float grid0 = grid_strides[feature_map_sum + feature_map_index].grid0;
                    const float grid1 = grid_strides[feature_map_sum + feature_map_index].grid1;
                    const int stride = grid_strides[feature_map_sum + feature_map_index].stride;

                    // 取出后4个像素值，是bbox的坐标，需要转为x,y,w,h，再乘以stride
                    int8_t *box_src = static_cast<int8_t *>(data_bbox_ptr);
                    int8_t *bbox_pred = box_src + feature_map_index;
                    // int8_t *bbox_pred = src + num_class * step + feature_map_index;
                    float x0 = grid0 - *(bbox_pred + 0 * step);
                    float y0 = grid1 - *(bbox_pred + 1 * step);
                    float x1 = grid0 + *(bbox_pred + 2 * step);
                    float y1 = grid1 + *(bbox_pred + 3 * step);

                    float x_center = (x0 + x1) / 2;
                    float y_center = (y0 + y1) / 2;
                    float width = x1 - x0;
                    float height = y1 - y0;

                    x_center = x_center * stride;
                    y_center = y_center * stride;
                    width = width * stride;
                    height = height * stride;

                    DetectObject obj;
                    obj.bbox.x = x_center - width / 2;
                    obj.bbox.y = y_center - height / 2;
                    obj.bbox.w = width;
                    obj.bbox.h = height;  
                    obj.score = max_score;
                    obj.class_id = max_score_index;

                    detect_img.detect_objs.emplace_back(obj);  
                }
            }
        }
        feature_map_sum += step;
    }
    return 0;
}

#define TS_MPI_TRP_RNE_MASK_BITS(m) ((1ll << (m)) - 1)
int decodeNHWCInt8(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                int model_w, int channel, int classNum, float threshold,
                std::vector<int> feature_h_w, int b, DetectImg &detect_img, float *anchors, std::vector<float> output_scales)
{
    std::vector<int> map_ww;
    for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
        int  ww = feature_h_w[a_idx];
        map_ww.emplace_back(ww);
    }
    // 对map_ww从大到小排序
    std::sort(map_ww.begin(), map_ww.end(), [](int lhs, int rhs)
              { return lhs > rhs; });

    std::map<int, int> anchors_map = {{map_ww[0], 0}, {map_ww[1], 1}, {map_ww[2], 2}};

    for (int f_idx = 0; f_idx < out_data.size(); f_idx++){ //循环多少个输出
        BufSurfWrapperPtr surf_ptr = out_data[f_idx];
        int8_t *data_ptr = static_cast<int8_t*>(surf_ptr->GetHostData(0, 0));
        
        int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx], cc = channel;
        int step = hh * ww;
        int ratio = model_w / ww;

        for(int h = 0; h < hh; h++)
        {
            for(int w = 0; w < ww; w++)
            {
                for (int a_idx = 0; a_idx < 3; a_idx++) // for anchor num
                {   
#if WITH_TS
                    // cc需要32对齐
                    int c_stride = (cc % 32 == 0) ? cc : ((cc / 32) + 1) * 32;
                    int8_t * base_ptr = data_ptr + h * ww * c_stride + w * c_stride + a_idx * cc / 3;
#else 
                    int8_t * base_ptr = data_ptr + h * ww * cc + w * cc + a_idx * cc / 3;
#endif
                    float conf = *((int *)(base_ptr + 4)) & TS_MPI_TRP_RNE_MASK_BITS(8);
                    conf = conf * output_scales[f_idx];

                    if(conf > threshold)
                    {
                        DetectObject obj;
                        memset(&obj, 0, sizeof(DetectObject));
                        float bbox_x = (*((int *)(base_ptr + 0)) & TS_MPI_TRP_RNE_MASK_BITS(8)) * output_scales[f_idx];
                        obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                        float bbox_y = (*((int *)(base_ptr + 1)) & TS_MPI_TRP_RNE_MASK_BITS(8)) * output_scales[f_idx];
                        obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                        float bbox_w = (*((int *)(base_ptr + 2)) & TS_MPI_TRP_RNE_MASK_BITS(8)) * output_scales[f_idx];
                        float bbox_h = (*((int *)(base_ptr + 3)) & TS_MPI_TRP_RNE_MASK_BITS(8)) * output_scales[f_idx];
                        obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2];
                        obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2+1];

                        obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                        obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                        for(int cls = 0; cls < classNum; cls++)
                        {
                            int8_t* cls_conf_channel = base_ptr + 5 + cls;
                            float cls_conf = (*((int *)cls_conf_channel) & TS_MPI_TRP_RNE_MASK_BITS(8)) * conf * output_scales[f_idx];

                            if(cls_conf > obj.score)
                            {
                                obj.score = cls_conf;
                                obj.class_id = cls;
                            }
                        }
                        // if (obj.score < threshold)
                        //     continue;

                        detect_img.detect_objs.emplace_back(obj);
                    }
                }
            }
        }
    }
    return 0;
}


int decodeNHWCFloat(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                int model_w, int channel, int classNum, float threshold,
                std::vector<int> feature_h_w, int b, DetectImg &detect_img, float *anchors)
{
    std::vector<int> map_ww;
    for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
        int  ww = feature_h_w[a_idx];
        map_ww.emplace_back(ww);
    }
    // 对map_ww从大到小排序
    std::sort(map_ww.begin(), map_ww.end(), [](int lhs, int rhs)
              { return lhs > rhs; });

    std::map<int, int> anchors_map = {{map_ww[0], 0}, {map_ww[1], 1}, {map_ww[2], 2}};


    for (size_t f_idx = 0; f_idx < out_data.size(); f_idx++){ //循环多少个输出
        BufSurfWrapperPtr surf_ptr = out_data[f_idx];
        float *data_ptr = static_cast<float*>(surf_ptr->GetData(0, 0));
        
        int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx], cc = channel;
        int step = hh * ww;
        int ratio = model_w / ww;

        for(int h = 0; h < hh; h++)
        {
            for(int w = 0; w < ww; w++)
            {
                for (int a_idx = 0; a_idx < 3; a_idx++) // for anchor num
                {   
                    float * base_ptr = data_ptr + h * ww * cc + w * cc + a_idx * cc / 3;
                    float conf = *(base_ptr + 4);

                    if(conf > threshold)
                    {
                        DetectObject obj;
                        memset(&obj, 0, sizeof(DetectObject));
                        float bbox_x = *(base_ptr + 0);
                        obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                        float bbox_y = *(base_ptr + 1);
                        obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                        float bbox_w = *(base_ptr + 2);
                        float bbox_h = *(base_ptr + 3);
                        // obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2];
                        // obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2+1];
                        obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2];
                        obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2+1];

                        obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                        obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                        for(int cls = 0; cls < classNum; cls++)
                        {
                            float* cls_conf_channel = base_ptr + 5 + cls;
                            float cls_conf = *cls_conf_channel * conf;

                            if(cls_conf > obj.score)
                            {
                                obj.score = cls_conf;
                                obj.class_id = cls;
                            }
                        }
                        if (obj.score < threshold)
                            continue;

                        detect_img.detect_objs.emplace_back(obj);
                    }
                }
            }
        }
    }
    return 0;
}
} // namespace yolov6

using namespace yolov6;

int Yolov6DecodeOutput(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::PostParam &param,  
                                    std::vector<FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{
    const DataLayout output_layout =  model_ptr->OutputLayout(0);
    auto dtype = output_layout.dtype;
    auto order = output_layout.order;
    int data_size = 0;
    if (dtype == DataType::INT8 || dtype == DataType::UINT8){
        data_size = sizeof(uint8_t);
    }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
        data_size = sizeof(uint16_t);
    }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
        data_size = sizeof(uint32_t);
    }

    float anchors[] = {10,  13, 16,  30,  33,  23, 
                30,  61, 62,  45,  59,  119, 
                116, 90, 156, 198, 373, 326};
    std::vector<std::string> labels = param.labels;
    float threshold = param.conf_thresh;
    float iou_thresh = param.iou_thresh;
    std::vector<float> output_scales = param.output_scale;
    std::vector<int> output_zp = param.output_zp;

    // 1. 获取Model信息
    int min_feature_size = 80; //std::sqrt(feature_size / (16 + 4 + 1));
    int channel = 18; //shape[1];
    int out_num = model_ptr->OutputNum();
    std::vector<int> feature_h_w;

    auto input_shape = model_ptr->InputShape(0);
    int model_w = input_shape[2];
    int model_h = input_shape[3];
    int model_b = input_shape[0];
    
    for (int b = 0; b < frame_info.size(); b++) {
        DetectImg detect_img; 
        detect_img.img_id = frame_info[b].frame_id;

        detect_img.img_w = frame_info[b].width;
        detect_img.img_h = frame_info[b].height; 
        
        
        if (order == DimOrder::NCHW){
            int classNum = 0;
            for (int i = 0; i < 3; i++){
                auto output_shape = model_ptr->OutputShape(i);
                
                int out_w, out_h;
                out_w = output_shape[3];
                out_h = output_shape[2];
                channel = output_shape[1];
                classNum = channel;
                feature_h_w.emplace_back(out_w);
            }
            const DataLayout input_layout =  model_ptr->InputLayout(0);
            if (input_layout.order == DimOrder::NCHW){
                model_w = input_shape[2];
                model_h = input_shape[3];
                model_b = input_shape[0];
            } else {
                model_w = input_shape[2];
                model_h = input_shape[1];
                model_b = input_shape[0];
            }

            if (dtype == DataType::INT8 || dtype == DataType::UINT8){
                decodeNCHWInt8(out_data, model_w, channel, classNum, threshold, feature_h_w, b, detect_img, anchors, output_scales, output_zp);
            }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
                decodeNCHWFloat(out_data, model_w, channel, classNum, threshold, feature_h_w, b, detect_img, anchors);
            }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
                decodeNCHWFloat(out_data, model_w, channel, classNum, threshold, feature_h_w, b, detect_img, anchors);
            }
        } else {
            int classNum = 0;
            for (int i = 0; i < out_num; i++){
                auto output_shape = model_ptr->OutputShape(i);
                
                int out_w, out_h;
                out_w = output_shape[2];
                out_h = output_shape[1];
                channel = output_shape[3];
                classNum = channel/3 - 5;
                feature_h_w.emplace_back(out_w);
            }
            model_w = input_shape[2];
            model_h = input_shape[1];
            model_b = input_shape[0];

            if (dtype == DataType::INT8 || dtype == DataType::UINT8){
                decodeNHWCInt8(out_data, model_w, channel, classNum, threshold, feature_h_w, b, detect_img, anchors, output_scales);
            }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
                decodeNHWCFloat(out_data, model_w, channel, classNum, threshold, feature_h_w, b, detect_img, anchors);
            }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
                decodeNHWCFloat(out_data, model_w, channel, classNum, threshold, feature_h_w, b, detect_img, anchors);
            }
        }

        // nms
        // resize to origin img size
        if (detect_img.detect_objs.size() > 0){
            detect_img.detect_objs = nms(detect_img.detect_objs, 0.45);

            for (auto &obj : detect_img.detect_objs){
                get_rect(frame_info[b].width, frame_info[b].height, model_w, model_h, &obj.bbox);
            }
        }
        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT);
        result.detect_result.detect_imgs.emplace_back(detect_img);
    }
    result.detect_result.batch_size = frame_info.size();
    return 0;
}


}