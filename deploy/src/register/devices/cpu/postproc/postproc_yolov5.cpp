#include "postproc_yolov5.h"

#include "core/mem/buf_surface.h"

#include <math.h>
#include <vector>

#include "opencv2/opencv.hpp"
#include "util/common_nms.h"

using namespace gddeploy;

// void get_rect(int img_w, int img_h,int model_w, int model_h, gddeploy::Bbox *bbox) {
//     int w, h, x, y;
//     float r_w = model_w / (img_w * 1.0);
//     float r_h = model_h / (img_h * 1.0);

//     if (r_h > r_w) 
//     {
//         bbox->x = bbox->x / r_w;
//         bbox->w = bbox->w / r_w;
//         bbox->h = bbox->h / r_w;

//         h = r_w * img_h;
//         y = (model_h - h) / 2;
//         bbox->y = (bbox->y - y) / r_w;
//     }else{
//         bbox->y = bbox->y / r_h;
//         bbox->w = bbox->w / r_h;
//         bbox->h = bbox->h / r_h;

//         w = r_h * img_w;
//         x = (model_w - w) / 2;
//         bbox->x = (bbox->x - x) / r_h;
//     }

//     bbox->x = std::max(0.0f, bbox->x);
//     bbox->y = std::max(0.0f, bbox->y);

//     bbox->w = std::min((float)bbox->x+img_w, bbox->x+bbox->w) - bbox->x;
//     bbox->h = std::min((float)bbox->x+img_h, bbox->y+bbox->h) - bbox->y;
// }

// std::vector<DetectObject> nms(std::vector<DetectObject> objInfos, float conf_thresh)
// {
//     std::sort(objInfos.begin(), objInfos.end(), [](DetectObject lhs, DetectObject rhs)
//               { return lhs.score > rhs.score; });
//     if (objInfos.size() > 1000)
//     {
//         objInfos.erase(objInfos.begin() + 1000, objInfos.end());
//     }

//     std::vector<DetectObject> result;

//     while (objInfos.size() > 0){
//         result.push_back(objInfos[0]);
  
//         for (auto it = objInfos.begin() + 1; it != objInfos.end();)
//         {
//             auto box1 = objInfos[0].bbox;
//             auto box2 = (*it).bbox;

//             float x1 = std::max(box1.x, box2.x);
//             float y1 = std::max(box1.y, box2.y);
//             float x2 = std::min(box1.x+box1.w, box2.x+box2.w);
//             float y2 = std::min(box1.y+box1.h, box2.y+box2.h);
//             float over_w = std::max(0.0f, x2 - x1);
//             float over_h = std::max(0.0f, y2 - y1);
//             float over_area = over_w * over_h;
//             float iou_value = over_area / ((box1.w ) * (box1.h ) + (box2.w ) * (box2.h ) - over_area);

//             if (iou_value > conf_thresh)
//                 it = objInfos.erase(it);
//             else
//                 it++; 
//         }
//         objInfos.erase(objInfos.begin());
//     }

//     return result;
// }

namespace gddeploy {

double sigmoid(double x){
    return (1 / (1 + exp(-x)));
    // return x;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

int Yolov5DecodeOutputNCHWSigmoid(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    if (out_data.size() == 0 || !out_data[0]->GetHostData(0)) {
        std::cout << "[Postprocess] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    float anchors[] = {10,  13, 16,  30,  33,  23, 
                30,  61, 62,  45,  59,  119, 
                116, 90, 156, 198, 373, 326};
    float threshold = any_cast<float>(param);

    // 1. 获取Model信息
    int min_feature_size = 80; //std::sqrt(feature_size / (16 + 4 + 1));
    int channel = 18; //shape[1];
    int out_num = model_ptr->OutputNum();
    std::vector<int> feature_h_w;

    for (int i = 0; i < out_num; i++){
        auto output_shape = model_ptr->OutputShape(i);
        
        int out_w, out_h;
        out_w = output_shape[3];
        out_h = output_shape[2];
        channel = output_shape[1];
        feature_h_w.emplace_back(out_w);
    }

    auto input_shape = model_ptr->InputShape(0);
    int model_w = input_shape[2];
    int model_h = input_shape[3];
    int model_b = input_shape[0];

    int classNum = channel/3 - 5;

    // 2. 解析网络
    for (size_t b = 0; b < frame_info.size(); b++) {
        DetectImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;

        detect_img.img_w = frame_info[b].width;
        detect_img.img_h = frame_info[b].height;

        for (size_t f_idx = 0; f_idx < out_data.size(); f_idx++){ //循环多少个输出
            BufSurfWrapperPtr surf_ptr = out_data[f_idx];

            float *data_ptr = static_cast<float*>(surf_ptr->GetHostData(0, b));

            for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
                int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx];
                int step = hh * ww;
                int ratio = model_w / ww;

                float *group_addr = data_ptr + a_idx * step * channel / 3;
                
                for(int h = 0; h < hh; h++)
                {
                    for(int w = 0; w < ww; w++)
                    {
                        float conf = sigmoid(*(group_addr + step * 4 + h * ww + w));

                        if(conf > threshold)
                        {
                            DetectObject obj;
                            memset(&obj, 0, sizeof(DetectObject));
                            float bbox_x = sigmoid(*(group_addr + step*0 + h * ww + w));
                            obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                            float bbox_y = sigmoid(*(group_addr + step*1 + h * ww + w));
                            obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                            float bbox_w = sigmoid(*(group_addr + step*2 + h * ww + w));
                            float bbox_h = sigmoid(*(group_addr + step*3 + h * ww + w));
                            obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2];
                            obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2+1];

                            obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                            obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                            for(int cls = 0; cls < classNum; cls++)
                            {
                                float* cls_conf_channel = group_addr + step*(5+cls);
                                float cls_conf = sigmoid(*(cls_conf_channel + h * ww + w)) * conf;

                                if(cls_conf > obj.score)
                                {
                                    obj.score = cls_conf;
                                    obj.class_id = cls;
                                }
                            }

                            detect_img.detect_objs.emplace_back(obj);
                        }
                    }
                }
            }

        }
        // nms
        // resize to origin img size
        if (detect_img.detect_objs.size() > 0){
            detect_img.detect_objs = nms(detect_img.detect_objs, 0.5);

            for (auto &obj : detect_img.detect_objs){
                get_rect(frame_info[b].width, frame_info[b].height, model_w, model_h, &obj.bbox);
            }
        }

        //--------------test begin---------------------------------------------
        #if 0
        cv::Mat frame = cv::imread("/gddeploy/data/pic/helmet2.jpg");
        std::cout << "---------------------------------------------------" << std::endl;

        std::cout << "Detect num: " << detect_img.detect_objs.size() << std::endl;
        for (auto &obj : detect_img.detect_objs) {
            std::cout << "Detect result: " << "box[" << obj.bbox.x \
                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                << obj.bbox.h << "]" \
                << "   score: " << obj.score 
                << "   class id: " << obj.class_id << std::endl;
            cv::Point p1(obj.bbox.x, obj.bbox.y);
            cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
            cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0), 1);
        }

        cv::imwrite("/gddeploy/preds/result_img.jpg", frame);
        #endif
        //--------------test end---------------------------------------------

        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT);
        result.detect_result.detect_imgs.emplace_back(detect_img);
    }
    
    return 0;
}

int decodeNCHWInt8(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                int model_w, int channel, int classNum, float threshold,
                std::vector<int> feature_h_w, int b, DetectImg &detect_img, float *anchors, 
                std::vector<float> output_scales, std::vector<int> output_zps)
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
        int8_t *data_ptr = static_cast<int8_t*>(surf_ptr->GetHostData(0, 0));
        
        for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
            int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx];
            int step = hh * ww;
            int ratio = model_w / ww;

            int8_t *group_addr = data_ptr + b * step * channel + a_idx * step * channel / 3;
            
            for(int h = 0; h < hh; h++)
            {
                for(int w = 0; w < ww; w++)
                {
                    float conf = deqnt_affine_to_f32(*(group_addr + step * 4 + h * ww + w), output_zps[f_idx], output_scales[f_idx]);

                    if(conf > threshold)
                    {
                        DetectObject obj;
                        memset(&obj, 0, sizeof(DetectObject));
                        float bbox_x = deqnt_affine_to_f32(*(group_addr + step*0 + h * ww + w), output_zps[f_idx], output_scales[f_idx]);
                        obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                        float bbox_y = deqnt_affine_to_f32(*(group_addr + step*1 + h * ww + w), output_zps[f_idx], output_scales[f_idx]);
                        obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                        float bbox_w = deqnt_affine_to_f32(*(group_addr + step*2 + h * ww + w), output_zps[f_idx], output_scales[f_idx]);
                        float bbox_h = deqnt_affine_to_f32(*(group_addr + step*3 + h * ww + w), output_zps[f_idx], output_scales[f_idx]);
                        // obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2];
                        // obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2+1];
                        obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2];
                        obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2+1];


                        obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                        obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                        for(int cls = 0; cls < classNum; cls++)
                        {
                            int8_t* cls_conf_channel = group_addr + step*(5+cls);
                            float cls_conf = deqnt_affine_to_f32(*(cls_conf_channel + h * ww + w) * conf, output_zps[f_idx], output_scales[f_idx]);

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


int decodeNCHWFloat(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
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
        
        for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
            int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx];
            int step = hh * ww;
            int ratio = model_w / ww;

            float *group_addr = data_ptr + b * step * channel + a_idx * step * channel / 3;
            
            for(int h = 0; h < hh; h++)
            {
                for(int w = 0; w < ww; w++)
                {
                    float conf = *(group_addr + step * 4 + h * ww + w);

                    if(conf > threshold)
                    {
                        DetectObject obj;
                        memset(&obj, 0, sizeof(DetectObject));
                        float bbox_x = *(group_addr + step*0 + h * ww + w);
                        obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                        float bbox_y = *(group_addr + step*1 + h * ww + w);
                        obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                        float bbox_w = *(group_addr + step*2 + h * ww + w);
                        float bbox_h = *(group_addr + step*3 + h * ww + w);
                        // obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2];
                        // obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2+1];
                        obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2];
                        obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(anchors_map[hh])*6 + a_idx*2+1];

                        obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                        obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                        for(int cls = 0; cls < classNum; cls++)
                        {
                            float* cls_conf_channel = group_addr + step*(5+cls);
                            float cls_conf = *(cls_conf_channel + h * ww + w) * conf;

                            if(cls_conf > obj.score)
                            {
                                obj.score = cls_conf;
                                obj.class_id = cls;
                            }
                        }

                        detect_img.detect_objs.emplace_back(obj);
                    }
                }
            }
        }
    }
    return 0;
}

        // for(TS_S32 n = 0; n < num; ++n) {
        //     for (TS_S32 i = 0; i < h; ++i) {
        //         for (TS_S32 j = 0; j < w; ++j) {
        //             for (TS_S32 k = 0; k < c; ++k) {
        //                 // TS_U8 *d = (TS_U8 *)(out + (((j + i * w) * cStride + k) * uSize));
        //                 TS_S32 *d = (TS_S32 *)(out_int_ptr + (((j + i * w) * cStride + k) * uSize));
        //                 TS_S32 data = *d & TS_MPI_TRP_RNE_MASK_BITS(out_blobs->stpBlob[idx].s32BitNum);

        //                 // *(out_ptr + ((h * w) * k + i * w + j)) = data * fCoeff;
        //                 *(out_float_ptr + n*h*w*c+i*w*c+j*c+k) = data * fCoeff;
        //                 // if (*(out_ptr + ((h * w) * k + i * w + j)) > 1 || *(out_ptr + ((h * w) * k + i * w + j)) < -1){
        //             }
        //         }
        //     }
        // }
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

int Yolov5DecodeOutput(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
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
            for (int i = 0; i < out_num; i++){
                auto output_shape = model_ptr->OutputShape(i);
                
                int out_w, out_h;
                out_w = output_shape[3];
                out_h = output_shape[2];
                channel = output_shape[1];
                classNum = channel/3 - 5;
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

int Yolov5DecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::PostParam &param,  
                                    std::vector<FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    if (out_data.size() == 0 || !out_data[0]->GetHostData(0)) {
        std::cout << "[Postprocess] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    float anchors[] = {10,  13, 16,  30,  33,  23, 
                30,  61, 62,  45,  59,  119, 
                116, 90, 156, 198, 373, 326};
    // float threshold = any_cast<float>(param);
    
    std::vector<std::string> labels = param.labels;
    float threshold = param.conf_thresh;
    float iou_thresh = param.iou_thresh;
    std::vector<float> output_scale = param.output_scale;

    // 1. 获取Model信息
    int min_feature_size = 80; //std::sqrt(feature_size / (16 + 4 + 1));
    int channel = 18; //shape[1];
    int out_num = model_ptr->OutputNum();
    std::vector<int> feature_h_w;

    for (int i = 0; i < out_num; i++){
        auto output_shape = model_ptr->OutputShape(i);
        
        int out_w, out_h;
        out_w = output_shape[3];
        out_h = output_shape[2];
        channel = output_shape[1];
        feature_h_w.emplace_back(out_w);
    }

    auto input_shape = model_ptr->InputShape(0);
    int model_w = input_shape[2];
    int model_h = input_shape[3];
    int model_b = input_shape[0];

    int classNum = channel/3 - 5;

    // 2. 解析网络
    for (size_t b = 0; b < frame_info.size(); b++) {
        DetectImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;

        detect_img.img_w = frame_info[b].width;
        detect_img.img_h = frame_info[b].height;

        for (size_t f_idx = 0; f_idx < out_data.size(); f_idx++){ //循环多少个输出
            BufSurfWrapperPtr surf_ptr = out_data[f_idx];
            float *data_ptr = static_cast<float*>(surf_ptr->GetHostData(0, 0));
            
            for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
                int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx];
                int step = hh * ww;
                int ratio = model_w / ww;

                float *group_addr = data_ptr + b * step * channel + a_idx * step * channel / 3;
                
                for(int h = 0; h < hh; h++)
                {
                    for(int w = 0; w < ww; w++)
                    {
                        float conf = *(group_addr + step * 4 + h * ww + w);

                        if(conf > threshold)
                        {
                            DetectObject obj;
                            memset(&obj, 0, sizeof(DetectObject));
                            float bbox_x = *(group_addr + step*0 + h * ww + w);
                            obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                            float bbox_y = *(group_addr + step*1 + h * ww + w);
                            obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                            float bbox_w = *(group_addr + step*2 + h * ww + w);
                            float bbox_h = *(group_addr + step*3 + h * ww + w);
                            obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2];
                            obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[(f_idx)*6 + a_idx*2+1];

                            obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                            obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                            for(int cls = 0; cls < classNum; cls++)
                            {
                                float* cls_conf_channel = group_addr + step*(5+cls);
                                float cls_conf = *(cls_conf_channel + h * ww + w) * conf;

                                if(cls_conf > obj.score)
                                {
                                    obj.score = cls_conf;
                                    obj.class_id = cls;
                                }
                            }

                            detect_img.detect_objs.emplace_back(obj);
                        }
                    }
                }
            }

        }
        // nms
        // resize to origin img size
        if (detect_img.detect_objs.size() > 0){
            detect_img.detect_objs = nms(detect_img.detect_objs, 0.5);
            printf("nms size: %d, thresh: %f\n", detect_img.detect_objs.size(), threshold);

            for (auto &obj : detect_img.detect_objs){
                get_rect(frame_info[b].width, frame_info[b].height, model_w, model_h, &obj.bbox);
            }
        }

        //--------------test begin---------------------------------------------
        #if 0
        // cv::Mat frame = cv::imread("/gddeploy/data/pic/helmet2.jpg");
        std::cout << "---------------------------------------------------" << std::endl;

        std::cout << "Detect num: " << detect_img.detect_objs.size() << std::endl;
        for (auto &obj : detect_img.detect_objs) {
            std::cout << "Detect result: " << "box[" << obj.bbox.x \
                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                << obj.bbox.h << "]" \
                << "   score: " << obj.score 
                << "   class id: " << obj.class_id << std::endl;
            cv::Point p1(obj.bbox.x, obj.bbox.y);
            cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
            // cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0), 1);
        }

        // cv::imwrite("/gddeploy/preds/result_img.jpg", frame);
        #endif
        //--------------test end---------------------------------------------

        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT);
        result.detect_result.detect_imgs.emplace_back(detect_img);
    }
    result.detect_result.batch_size = frame_info.size();
    
    return 0;
}


int Yolov5DecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    if (out_data.size() == 0 || !out_data[0]->GetHostData(0)) {
        std::cout << "[Postprocess] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    float anchors[3][6] = {{10,  13, 16,  30,  33,  23}, 
                {30,  61, 62,  45,  59,  119}, 
                {116, 90, 156, 198, 373, 326},
                };
    float threshold = any_cast<float>(param);

    // 1. 获取Model信息
    int min_feature_size = 80; //std::sqrt(feature_size / (16 + 4 + 1));
    int channel = 18; //shape[1];
    int out_num = model_ptr->OutputNum();
    std::vector<int> feature_h_w;

    for (int i = 0; i < out_num; i++){
        auto output_shape = model_ptr->OutputShape(i);
        
        int out_w, out_h;
        out_w = output_shape[2];
        out_h = output_shape[1];
        channel = output_shape[3];
        feature_h_w.emplace_back(out_w);
    }

    auto input_shape = model_ptr->InputShape(0);
    int model_w = input_shape[2];
    int model_h = input_shape[1];
    int model_b = input_shape[0];

    int classNum = channel/3 - 5;

    //anchors map
    // std::map<int, int> anchors_map = {{feature_h_w[0], 0}, {feature_h_w[1], 1}, {feature_h_w[2], 2}};
    std::map<int, int> anchors_map = {{80, 0}, {40, 1}, {20, 2}};

    // 2. 解析网络
    for (size_t b = 0; b < model_b; b++) {
        DetectImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;

        detect_img.img_w = frame_info[b].width;
        detect_img.img_h = frame_info[b].height;

        for (size_t f_idx = 0; f_idx < out_data.size(); f_idx++){ //循环多少个输出
            BufSurfWrapperPtr surf_ptr = out_data[f_idx];

            float *data_ptr = static_cast<float*>(surf_ptr->GetHostData(0, b));

            int hh = feature_h_w[f_idx], ww = feature_h_w[f_idx];
            int step = hh * ww;
            int ratio = model_w / ww;
                
            for(int h = 0; h < hh; h++)
            {
                for(int w = 0; w < ww; w++)
                {
                    for (int a_idx = 0; a_idx < 3; a_idx++) {   // for anchor num
                        float *group_addr = data_ptr + b * channel * step + \
                                h * ww * channel + w * channel + a_idx * channel / 3 ;

                        float conf = *(group_addr + 4);

                        if(conf > threshold)
                        {
                            DetectObject obj;
                            memset(&obj, 0, sizeof(DetectObject));
                            float bbox_x = *(group_addr + 0);
                            obj.bbox.x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                            float bbox_y = *(group_addr + 1);
                            obj.bbox.y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                            float bbox_w = *(group_addr + 2);
                            float bbox_h = *(group_addr + 3);
                            obj.bbox.w = pow(bbox_w * 2.0f,2) * anchors[anchors_map[hh]][a_idx*2];
                            obj.bbox.h = pow(bbox_h * 2.0f,2) * anchors[anchors_map[hh]][a_idx*2+1];

                            obj.bbox.x = obj.bbox.x - obj.bbox.w / 2;
                            obj.bbox.y = obj.bbox.y - obj.bbox.h / 2;

                            for(int cls = 0; cls < classNum; cls++)
                            {
                                float cls_conf = *(group_addr + 5 + cls) * conf;

                                if(cls_conf > obj.score)
                                {
                                    obj.score = cls_conf;
                                    obj.class_id = cls;
                                }
                            }

                            detect_img.detect_objs.emplace_back(obj);
                        }
                    }
                }
            }

        }
        // nms
        // resize to origin img size
        if (detect_img.detect_objs.size() > 0){
            detect_img.detect_objs = nms(detect_img.detect_objs, 0.5);

            for (auto &obj : detect_img.detect_objs){
                get_rect(frame_info[b].width, frame_info[b].height, model_w, model_h, &obj.bbox);
            }
        }

        //--------------test begin---------------------------------------------
        #if 0
        cv::Mat frame = cv::imread("/data/data/pic/helmet2.jpg");
        std::cout << "---------------------------------------------------" << std::endl;

        std::cout << "Detect num: " << detect_img.detect_objs.size() << std::endl;
        for (auto &obj : detect_img.detect_objs) {
            std::cout << "Detect result: " << "box[" << obj.bbox.x \
                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                << obj.bbox.h << "]" \
                << "   score: " << obj.score 
                << "   class id: " << obj.class_id << std::endl;
            cv::Point p1(obj.bbox.x, obj.bbox.y);
            cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
            cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0), 1);
        }

        cv::imwrite("/data/preds/result_img.jpg", frame);
        #endif
        //--------------test end---------------------------------------------

        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT);
        result.detect_result.detect_imgs.emplace_back(detect_img);
    }
    result.detect_result.batch_size = frame_info.size();
    
    return 0;
}

}