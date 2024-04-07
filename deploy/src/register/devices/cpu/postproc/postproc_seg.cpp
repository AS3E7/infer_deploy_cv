#include "postproc_seg.h"

#include <math.h>

#include "opencv2/opencv.hpp"

using namespace gddeploy;

typedef struct {
    int x;
    int y;
    int w;
    int h;
}Roi_t;


int SegDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::PostParam &param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    if (!out_data[0]->GetHostData(0)) {
        std::cout << "[gddeploy Samples] [DetectionRunner] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    // 1. 获取Model信息
    auto input_shape = model_ptr->InputShape(0);
    int batch_size  = input_shape[0];
    int model_w     = input_shape[3];
    int model_h     = input_shape[2];
    int model_c     = input_shape[1];

    auto output_shape = model_ptr->OutputShape(0);
    int class_num = output_shape[1];

    for (size_t b = 0; b < batch_size; b++) {
        int img_w = frame_info[b].width;
        int img_h = frame_info[b].height;

        DetectPoseImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;
        detect_img.img_w = img_w;
        detect_img.img_h = img_h;

        BufSurfWrapperPtr surf_ptr = out_data[0];
        float *data_ptr = static_cast<float*>(surf_ptr->GetHostData(0, b));

        //计算图形的偏移
        Roi_t dst_roi;
        memset(&dst_roi, 0, sizeof(Roi_t));
        float r_w = model_w / (img_w*1.0);
        float r_h = model_h / (img_h*1.0);
        if (r_h > r_w) {
            dst_roi.w = model_w;
            // dst_roi.x = pad_lenth/2;
            dst_roi.x = 0;
            dst_roi.y = 0;
            dst_roi.h = img_h * r_w;
        } else {
            dst_roi.h = model_h;
            // dst_roi.y = pad_lenth/2;
            dst_roi.y = 0;
            dst_roi.x = 0;
            dst_roi.w = img_w * r_h;
        }

        std::vector<SegImg> out_result;
        out_result.resize(1);

        SegImg & seg_result = out_result[0];
        // auto seg_map_ptr_tmp = std::shared_ptr<uint8_t>(new uint8_t[dst_roi.w * dst_roi.h]);
        // uint8_t *seg_map_ptr = seg_map_ptr_tmp.get();
        uint8_t seg_map_ptr[dst_roi.w * dst_roi.h];
        int step = model_w * model_h;

        // 3. 逐像素获取类别
        for (int h = 0; h < model_h; h++){
            if (h < dst_roi.y || h > dst_roi.y + dst_roi.h)
                continue;
            for (int w = 0; w < model_w; w++){
                if (w < dst_roi.x || w > dst_roi.x + dst_roi.w)
                    continue;

                float max_score = -1;
                for (int c = 0; c < class_num; c++){
                    float data = *(data_ptr + b*model_c*step + h*model_c*model_w + model_c*w + c);
                    if (data > max_score){
                        max_score = data;
                        
                        seg_map_ptr[(h-dst_roi.y)*dst_roi.w + (w-dst_roi.x)] = c;
                    }
                }
            }
        }

        cv::Mat seg_src_mat(dst_roi.h, dst_roi.w, CV_8UC1, seg_map_ptr);
        
        // 4. resize回原图
        seg_result.map_h = img_h;
        seg_result.map_w = img_w;
        seg_result.seg_map.resize(img_h * img_w);
        cv::Mat seg_dst_mat(img_h, img_w, CV_8UC1, seg_result.seg_map.data());
        cv::resize(seg_src_mat, seg_dst_mat, seg_dst_mat.size(), 0, 0, cv::INTER_LINEAR);

        #if 0
        cv::Mat thresh_mat;
        cv::threshold(seg_src_mat, thresh_mat, 0, 255, cv::THRESH_BINARY);
        cv::imwrite("/data/preds/seg_mask.jpg", thresh_mat);
        #endif

        result.result_type.emplace_back(GDD_RESULT_TYPE_SEG);
        result.seg_result.batch_size = 1;
        result.seg_result.seg_imgs = out_result;
    }

    return 0;
}


int SegDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::PostParam &param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    if (!out_data[0]->GetHostData(0)) {
        std::cout << "[gddeploy Samples] [DetectionRunner] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    // 1. 获取Model信息
    auto input_shape = model_ptr->InputShape(0);
    int batch_size  = input_shape[0];
    int model_w     = input_shape[3];
    int model_h     = input_shape[2];
    int model_c     = input_shape[1];

    auto output_shape = model_ptr->OutputShape(0);
    int class_num = output_shape[1];

    for (size_t b = 0; b < frame_info.size(); b++) {
        int img_w = frame_info[b].width;
        int img_h = frame_info[b].height;

        DetectPoseImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;
        detect_img.img_w = img_w;
        detect_img.img_h = img_h;

        BufSurfWrapperPtr surf_ptr = out_data[0];
        float *data_ptr = static_cast<float*>(surf_ptr->GetHostData(0, b));

        //计算图形的偏移
        Roi_t dst_roi;
        memset(&dst_roi, 0, sizeof(Roi_t));
        float r_w = model_w / (img_w*1.0);
        float r_h = model_h / (img_h*1.0);
        if (r_h > r_w) {
            dst_roi.w = model_w;
            // dst_roi.x = pad_lenth/2;
            dst_roi.x = 0;
            dst_roi.y = 0;
            dst_roi.h = img_h * r_w;
        } else {
            dst_roi.h = model_h;
            // dst_roi.y = pad_lenth/2;
            dst_roi.y = 0;
            dst_roi.x = 0;
            dst_roi.w = img_w * r_h;
        }

        std::vector<SegImg> out_result;
        out_result.resize(1);

        SegImg & seg_result = out_result[0];
        // auto seg_map_ptr_tmp = std::shared_ptr<uint8_t>(new uint8_t[dst_roi.w * dst_roi.h]);
        // uint8_t *seg_map_ptr = seg_map_ptr_tmp.get();
        uint8_t seg_map_ptr[dst_roi.w * dst_roi.h];
        int step = model_w * model_h;

        // 3. 逐像素获取类别
        for (int h = 0; h < model_h; h++){
            if (h < dst_roi.y || h > dst_roi.y + dst_roi.h)
                continue;
            for (int w = 0; w < model_w; w++){
                if (w < dst_roi.x || w > dst_roi.x + dst_roi.w)
                    continue;

                float max_score = -1;
                for (int c = 0; c < class_num; c++){
                    float data = *(data_ptr + b*class_num*step + c*step + h*model_w + w);
                    if (data > max_score){
                        max_score = data;
                        
                        seg_map_ptr[(h-dst_roi.y)*dst_roi.w + (w-dst_roi.x)] = c;
                    }
                }
            }
        }

        cv::Mat seg_src_mat(dst_roi.h, dst_roi.w, CV_8UC1, seg_map_ptr);
        
        // 4. resize回原图
        seg_result.map_h = img_h;
        seg_result.map_w = img_w;
        seg_result.seg_map.resize(img_h * img_w);
        cv::Mat seg_dst_mat(img_h, img_w, CV_8UC1, seg_result.seg_map.data());
        cv::resize(seg_src_mat, seg_dst_mat, seg_dst_mat.size(), 0, 0, cv::INTER_LINEAR);

        #if 0
        cv::Mat thresh_mat;
        cv::threshold(seg_src_mat, thresh_mat, 0, 255, cv::THRESH_BINARY);
        cv::imwrite("./seg_mask.jpg", thresh_mat);
        #endif

        result.result_type.emplace_back(GDD_RESULT_TYPE_SEG);
        result.seg_result.batch_size = 1;
        result.seg_result.seg_imgs = out_result;
    }

    return 0;
}
