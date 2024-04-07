#include <algorithm>
#include <cmath>
#include <chrono>
#include <omp.h>

#include "opencv2/opencv.hpp"

#include "action.h"
#include "sample_frame.h"

using namespace gddeploy::action;

inline float fast_exp_256(float x) {
    x = 1.0 + x / 256.0;
    x *= x; x *= x; x *= x; x *= x;
    x *= x; x *= x; x *= x; x *= x;
    return x;
}

inline float fast_exp_1024(double x) { 
    x = 1.0 + x / 1024;   
    x *= x; x *= x; x *= x; x *= x;   
    x *= x; x *= x; x *= x; x *= x;   
    x *= x; x *= x;   
    return x; 
}


inline float expf_sum(float* score,int len)
{
    float sum=0;
    // float32x4_t sum_vec=vdupq_n_f32(0);
    // float32x4_t ai=vdupq_n_f32(1064807160.56887296), bi;
    // int32x4_t   int_vec;
    // int value;
    // for(int i=0;i<len;i+=4)
    // {
    //     bi=vld1q_f32(score+4*i);
    //     sum_vec=vmlaq_n_f32(ai,bi,12102203.1616540672);
    //     int_vec=vcvtq_s32_f32(sum_vec);
 
    //     value=vgetq_lane_s32(int_vec,0);
    //     sum+=(*(float*)(&value));
    //     value=vgetq_lane_s32(int_vec,1);
    //     sum+=(*(float*)(&value));
    //     value=vgetq_lane_s32(int_vec,2);
    //     sum+=(*(float*)(&value));
    //     value=vgetq_lane_s32(int_vec,3);
    //     sum+=(*(float*)(&value));
    // }
 
    return sum;
}

float SumSquareError_NEON2(const float* src_a, const float* src_b, int count)
{
    float sse;
    // asm volatile (
    //     // Clear q8, q9, q10, q11
    //     "veor    q8, q8, q8                  \n"
    //     "veor    q9, q9, q9                  \n"
    //     "veor    q10, q10, q10               \n"
    //     "veor    q11, q11, q11               \n"
    //     "1:                                    \n"

    //     "vld1.32     {q0, q1}, [%[src_a]]!   \n"
    //     "vld1.32     {q2, q3}, [%[src_a]]!   \n"

    //     "vld1.32     {q12, q13}, [%[src_b]]! \n"
    //     "vld1.32     {q14, q15}, [%[src_b]]! \n"

    //     "subs       %[count], %[count], #16  \n"

    //     "vsub.f32   q0, q0, q12              \n"
    //     "vsub.f32   q1, q1, q13              \n"
    //     "vsub.f32   q2, q2, q14              \n"
    //     "vsub.f32   q3, q3, q15              \n"
    //     "vmla.f32   q8, q0, q0               \n"
    //     "vmla.f32   q9, q1, q1               \n"
    //     "vmla.f32   q10, q2, q2              \n"
    //     "vmla.f32   q11, q3, q3              \n"
    //     "bgt        1b                       \n"

    //     "vadd.f32   q8, q8, q9               \n"
    //     "vadd.f32   q10, q10, q11            \n"
    //     "vadd.f32   q11, q8, q10             \n"
    //     "vpadd.f32  d2, d22, d23             \n"
    //     "vpadd.f32  d0, d2, d2               \n"
    //     "vmov.32    %3, d0[0]                \n"
    //     : "+r"(src_a),
    //         "+r"(src_b),
    //         "+r"(count),
    //         "=r"(sse)
    //     :
    //     : "memory", "cc", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13","q14", "q15");
    return sse;
}

// 画图
int ActionPreProc::DrawPoint(std::string save_path, std::string prefix, OutputData out)
{
    int model_h = out.shape[3];
    int model_w = out.shape[2];
    int batch = out.shape[1];
    for (int b = 0; b < batch; b++){
        cv::Mat img(model_h, model_w, CV_8UC1, cv::Scalar(0, 0, 0));
        // 一般的output shape是[1, 5, 64, 64]，会输出5张图

        for (int i = 0; i < model_h; i++){
            for (int j = 0; j < model_w; j++){
                img.data[i*model_w+j] = (int)255 * out.gaussian_blur_data[b*model_h*model_w+i*model_w+j];
                // ACTION_LOG_ERROR("%f ", out.gaussian_blur_data[b*model_h*model_w+i*model_w+j])
            }
            // ACTION_LOG_ERROR("\n")
        }
        // ACTION_LOG_ERROR("\n")

        std::string save_file_path = save_path + prefix + std::to_string(b) + ".jpg";
        cv::imwrite(save_file_path, img);
    }
    return 0;
}


static void PrintPosePoint(DetectPoseData detect_pose)
{
    int tmp = 0;
    ACTION_LOG_DEBUG("bbox: [%f, %f, %f, %f]\n", detect_pose.m_bbox[0], detect_pose.m_bbox[1], detect_pose.m_bbox[2], detect_pose.m_bbox[3]);
    ACTION_LOG_DEBUG("point:\n");
    for(auto point : detect_pose.point){
        ACTION_LOG_DEBUG("\t[%d, %d],", point.x, point.y);
        if (tmp++ % 9 == 8) ACTION_LOG_DEBUG("\n");
    }
    ACTION_LOG_DEBUG("\n");
}

static void PrintPosePoint(std::vector<DetectPoseData> detect_pose)
{
    for (auto pose_point : detect_pose){
        PrintPosePoint(pose_point);
    }
}

// 求17个关键点的中心坐标
int FindCenterPoint(DetectPoseData &detect_pose, gddeploy::action::PoseKeyPoint &center_point)
{
    int sum_x = 0, sum_y = 0, sum_num = 0;
    for (auto point : detect_pose.point){
        sum_x += point.x;
        sum_y += point.y;
        sum_num++;
    }

    center_point.x = (sum_x / sum_num) / 2;
    center_point.y = (sum_y / sum_num) / 2;

    return 0;
}

// 找到所有框中xy的最大最小，得到max_w/h的ROI区域可以包住所有的框，相对于PoseCompct功能
int FindMaxRoi(std::vector<DetectPoseData> input, Rect &roi_rect)
{
    // 根据bbox计算最大最小
    auto min_bbox_x_iter = std::min_element(input.begin(), input.end(), [](DetectPoseData a, DetectPoseData b){
        return (a.m_bbox[0] < b.m_bbox[0]);
    });
    int min_bbox_x = (*min_bbox_x_iter).m_bbox[0];
    
    auto min_bbox_y_iter = std::min_element(input.begin(), input.end(), [](DetectPoseData a, DetectPoseData b){
        return (a.m_bbox[1] < b.m_bbox[1]);
    });
    int min_bbox_y = (*min_bbox_y_iter).m_bbox[1];

    auto max_bbox_x_iter = std::max_element(input.begin(), input.end(), [](DetectPoseData a, DetectPoseData b){
        return (a.m_bbox[0] + a.m_bbox[2] > b.m_bbox[0] + b.m_bbox[2]);
    });
    int max_bbox_x = (*max_bbox_x_iter).m_bbox[0] + (*max_bbox_x_iter).m_bbox[2];

    auto max_bbox_y_iter = std::max_element(input.begin(), input.end(), [](DetectPoseData a, DetectPoseData b){
        return (a.m_bbox[1] + a.m_bbox[3] > b.m_bbox[1] + b.m_bbox[3]);
    });
    int max_bbox_y = (*max_bbox_y_iter).m_bbox[1] + (*max_bbox_y_iter).m_bbox[3];


    //根据point计算最大最小
    std::vector<gddeploy::action::PoseKeyPoint> all_point;
    for (auto pose_point : input){
        for (auto point : pose_point.point){
            all_point.emplace_back(point);
        }
    }

    auto min_x_iter = std::min_element(all_point.begin(), all_point.end(), [](gddeploy::action::PoseKeyPoint a, gddeploy::action::PoseKeyPoint b){
        return (a.x < b.x);
    });
    int min_point_x = (*min_x_iter).x;
    
    auto min_y_iter = std::min_element(all_point.begin(), all_point.end(), [](gddeploy::action::PoseKeyPoint a, gddeploy::action::PoseKeyPoint b){
        return (a.y < b.y);
    });
    int min_point_y = (*min_y_iter).y;

    auto max_x_iter = std::max_element(all_point.begin(), all_point.end(), [](gddeploy::action::PoseKeyPoint a, gddeploy::action::PoseKeyPoint b){
        return (a.x < b.x);
    });
    int max_point_x = (*max_x_iter).x;

    auto max_y_iter = std::max_element(all_point.begin(), all_point.end(), [](gddeploy::action::PoseKeyPoint a, gddeploy::action::PoseKeyPoint b){
        return (a.y < b.y);
    });
    int max_point_y = (*max_y_iter).y;


    // roi_rect.start_x = std::min(min_point_x, min_bbox_x);
    // roi_rect.start_y = std::min(min_point_y, min_bbox_y);
    // roi_rect.crop_w = std::max((max_point_x - min_point_x), (max_bbox_x - min_bbox_x));
    // roi_rect.crop_h = std::max((max_point_y - min_point_y), (max_bbox_y - min_bbox_y));
    roi_rect.start_x = min_point_x;
    roi_rect.start_y = min_point_y;
    roi_rect.crop_w = max_point_x - min_point_x;
    roi_rect.crop_h = max_point_y - min_point_y;

    return 0;
}

// 放锁图片
int Resize(Rect in_rect, Rect out_rect,
    std::vector<DetectPoseData> &in_points, std::vector<DetectPoseData> &out_points)
{
    float ratio_w = (float) out_rect.crop_w / in_rect.crop_w;
    float ratio_h = (float) out_rect.crop_h / in_rect.crop_h;

    Rect dst_rect;
    if (ratio_h > ratio_w){
        dst_rect.crop_w = out_rect.crop_w;
        dst_rect.crop_h= in_rect.crop_h * ratio_w;
        dst_rect.start_y = (out_rect.crop_h - dst_rect.crop_h) / 2;
        dst_rect.start_x = 0;

        for (auto pose_point : in_points){
            DetectPoseData dst_pose_point;
            auto points = pose_point.point;
            for (auto point : points){
                gddeploy::action::PoseKeyPoint dst_point;
                dst_point.x = point.x * ratio_w + dst_rect.start_x;
                dst_point.y = point.y * ratio_w + dst_rect.start_y;
                dst_point.score = point.score;
                dst_pose_point.point.emplace_back(dst_point);
            }

            dst_pose_point.m_fProb = pose_point.m_fProb;
            dst_pose_point.m_iClassId = pose_point.m_iClassId;
            dst_pose_point.m_bbox[0] = pose_point.m_bbox[0] * ratio_w + dst_rect.start_x;
            dst_pose_point.m_bbox[1] = pose_point.m_bbox[1] * ratio_w + dst_rect.start_y;
            dst_pose_point.m_bbox[2] = pose_point.m_bbox[2] * ratio_w;
            dst_pose_point.m_bbox[3] = pose_point.m_bbox[3] * ratio_w;
            out_points.emplace_back(dst_pose_point);
        }
    }else{
        dst_rect.crop_w = in_rect.crop_w * ratio_h;
        dst_rect.crop_h = out_rect.crop_h;
        dst_rect.start_y = 0;
        dst_rect.start_x = (out_rect.crop_w - dst_rect.crop_w) / 2;

        for (auto pose_point : in_points){
            DetectPoseData dst_pose_point;
            auto points = pose_point.point;
            for (auto point : points){
                gddeploy::action::PoseKeyPoint dst_point;
                dst_point.x = point.x * ratio_h + dst_rect.start_x;
                dst_point.y = point.y * ratio_h + dst_rect.start_y;
                dst_point.score = point.score;
                dst_pose_point.point.emplace_back(dst_point);
            }

            dst_pose_point.m_fProb = pose_point.m_fProb;
            dst_pose_point.m_iClassId = pose_point.m_iClassId;
            dst_pose_point.m_bbox[0] = pose_point.m_bbox[0] * ratio_h + dst_rect.start_x;
            dst_pose_point.m_bbox[1] = pose_point.m_bbox[1] * ratio_h + dst_rect.start_y;
            dst_pose_point.m_bbox[2] = pose_point.m_bbox[2] * ratio_h;
            dst_pose_point.m_bbox[3] = pose_point.m_bbox[3] * ratio_h;
            out_points.emplace_back(dst_pose_point);
        }
    }

    

    return 0;
}

int ActionPreProc::Init(ActionParam param)
{
    param_ = param;
    return 0;
}

int ActionPreProc::Process(InputData &in, OutputData &out)
{
    // auto action_preproc_t0 = std::chrono::high_resolution_clock::now();
    input_.emplace_back(in.pose_data);
    int img_w = in.img_w;
    int img_h = in.img_h;
    
    if (input_.size() < param_.num_frames)
        return -1;
    // input_.pop_front();

    // 1. 抽样，抽取5针作为处理针
    ACTION_LOG_INFO("1. sample frame\n");
    // auto sample_frame_t0 = std::chrono::high_resolution_clock::now();

    sample_frame_ = UniformSampleFrames(param_.clip_len, param_.num_clips, param_.test_mode, param_.seed);
    std::vector<int> indices = sample_frame_.sample_frames(param_.num_frames);

    // 抽取的帧数据
    std::vector<DetectPoseData> sample_frame;
    for (auto index : indices){
        auto iter = input_.begin();
        std::advance(iter, index);
        sample_frame.emplace_back(*iter);
    }

    PrintPosePoint(sample_frame);

    // auto sample_frame_t1 = std::chrono::high_resolution_clock::now();
    // ACTION_LOG_INFO("Sample frame time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(sample_frame_t1 - sample_frame_t0).count());

    // 2. 查找所有目标任务关键点的中心位置
    ACTION_LOG_INFO("2. find center point\n");
    // auto find_center_point_t0 = std::chrono::high_resolution_clock::now();

    std::vector<PoseKeyPoint> center_points;
    for (auto pose_point : sample_frame){
        PoseKeyPoint center_point;
        FindCenterPoint(pose_point, center_point);
        center_points.emplace_back(center_point);
    }

    // auto find_center_point_t1 = std::chrono::high_resolution_clock::now();
    // ACTION_LOG_INFO("Find center point time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(find_center_point_t1 - find_center_point_t0).count());



    // 3. 所有帧的所有点，该帧中心点相对于原图中心点的偏移幅度，并且改为相对于图片原点的位置
    ACTION_LOG_INFO("3. move to roi\n");
    // auto move_roi_t0 = std::chrono::high_resolution_clock::now();

    int img_center_x = img_w / 2;
    int img_center_y = img_h / 2;
    std::vector<DetectPoseData> relative_pose_points;
    for (auto pose_point_iter = sample_frame.begin(); pose_point_iter != sample_frame.end(); pose_point_iter++){
        DetectPoseData relative_points = *pose_point_iter;
        relative_points.point.clear();

        int index = std::distance(sample_frame.begin(), pose_point_iter);
        PoseKeyPoint center_point = center_points[index];
        int center_distance_x = abs(center_point.x - img_center_x);    //获取模板中心点和图片中心点的距离
        int center_distance_y = abs(center_point.y - img_center_y);

        for (auto point : (*pose_point_iter).point){
            point.x = (center_point.x > img_center_x) ? (point.x - center_distance_x) : (point.x + center_distance_x);
            point.y = (center_point.y > img_center_y) ? (point.y - center_distance_y) : (point.y + center_distance_y);

            relative_points.point.emplace_back(point);
        }

        relative_points.m_bbox[0] = (center_point.x > img_center_x) ? (relative_points.m_bbox[0] - center_distance_x) : (relative_points.m_bbox[0] + center_distance_x);
        relative_points.m_bbox[1] = (center_point.y > img_center_y) ? (relative_points.m_bbox[1] - center_distance_y) : (relative_points.m_bbox[1] + center_distance_y);

        relative_pose_points.emplace_back(relative_points);
        PrintPosePoint(relative_points);
    }
    PrintPosePoint(relative_pose_points);
    // auto move_roi_t1 = std::chrono::high_resolution_clock::now();
    // ACTION_LOG_INFO("Move roi time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(move_roi_t1 - move_roi_t0).count());


    relative_pose_points = sample_frame;
    // 4. 查找多帧数据并集后的边界宽高
    ACTION_LOG_INFO("4. calation max rect include all obj\n");
    // auto calc_max_rect_t0 = std::chrono::high_resolution_clock::now();
    Rect max_roi;
    FindMaxRoi(relative_pose_points, max_roi);
    // int roi_center_x = max_roi.start_x + max_roi.crop_w / 2;
    // int roi_center_y = max_roi.start_y + max_roi.crop_h / 2;
    // int roi_center_relative_x = roi_center_x - max_roi.start_x;
    // int roi_center_relative_y = roi_center_y - max_roi.start_y;

    // 5. 偏移最大框
    for (auto &pose_point : relative_pose_points){
        for (auto &point : pose_point.point){
            point.x = point.x - max_roi.start_x;
            point.y = point.y - max_roi.start_y;
        }
        pose_point.m_bbox[0] = pose_point.m_bbox[0] - max_roi.start_x;
        pose_point.m_bbox[1] = pose_point.m_bbox[1] - max_roi.start_y;
    }
    PrintPosePoint(relative_pose_points);
    // auto calc_max_rect_t1 = std::chrono::high_resolution_clock::now();
    // ACTION_LOG_INFO("Move roi time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(calc_max_rect_t1 - calc_max_rect_t0).count());



    // 6. 把多帧数据抠图并且resize到64x64, 计算resize 64x64后的关键点的坐标偏移
    ACTION_LOG_INFO("5. resize to max rect\n");
    // auto resize_t0 = std::chrono::high_resolution_clock::now();

    std::vector<DetectPoseData> resize_pose_points;
    Rect in_rect = {0, 0, max_roi.crop_w, max_roi.crop_h};
    Rect out_rect = {0, 0, param_.model_w, param_.model_h};

    Resize(in_rect, out_rect, relative_pose_points, resize_pose_points);
    
    PrintPosePoint(resize_pose_points);

    // auto resize_t1 = std::chrono::high_resolution_clock::now();
    // ACTION_LOG_INFO("Resize time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(resize_t1 - resize_t0).count());



    // 7. 高斯分布并且赋值到64x64区域内Gaussian Blur）
    ACTION_LOG_INFO("6. gaussian blur\n");
    // auto gaussian_t0 = std::chrono::high_resolution_clock::now();
    float gaussian_const_value = 2 * std::pow(param_.sigma, 2);
    out.gaussian_blur_data.resize(param_.clip_len * param_.model_w * param_.model_h);
    
    uint32_t batch = 0;
    for (auto pose_point : resize_pose_points){
        for (auto point : pose_point.point){
            int st_x = (int)(point.x - 3 * param_.sigma);
            int ed_x = (int)(point.x + 3 * param_.sigma) + 1;
            int st_y = (int)(point.y - 3 * param_.sigma);
            int ed_y = (int)(point.y + 3 * param_.sigma) + 1;

            // if the keypoint not in the heatmap coordinate system, then skip it.
            if (st_x >= ed_x || st_y >= ed_y){
                continue;
            }
           
            // #pragma omp parallel for 
            for (int x = st_x; x < ed_x; x++)
            {
                if (x > param_.model_w - 1|| x < 0)
                    continue;
                // #pragma omp parallel for 
                for (int y = st_y; y < ed_y; y++)
                {                    
                    if (y > param_.model_h  - 1|| y < 0)
                        continue;

                    int x_tmp = std::pow(x - point.x, 2);
                    int y_tmp = std::pow(y - point.y, 2);

                    // float value = fast_exp_256(-(x_tmp+y_tmp)/gaussian_const_value);
                    // float tmp = -(x_tmp+y_tmp)/gaussian_const_value;
                    // float value = expf_sum(&tmp, 1);

                    float value = std::exp(-(x_tmp+y_tmp)/gaussian_const_value);
                    // float value = std::exp(-(std::pow(x - point.x, 2) + std::pow(y - point.y, 2)) / gaussian_const_value);
                    // if (value >=1)
                        // printf("gaussian_blur_data: b, x, y: [%d, %d, %d], %f\n", batch, x, y, value);
                    out.gaussian_blur_data[batch * param_.model_w * param_.model_h + y * param_.model_w + x] = value;
                    
                }
            }
        }
        batch++;
    }

    // auto gaussian_t1 = std::chrono::high_resolution_clock::now();
    // ACTION_LOG_INFO("Gaussian blur time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(gaussian_t1 - gaussian_t0).count());

    // 填充outData
    out.img_h = in.img_h;
    out.img_w = in.img_w;
    out.img_id = in.img_id;
    out.trace_id = in.trace_id;
    out.shape = {1, param_.clip_len, param_.model_w, param_.model_h};

    // auto action_preproc_t1 = std::chrono::high_resolution_clock::now();
    // ACTION_LOG_INFO("Action preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(action_preproc_t1 - action_preproc_t0).count());

    return 0;

}