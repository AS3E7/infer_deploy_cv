#pragma once

#include <iostream>
#include <vector>
#include <list>
#include "sample_frame.h"

namespace gddeploy{
namespace action{

typedef struct Rect {
    int start_x;
    int start_y;
    int crop_w;
    int crop_h;
} Rect;

typedef struct {
    int x;
    int y;
    float score;
} PoseKeyPoint;

typedef struct 
{
    int m_iClassId;		// 类别
    float m_fProb;      // 概率
    float m_bbox[4];    //框框 m_bbox[0]: x0 m_bbox[1]: y0 m_bbox[2]: width m_bbox[3]:height
    std::vector<PoseKeyPoint> point;    //关键点
}DetectPoseData;

typedef struct{
    int img_w;
    int img_h;
    int trace_id;
    int img_id;
    DetectPoseData pose_data;
}InputData;

typedef struct{    
    int img_w;
    int img_h;
    int trace_id;
    int img_id;

    std::vector<float> gaussian_blur_data;
    std::vector<int> shape;
}OutputData;

typedef struct{
    int clip_len = 5;
    // 输出的片段个数，默认1帧
    int num_clips = 1;
    int seed = 0;
    int num_frames = 20;
    bool test_mode = true;
    float sigma = 0.6;

    int model_h = 56;
    int model_w = 56;
}ActionParam;

class ActionPreProc{
public:
    ActionPreProc() = default;
    int Init(ActionParam param);

    int Process(InputData &in, OutputData &out);

    int DrawPoint(std::string save_path, std::string prefix, OutputData out);

    void Clear(){
        input_.clear();
    }

private:
    std::list<DetectPoseData> input_;
    UniformSampleFrames sample_frame_;
    ActionParam param_;
};

#define ACTION_LOG_ERRORLEVEL_DEBUG 1
#define ACTION_LOG_ERRORLEVEL_INFO 2
#define ACTION_LOG_ERRORLEVEL_ERROR 3

#define ACTION_LOG_ERRORLOG(...) printf(__VA_ARGS__);

#if !defined(ACTION_LOG_ERRORACTIVE_LEVEL)
#define ACTION_LOG_ERRORACTIVE_LEVEL ACTION_LOG_ERRORLEVEL_ERROR
#endif

#if ACTION_LOG_ERRORACTIVE_LEVEL <= ACTION_LOG_ERRORLEVEL_INFO
#define ACTION_LOG_INFO(...) ACTION_LOG_ERRORLOG(__VA_ARGS__)
#else
#define ACTION_LOG_INFO(...) (void)0;
#endif

#if ACTION_LOG_ERRORACTIVE_LEVEL <= ACTION_LOG_ERRORLEVEL_DEBUG
#define ACTION_LOG_DEBUG(...) ACTION_LOG_ERRORLOG(__VA_ARGS__)
#else
#define ACTION_LOG_DEBUG(...) (void)0;
#endif

#if ACTION_LOG_ERRORACTIVE_LEVEL <= ACTION_LOG_ERRORLEVEL_ERROR
#define ACTION_LOG_ERROR(...) ACTION_LOG_ERRORLOG(__VA_ARGS__)
#else
#define ACTION_LOG_ERROR(...) (void)0;
#endif

}   // namespace action
}   // namespace gddeploy