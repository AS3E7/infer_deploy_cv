#ifndef YOLOV5_COMMON_H
#define YOLOV5_COMMON_H

#include <opencv2/opencv.hpp>
#include <vector>

struct OutputLayer {
    int layerIdx;
    int width;
    int height;
    int channel;
    // float anchors[6];
};

struct ObjDetectInfo {
    float leftTopX;
    float leftTopY;
    float rightBotX;
    float rightBotY;
    float confidence;
    float classId;
};

struct ObjDetectInfos{
    int channelId;
    int frameId;
    std::vector<ObjDetectInfo> objInfos;
};


cv::Mat preprocess(cv::Mat input_mat);
void DrawRect(std::string src_file, std::string dst_file, std::vector<ObjDetectInfo> detect_infos);
void Yolov5DetectionOutput(std::vector<void *> featLayerData,  
                           std::vector<ObjDetectInfos>& objInfoss,
                           int picIndex, int imgWidth, int imgHeight);

#endif