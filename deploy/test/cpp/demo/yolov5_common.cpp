#include "yolov5_common.h"
#include <iostream>
#include <string.h>

cv::Mat preprocess(cv::Mat input_mat)
{
    cv::Mat img;
    int letterbox_rows = 640;
    int letterbox_cols = 640;

    if (input_mat.channels() == 1)
        cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

    int w, h, x, y;

    int input_mat_w_ = img.cols;
    int input_mat_h_ = img.rows;
    float r_w = letterbox_rows / (input_mat_w_*1.0);
    float r_h = letterbox_cols / (input_mat_h_*1.0);

    if (r_h > r_w) {
        w = letterbox_rows;
        h = r_w * input_mat_h_;
        x = 0;
        y = (letterbox_cols - h) / 2;
    } else {
        w = r_h * input_mat_w_;
        h = letterbox_cols;
        x = (letterbox_rows - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::Mat out(letterbox_cols, letterbox_rows, CV_8UC3, cv::Scalar(114, 114, 114));

    cv::resize(img, re, re.size(), 0, 0, cv::INTER_NEAREST);
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    cv::Mat img_new(letterbox_cols, letterbox_rows, CV_32FC3);
    out.convertTo(img_new, CV_32FC3, 1/255.0);

    return img_new;
}


void DrawRect(std::string src_file, std::string dst_file, std::vector<ObjDetectInfo> detect_infos)
{
    cv::Mat src = cv::imread(src_file);

    for (auto detect_info : detect_infos){
        cv::Rect rect(detect_info.leftTopX, detect_info.leftTopY, detect_info.rightBotX-detect_info.leftTopX, detect_info.rightBotY-detect_info.leftTopY);
        cv::rectangle(src, rect, cv::Scalar(255, 0, 0), 1, cv::LINE_8, 0);
    }
    cv::imwrite(dst_file, src);
}

float iou(const ObjDetectInfo& box1, const ObjDetectInfo& box2)
{
    float x1 = std::max(box1.leftTopX, box2.leftTopX);
    float y1 = std::max(box1.leftTopY, box2.leftTopY);
    float x2 = std::min(box1.leftTopX+box1.rightBotX, box2.leftTopX+box2.rightBotX);
    float y2 = std::min(box1.leftTopY+box1.rightBotY, box2.leftTopY+box2.rightBotY);
    float over_w = std::max(0.0f, x2 - x1);
    float over_h = std::max(0.0f, y2 - y1);
    float over_area = over_w * over_h;
    float iou = over_area / ((box1.rightBotX ) * (box1.rightBotY ) + (box2.rightBotX ) * (box2.rightBotY ) - over_area);
    
    return iou;
}

void get_rect(int img_width, int img_height,int crop_width, int crop_height, ObjDetectInfo *rectBbox) {
    int w, h, x, y;
    float r_w = crop_width / (img_width * 1.0);
    float r_h = crop_height / (img_height * 1.0);

    int result_width = rectBbox->rightBotX;
    int result_height = rectBbox->rightBotY;

    if (r_h > r_w) 
    {
        rectBbox->leftTopX = rectBbox->leftTopX / r_w;
        rectBbox->rightBotX = result_width / r_w;
        rectBbox->rightBotY = result_height / r_w;

        h = r_w * img_height;
        y = (crop_height - h) / 2;
        rectBbox->leftTopY = (rectBbox->leftTopY - y) / r_w;

        rectBbox->rightBotX = rectBbox->leftTopX + rectBbox->rightBotX;
        rectBbox->rightBotY = rectBbox->leftTopY + rectBbox->rightBotY;
    }else
    {
        rectBbox->leftTopY = rectBbox->leftTopY / r_h;
        rectBbox->rightBotX = result_width / r_h;
        rectBbox->rightBotY = result_height / r_h;

        w = r_h * img_width;
        x = (crop_width - w) / 2;
        rectBbox->leftTopX = (rectBbox->leftTopX - x) / r_h;

        rectBbox->rightBotX = rectBbox->leftTopX + rectBbox->rightBotX;
        rectBbox->rightBotY = rectBbox->leftTopY + rectBbox->rightBotY;
    }

    if (rectBbox->leftTopX + result_width > img_width)
        result_width = img_width - rectBbox->leftTopX;
    if (rectBbox->leftTopY + result_height > img_height)
        result_height = img_height - rectBbox->leftTopY;
}

std::vector<ObjDetectInfo> nms(std::vector<ObjDetectInfo> objInfos, float conf_thresh, bool isMap)
{
    std::sort(objInfos.begin(), objInfos.end(), [](ObjDetectInfo lhs, ObjDetectInfo rhs)
              { return lhs.confidence > rhs.confidence; });
    if (objInfos.size() > 1000)
    {
        objInfos.erase(objInfos.begin() + 1000, objInfos.end());
    }

    std::vector<ObjDetectInfo> result;

    while (objInfos.size() > 0)
    {
        result.push_back(objInfos[0]);
  
        for (auto it = objInfos.begin() + 1; it != objInfos.end();)
        {
            if (isMap){
                if (objInfos[0].classId == it->classId)
                {

                    float iou_value = iou(objInfos[0], *it);
                    if (iou_value > conf_thresh){
                        it = objInfos.erase(it);
                    }else{
                        it++;
                    }
                }else{
                    it++;
                }
            }else{
                float iou_value = iou(objInfos[0], *it);
                if (iou_value > conf_thresh)
                    it = objInfos.erase(it);
                else
                    it++;  
            }
        }
        objInfos.erase(objInfos.begin());
    }

    return result;
}



/*
 * @description: Realize the Yolo layer to get detiction object info
 * @param featLayerData  Vector of 3 output feature data
 * @param objInfos  DetectBox vector after transformation
 * @param netWidth  Model input width
 * @param netHeight  Model input height
 * @param imgWidth  Real image width
 * @param imgHeight  Real image height
 */
void Yolov5DetectionOutput(std::vector<void *> featLayerData,  
                           std::vector<ObjDetectInfos>& objInfoss,
                           int picIndex, int imgWidth, int imgHeight)
{
    int modelWidth = 512;
    int modelHeight = 512;

    ObjDetectInfos frameObjInfos;
    frameObjInfos.channelId = 0;
    frameObjInfos.frameId = 0;
    bool isMap_ = false;

    std::vector<OutputLayer> outputlayers;
    // int outputShape[] = {80, 40, 20};
    int outputShape[] = {64, 32, 16};
    for (int i = 0; i < 3; ++i) {
        int shape = outputShape[i];
        const int scale = 8 << i;
        // const int scale = 32 >> i;
        int width = shape;
        int height = shape;
        int channel = 24;
        OutputLayer layer = {i, width, height, channel};
        outputlayers.push_back(layer);
    }

    for (const auto& layer : outputlayers) {
        float *castData = (float *)featLayerData[layer.layerIdx];

        int ratio = int(modelWidth / layer.width);
        int hh = layer.height;
        int ww = layer.width;
        int cc = layer.channel;
        int step = hh * ww;
        int group[3] = {0, cc / 3, cc / 3 *2};
        
        int classNum = cc / 3 - 5;

        float anchors_[] = {10,  13, 16,  30,  33,  23, 
                    30,  61, 62,  45,  59,  119, 
                    116, 90, 156, 198, 373, 326};
    
        for(int j = 0; j < 3; j++)
        {

            float* group_base_addr = castData + picIndex * step * cc + group[j]*step ;
            float* conf_channel = group_base_addr + step*4;
            float* x_channel = group_base_addr + step*0;
            float* y_channel = group_base_addr + step*1;
            float* w_channel = group_base_addr + step*2;
            float* h_channel = group_base_addr + step*3;
            
            for(int h = 0; h < hh; h++)
            {
                for(int w = 0; w < ww; w++)
                {
                    float conf = *(conf_channel + h * ww + w);
                    float conf_thresh = 0.1f;
                    if(conf > conf_thresh)
                    {
                        float bbox_x = *(x_channel + h * ww + w);
                        bbox_x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                        float bbox_y = *(y_channel + h * ww + w);
                        bbox_y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                        float bbox_w = *(w_channel + h * ww + w);
                        float bbox_h = *(h_channel + h * ww + w);
                        bbox_w = pow(bbox_w * 2.0f,2) * anchors_[layer.layerIdx*6 + j*2];
                        bbox_h = pow(bbox_h * 2.0f,2) * anchors_[layer.layerIdx*6 + j*2+1];

                        
                        float max_prob = 0.0f;
                        int classId = 0;
                        for(int cls = 0; cls < classNum; cls++)
                        {
                            float* cls_conf_channel = group_base_addr + step*(5+cls);
                            float cls_conf = *(cls_conf_channel + h * ww + w);
                            cls_conf = cls_conf * conf;
                            if (isMap_){
                                if(cls_conf > 0.01)
                                {
                                    max_prob = cls_conf;
                                    classId = cls;

                                    ObjDetectInfo result;
                                    result.leftTopX = bbox_x - bbox_w/2;
                                    result.leftTopY = bbox_y - bbox_h/2;
                                    result.rightBotX = bbox_w;
                                    result.rightBotY = bbox_h;
                                    result.confidence = max_prob;
                                    result.classId = classId;
                                    frameObjInfos.objInfos.push_back(result);
                                }
                            }else{
                                if(cls_conf > max_prob)
                                {
                                    max_prob = cls_conf;
                                    classId = cls;
                                }
                            }
                        }

                        if (!isMap_){
                            ObjDetectInfo result;
                            result.leftTopX = bbox_x - bbox_w/2;
                            result.leftTopY = bbox_y - bbox_h/2;
                            result.rightBotX = bbox_w;
                            result.rightBotY = bbox_h;
                            result.confidence = max_prob;
                            result.classId = classId;
                            frameObjInfos.objInfos.push_back(result);
                        }
                    }
                }
            }
        }
    } 

    if (frameObjInfos.objInfos.size() > 0){
        frameObjInfos.objInfos = nms(frameObjInfos.objInfos, 0.5, isMap_);

        for (auto &objInfo : frameObjInfos.objInfos){
            get_rect(imgWidth, imgHeight, modelWidth, modelHeight, &objInfo);
        }
    }

    objInfoss.push_back(frameObjInfos);
}