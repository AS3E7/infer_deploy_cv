#include <iostream>
#include <string>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "opencv2/opencv.hpp"

#include "rknn_api.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rga.h"

#include "yolov5_common.h"

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static cv::Mat preprocess_noscale(cv::Mat input_mat)
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
    out.convertTo(img_new, CV_32FC3, 1.0);

    return img_new;
}




// static void DrawRect(std::string src_file, std::string dst_file, std::vector<ObjDetectInfo> detect_infos)
// {
//     cv::Mat src = cv::imread(src_file);

//     for (auto detect_info : detect_infos){
//         cv::Rect rect(detect_info.leftTopX, detect_info.leftTopY, detect_info.rightBotX-detect_info.leftTopX, detect_info.rightBotY-detect_info.leftTopY);
//         cv::rectangle(src, rect, cv::Scalar(255, 0, 0), 1, cv::LINE_8, 0);
//     }
//     cv::imwrite(dst_file, src);
// }

static float iou(const ObjDetectInfo& box1, const ObjDetectInfo& box2)
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

static void get_rect(int img_width, int img_height,int crop_width, int crop_height, ObjDetectInfo *rectBbox) {
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

static std::vector<ObjDetectInfo> nms(std::vector<ObjDetectInfo> objInfos, float conf_thresh, bool isMap)
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


static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}
static float sigmoid(float x)
{
    // return 1.0 / (1.0 + expf(-x));
    return x;
}

void Yolov5DetectionOutputScale(std::vector<void *> featLayerData,  
                           std::vector<ObjDetectInfos>& objInfoss,
                           int picIndex, int imgWidth, int imgHeight,
                            std::vector<float> out_scales,
                            std::vector<int32_t> out_zps)
{
    int modelWidth = 640;
    int modelHeight = 640;

    ObjDetectInfos frameObjInfos;
    frameObjInfos.channelId = 0;
    frameObjInfos.frameId = 0;
    bool isMap_ = false;

    std::vector<OutputLayer> outputlayers;
    int outputShape[] = {80, 40, 20};
    // int outputShape[] = {64, 32, 16};
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
                    float conf = sigmoid(deqnt_affine_to_f32(*(conf_channel + h * ww + w), out_zps[layer.layerIdx], out_scales[layer.layerIdx]));
                    float conf_thresh = 0.1f;
                    if(conf > conf_thresh)
                    {
                        float bbox_x = sigmoid(deqnt_affine_to_f32(*(x_channel + h * ww + w), out_zps[layer.layerIdx], out_scales[layer.layerIdx]));
                        bbox_x = (bbox_x * 2.0f - 0.5f + w) * ratio;
                        float bbox_y = sigmoid(deqnt_affine_to_f32(*(y_channel + h * ww + w), out_zps[layer.layerIdx], out_scales[layer.layerIdx]));
                        bbox_y = (bbox_y * 2.0f - 0.5f + h) * ratio;
                        float bbox_w = sigmoid(deqnt_affine_to_f32(*(w_channel + h * ww + w), out_zps[layer.layerIdx], out_scales[layer.layerIdx]));
                        float bbox_h = sigmoid(deqnt_affine_to_f32(*(h_channel + h * ww + w), out_zps[layer.layerIdx], out_scales[layer.layerIdx]));
                        bbox_w = pow(bbox_w * 2.0f,2) * anchors_[layer.layerIdx*6 + j*2];
                        bbox_h = pow(bbox_h * 2.0f,2) * anchors_[layer.layerIdx*6 + j*2+1];

                        
                        float max_prob = 0.0f;
                        int classId = 0;
                        for(int cls = 0; cls < classNum; cls++)
                        {
                            float* cls_conf_channel = group_base_addr + step*(5+cls);
                            float cls_conf = sigmoid(deqnt_affine_to_f32(*(cls_conf_channel + h * ww + w), out_zps[layer.layerIdx], out_scales[layer.layerIdx]));
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

int main(int argc, char** argv)
{
    std::string model_path = argv[1];
    std::string pic_path = argv[2];

    printf("hello\n");

    int model_size = 0;
    auto model_data = load_model(model_path.c_str(), &model_size);

    // 1. 初始化
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);
    
    rknn_input_output_num rknn_io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &rknn_io_num, sizeof(rknn_io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // printf("model input num: %d, output num: %d\n", rknn_io_num.n_input,
    //        rknn_io_num.n_output);
    rknn_tensor_attr input_attrs[rknn_io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < rknn_io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
    }

#if 1
    // 2. 前处理和设置输入参数
    int channel = 3;
    int model_w = 0;
    int model_h = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        // printf("model is NCHW input fmt\n");
        model_w = input_attrs[0].dims[0];
        model_h = input_attrs[0].dims[1];
    }
    else
    {
        // printf("model is NHWC input fmt\n");
        model_w = input_attrs[0].dims[1];
        model_h = input_attrs[0].dims[2];
    }

    rknn_input inputs[1];
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = model_w * model_h * channel* sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

#if 0
    cv::Mat img_mat = cv::imread(pic_path);
    cv::Mat pre_mat = preprocess_noscale(img_mat);
    // std::vector<cv::Mat> bgrChannels;
    // bgrChannels.emplace_back(model_w,model_h,CV_32F);
    // bgrChannels.emplace_back(model_w,model_h,CV_32F);
    // bgrChannels.emplace_back(model_w,model_h,CV_32F);
    // cv::split(pre_mat, bgrChannels);

    char pre_data[model_w*model_h*3*sizeof(float)] = {0};

    // for (unsigned int i = 0; i < 3; ++i){
    //   std::memcpy((void*)&pre_data[i * (model_w*model_h)],
    //               bgrChannels.at(i).data, model_w*model_h * sizeof(float));
    // }
    std::memcpy((void*)&pre_data[0],
                  pre_mat.data, model_w*model_h *channel* sizeof(float));
    inputs[0].buf = pre_data;
#endif
    printf("Read %s ...\n", pic_path);
    cv::Mat orig_img = cv::imread(pic_path, 1);
    if (!orig_img.data)
    {
        printf("cv::imread %s fail!\n", pic_path);
        return -1;
    }
    cv::Mat img = orig_img.clone();
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;
    printf("img width = %d, img height = %d\n", img_width, img_height);

    void *resize_buf = malloc(model_h * model_w * channel);
    // init rga context
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    
    src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void *)resize_buf, model_w, model_h, RK_FORMAT_RGB_888);
    ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }
    IM_STATUS STATUS = imresize(src, dst);
    cv::Mat resize_img(cv::Size(model_w, model_h), CV_8UC3, resize_buf);
    cv::imwrite("resize_input.jpg", resize_img);

    inputs[0].buf = resize_buf;

    ret = rknn_inputs_set(ctx, rknn_io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set error ret=%d\n", ret);
        return -1;
    }

    // 3. 推理
    ret = rknn_run(ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_run error ret=%d\n", ret);
        return -1;
    }

    // 分析逐层耗时
    // rknn_perf_detail perf_detail;
    // rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &perf_detail, sizeof(perf_detail));

    // rknn_perf_run perf_run;
    // ret = rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf_run, sizeof(perf_run));

    // rknn_mem_size mem_size;
    // ret = rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));

    // 4. 解析结果数据
    rknn_tensor_attr output_attrs[rknn_io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < rknn_io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                   sizeof(rknn_tensor_attr));
    }

    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;

    rknn_output outputs[rknn_io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < rknn_io_num.n_output; i++)
    {
        outputs[i].want_float = 1;
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    ret = rknn_outputs_get(ctx, rknn_io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get error ret=%d\n", ret);
        return -1;
    }

    std::vector<void *> cpu_output_ptrs;
    for (unsigned int i = 0; i < rknn_io_num.n_output; ++i){
        cpu_output_ptrs.push_back(outputs[i].buf);
    }
    
    std::vector<ObjDetectInfos> objInfoss;
    int imgWidth = 450;
    int imgHeight = 350;
    Yolov5DetectionOutputScale(cpu_output_ptrs, objInfoss, 0, imgWidth, imgHeight, out_scales, out_zps);

    printf("obj size:%d\n", objInfoss[0].objInfos.size());
    DrawRect(pic_path, "/home/linaro/gddeploy/preds/helmet3.jpg", objInfoss[0].objInfos);

    printf("over\n");

    rknn_outputs_release(ctx, rknn_io_num.n_output, outputs);

    rknn_destroy(ctx);

#endif
    return 0;
}