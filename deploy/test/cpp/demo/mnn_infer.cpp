#include <iostream>
#include <string.h>

#include <opencv2/opencv.hpp>

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>

#include "yolov5_common.h"

int main()
{
    printf("hello\n");
    std::string pic_path = "/root/gddeploy/data/pic/helmet3.jpg";
    std::string model_path = "/root/gddeploy/data/models/gddi_model.mnn";
#if 1
    // 1. 读取模型及其相关属性
    std::shared_ptr<MNN::Interpreter> net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    if (nullptr == net) {
        return 0;
    }

    MNN::ScheduleConfig config;
    config.numThread = 4;
    config.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
    // backendConfig.precision =  MNN::PrecisionMode Precision_Normal; // static_cast<PrecisionMode>(Precision_Normal);
    config.backendConfig = &backendConfig;
    MNN::Session *session = net->createSession(config);;

    auto inputTensor = net->getSessionInput(session, nullptr);
    
    auto input_dims_num = inputTensor->dimensions();
    auto input_dims = inputTensor->shape();

    // printf("input dims: ");
    // for (auto dim : input_dims){
    //     printf("%d ", dim);
    // }
    // printf("\n");

#define INPUT_SIZE 640
    

    // 2. 准备数据
    int model_h = INPUT_SIZE;
    int model_w = INPUT_SIZE;

    cv::Mat img_mat = cv::imread(pic_path);
    cv::Mat pre_mat = preprocess(img_mat);
    std::vector<cv::Mat> bgrChannels;
    bgrChannels.emplace_back(model_w,model_h,CV_32F);
    bgrChannels.emplace_back(model_w,model_h,CV_32F);
    bgrChannels.emplace_back(model_w,model_h,CV_32F);
    cv::split(pre_mat, bgrChannels);

    std::vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_data   = nhwc_Tensor->host<float>();
    auto nhwc_size   = nhwc_Tensor->size();
    std::memcpy(nhwc_data, pre_mat.data, nhwc_size);

    inputTensor->copyFromHostTensor(nhwc_Tensor);

    

    // 3. 推理
    net->runSession(session);

    // get output data
    const std::map<std::string, MNN::Tensor*> output_maps = net->getSessionOutputAll(session);
    
    // 4. 后处理，画图
    std::vector<std::pair<std::string, MNN::Tensor*>> output_maps_v(output_maps.begin(), output_maps.end());
    std::sort(output_maps_v.begin(), output_maps_v.end(), 
            [](std::pair<std::string, MNN::Tensor*> &lhs, std::pair<std::string, MNN::Tensor*> &rhs){
                auto lhs_shape = lhs.second->shape();
                auto rhs_shape = rhs.second->shape();
        return lhs_shape[3] > rhs_shape[3];
    });

    std::vector<void *> cpu_output_ptrs;
    for (auto output_tensor: output_maps_v){
        MNN::Tensor *output  = net->getSessionOutput(session, output_tensor.first.c_str());
        MNN::Tensor host(output, output->getDimensionType());
        output->copyToHostTensor(&host);
        // MNN::Tensor *host = MNN::Tensor::createHostTensorFromDevice(output, true);
        cpu_output_ptrs.push_back(host.host<float>());
    }

    std::vector<ObjDetectInfos> objInfoss;
    int imgWidth = 500;
    int imgHeight = 350;
    Yolov5DetectionOutput(cpu_output_ptrs, objInfoss, 0, imgWidth, imgHeight);

    DrawRect(pic_path, "/root/gddeploy/data/pic/preds/helmet3.jpg", objInfoss[0].objInfos);

    printf("over\n");
#endif

    return 0;
}