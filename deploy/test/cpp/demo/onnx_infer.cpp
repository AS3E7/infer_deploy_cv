#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string.h>

#include <opencv2/opencv.hpp>

#include "yolov5_common.h"

int main()
{
    printf("hello\n");
    std::string pic_path = "/data/data/pic/helmet3.jpg";

    // 1. 读取模型及其相关属性
    Ort::Env env;
    Ort::Session session{env, "/data/data/models/gddi_model.onnx", Ort::SessionOptions{nullptr}};

    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char *> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes;
    std::vector<std::vector<float>> input_values_handlers; // multi handlers.
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs

    int num_inputs = session.GetInputCount();
    input_node_names.resize(num_inputs);
    for (unsigned int i = 0; i < num_inputs; ++i)
    {
        input_node_names[i] = session.GetInputName(i, allocator);
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
        size_t tensor_size = 1;
        for (unsigned int j = 0; j < input_dims.size(); ++j)
            tensor_size *= input_dims.at(j);
        input_tensor_sizes.push_back(tensor_size);
        input_values_handlers.push_back(std::vector<float>(tensor_size));
    }
    // 4. output names & output dimms
    int num_outputs = session.GetOutputCount();
    output_node_names.resize(num_outputs);
    for (unsigned int i = 0; i < num_outputs; ++i)
    {
        output_node_names[i] = session.GetOutputName(i, allocator);
        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

    // 2. 准备数据
    const unsigned int target_channel = input_node_dims.at(0).at(1);
    const unsigned int target_height = input_node_dims.at(0).at(2);
    const unsigned int target_width = input_node_dims.at(0).at(3);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;

    std::vector<float> tensor_value_handler;
    tensor_value_handler.resize(target_tensor_size);

    int model_h = input_node_dims[0].at(2);
    int model_w = input_node_dims[0].at(2);

    cv::Mat img_mat = cv::imread(pic_path);
    cv::Mat pre_mat = preprocess(img_mat);
    std::vector<cv::Mat> bgrChannels;
    bgrChannels.emplace_back(model_w,model_h,CV_32F);
    bgrChannels.emplace_back(model_w,model_h,CV_32F);
    bgrChannels.emplace_back(model_w,model_h,CV_32F);
    cv::split(pre_mat, bgrChannels);

    for (unsigned int i = 0; i < target_channel; ++i){
      std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                  bgrChannels.at(i).data, target_height * target_width * sizeof(float));
    }

    std::vector<int64_t> input_node1_dims = input_node_dims.at(0);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                           target_tensor_size, input_node1_dims.data(), input_node1_dims.size());
    

    // 3. 推理
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, num_inputs,  output_node_names.data(), num_outputs);


    // 4. 后处理，画图
    std::vector<void *> cpu_output_ptrs;
    for (unsigned int i = 0; i < output_tensors.size(); ++i){
        auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        cpu_output_ptrs.push_back(output_tensors.at(i).GetTensorMutableData<float>());
    }
    
    std::vector<ObjDetectInfos> objInfoss;
    int imgWidth = 500;
    int imgHeight = 350;
    Yolov5DetectionOutput(cpu_output_ptrs, objInfoss, 0, imgWidth, imgHeight);

    DrawRect(pic_path, "/root/gddeploy/data/pic/preds/helmet3.jpg", objInfoss[0].objInfos);

    printf("over\n");


    return 0;
}