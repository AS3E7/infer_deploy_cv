#include <iostream>
#include "api/processor_api.h"
#include "opencv2/opencv.hpp"

using namespace gddeploy;

PackagePtr mat2package(cv::Mat &img)
{
    PackagePtr pack = std::make_shared<Package>();

    int size = img.cols * img.rows * 3 * sizeof(uint8_t);

    Buffer buf(img.data, size, [](void *memory, int device_id){});


    ModelIO out_io;
    gddeploy::Shape shape(std::vector<int64_t>{1, 3, img.rows, img.cols});
    out_io.buffers.emplace_back(buf);
    out_io.shapes.emplace_back(shape);

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(out_io));
    
    pack->data.emplace_back(infer_data);

    return pack;
}

int main(int argc, char **argv)
{
    if(argc < 2) {
        std::cout << "USAGE:" << std::endl;
        std::cout << "<model path> <pic path> [config]" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string pic_path = argv[2];
    std::string config = "";
    if (argc > 3)
        config = argv[3];

    // 1. 创建API对象，传入配置参数和模型路径
    ProcessorAPI api;
    api.Init(config, model_path);

    // 2. 获取多个processor
    std::vector<ProcessorPtr> processors = api.GetProcessor();

    // 3. 读取图片，拷贝到Package格式
    cv::Mat img = cv::imread(pic_path);
    PackagePtr pack = mat2package(img);

    // 4. 循环执行每个processor的Process函数
    for (auto processor : processors){
        processor->Process(pack);
    }
    
    // 5. 解析结果


    return 0;
}