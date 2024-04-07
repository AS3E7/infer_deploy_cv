#include "core/preprocess.h"
#include "core/predictor.h"
#include "core/postprocess.h"
#include "common/json.hpp"
#include "core/register.h"
#include "register/devices/processor/preprocess/cpu/opencv_preproc.h"

#include <dlfcn.h>

#include "opencv2/opencv.hpp"

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"
#include "bm_wrapper.hpp"

using namespace nlohmann;
using namespace gddeploy;

// extern PreProcRegister<OpencvPreProcCreator> g_register_cpu_preproc_creator;

PackagePtr mat2package(cv::Mat &img)
{
    PackagePtr pack = std::make_shared<Package>();

    // int size = img.cols * img.rows * 3 * sizeof(uint8_t);

    // Buffer buf(img.data, size, [](void *memory, int device_id){});


    // ModelIO out_io;
    // gddeploy::Shape shape(std::vector<int64_t>{1, 3, img.rows, img.cols});
    // out_io.buffers.emplace_back(buf);
    // out_io.shapes.emplace_back(shape);

    // std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    // infer_data->Set(std::move(out_io));
    
    // pack->data.emplace_back(infer_data);

    return pack;
}

PackagePtr mat2package2(bm_handle_t handle , cv::Mat &img_mat)
{
    PackagePtr pack = std::make_shared<Package>();

    bm_image img;
    bm_image_create(handle, img_mat.rows, img_mat.cols, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE, &img);
    bm_image_from_mat(handle, img_mat, img);

    std::vector<bm_image> imgs;
    imgs.emplace_back(img);
    imgs.emplace_back(img);
    imgs.emplace_back(img);
    imgs.emplace_back(img);

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(imgs));
    
    pack->data.emplace_back(infer_data);

    return pack;
}

int main()
{
    gddeploy::register_all_module();

    // PreProcManagerTest mgr1;
    // auto test = mgr1.Get(true);
    // json j{
    //     { "input_data_type", "NV12"},
    //     { "resize", 640},
    //     { "norm", true},
    //     {"pad", true},
    //     {"pad_value", 114}
    // };

    // std::string pre_config_str = j.dump();
    
    // gddeploy::PreProcConfig pc(pre_config_str);

    // pc.PrintConfig();

    // PreProcManager mgr;
    // mgr.GetName();
    // mgr.GetPreProcCreator("any", "cpu");
    // OpencvPreProc opencv_preproc;


    // const std::string model_path = "/data/data/models/gddi_model_cry.onnx";
    const std::string model_path = "/data/gddeploy/data/models/gddi_model_jit.int8_cry.bmodel";
    const std::string properties_path = "/data/gddeploy/data/models/gddi_model.properties";
    const std::string pic_path = "/data/gddeploy/data/pic/helmet3.jpg";
    
    ModelPtr model = gddeploy::ModelManager::Instance()->Load(model_path, properties_path, "");



    // 1. 创建前处理模块
    // PreProcCreator* preproc_creator = gGetPreProcCreator("any", "cpu");
    PreProcManager* preprocmgr = PreProcManager::Instance();
    PreProcCreator* preproc_creator = preprocmgr->GetPreProcCreator("bmnn", "bm1684");

    auto p_bmrt = gddeploy::any_cast<std::shared_ptr<void>>(model->GetModel());

    auto bm_handle = (bm_handle_t)bmrt_get_bm_handle(p_bmrt.get());

    // if (preproc_creator != nullptr){
        PreProcPtr preproc =  preproc_creator->Create();
        preproc->Init(model, "");

    //     // 2. 读取图片，进行前处理
        cv::Mat img = cv::imread(pic_path);
        PackagePtr pack = mat2package2(bm_handle, img);

        preproc->Process(pack);

    //     // 3. 推理
        auto predictor_creator = PredictorManager::Instance()->GetPredictorCreator("bmnn", "bm1684");
        auto predictor = predictor_creator->Create(nullptr, model);
        predictor->Init(nullptr, model);

        predictor->Process(pack);

    //     // 4. 后处理
        auto postproc_creator = PostProcManager::Instance()->GetPostProcCreator("any", "cpu");
        auto postproc = postproc_creator->Create();
        postproc->Init(model, "");

        postproc->Process(pack);

    // }

    return 0;
}