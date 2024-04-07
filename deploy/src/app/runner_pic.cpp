#include "app/runner_pic.h"

#include <fstream>
#include <dirent.h>
#include <memory>
#include <sys/stat.h>

#include "api/infer_api.h"
#include "api/global_config.h"
// #include "app/endecode.h"
#include "app/result_handle.h"
#include "common/type_convert.h"
#include "core/mem/buf_surface_util.h"
#include "core/result_def.h"

#include "opencv2/opencv.hpp"

#if WITH_NVIDIA
#include "pic/nvjpeg.h"
#endif

namespace gddeploy {

static std::string GetFilename(std::string path)
{
    std::string::size_type iPos = path.find_last_of('/') + 1;
    std::string filename = path.substr(iPos, path.length() - 1);
    return filename;
}

class PicRunnerPriv{
public:
    PicRunnerPriv()=default;

    int Init(const std::string config, std::string model_path, std::string license = "");
    int Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license);

    int InferSync(std::string pic_path, std::string save_path="", bool is_draw=false);    //推理路径下所有图片
    int InferMultiPicSync(std::vector<std::string> pic_path, std::string save_path="", bool is_draw=false);    //推理路径下所有图片

    int InferAsync(std::string pic_path, std::string save_path="", bool is_draw=false);

private:
    InferAPI infer_api_;

    std::vector<InferAPI> infer_api_v_;

    any pic_hw_decoder_; //选择图片解码器

    any pic_hw_encoder_;//选择图片编码器
};

}

using namespace gddeploy;

int PicRunnerPriv::Init(const std::string config, std::string model_path, std::string license)
{
    gddeploy_init("");

    infer_api_.Init(config, model_path);

    // TODO: 判断该硬件是否有硬件编解码接口

    return 0;
}

int PicRunnerPriv::Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths)
{
    gddeploy_init("");

    for (int i = 0; i < model_paths.size(); i++){
        std::string model_path = model_paths[i];
        std::string license_path = license_paths[i];
        
        InferAPI infer_api;
        infer_api.Init(config, model_path, license_path, ENUM_API_SESSION_API);

        infer_api_v_.push_back(infer_api);
    }
    // TODO: 判断该硬件是否有硬件编解码接口

    return 0;
}

int PicRunnerPriv::InferSync(std::string pic_path, std::string save_path, bool is_draw)
{
    // 1. 图片解码
    // if(pic_hw_decoder_->is_support(pic_path)){  // 硬件支持可支持该图片编码格式，否则改为OpenCV软解

    // }else{
    //     //采用OpenCV解码
    // }
#if 0
    NvjpegImageProcessor jpg_processor;
    BufSurface surface;
    std::vector<std::string> pic_files = {pic_path};

    jpg_processor.Decode(pic_files, &surface);

    gddeploy::BufSurfWrapperPtr surf = std::make_shared<gddeploy::BufSurfaceWrapper>(&surface, false);
#else
    cv::Mat in_mat = cv::imread(pic_path);

    // 2. 推理
    gddeploy::BufSurfWrapperPtr surf;
#if WITH_BM1684
    bm_image img;
    cv::bmcv::toBMI(in_mat, &img, true);
    convertBmImage2BufSurface(img, surf, false);
#else
    convertMat2BufSurface(in_mat, surf, true);
#endif

#endif
    gddeploy::PackagePtr in = gddeploy::Package::Create(1);
    in->data[0]->Set(surf);

    gddeploy::PackagePtr out = gddeploy::Package::Create(1);

    if (infer_api_v_.size() > 0){
        for (auto &infer_api : infer_api_v_){
            // 循环将上一模型结果传入下一模型的输入
            if (true == out->data[0]->HasMetaValue()){
                // in->data[0]->SetMetaData(std::move(out->data[0]->GetMetaData<gddeploy::InferResult>()));
                in->data[0] = std::move(out->data[0]);
            }
                
            infer_api.InferSync(in, out);
            if (false == out->data[0]->HasMetaValue())
                break;
            
            gddeploy::InferResult result = out->data[0]->GetMetaData<gddeploy::InferResult>();

            PrintResult(result);
            // 3. 编码输出
            if (is_draw && save_path != ""){
            // if(pic_hw_encoder_->is_support(pic_path)){  // 硬件支持可支持该图片编码格式，否则改为OpenCV软解

            // }else{
            //     //采用OpenCV编码
            // }
                auto pic_paths = std::vector<std::string>{pic_path};
                auto surfs = std::vector<gddeploy::BufSurfWrapperPtr>{};
                auto in_mats = std::vector<cv::Mat>{};
                std::string img_save_path = save_path + GetFilename(pic_path);

                DrawBbox(result, pic_paths, in_mats, surfs, img_save_path);
            }
        }
    } else {
        infer_api_.InferSync(in, out);

        if (false == out->data[0]->HasMetaValue())
        return -1;

        gddeploy::InferResult result = out->data[0]->GetMetaData<gddeploy::InferResult>(); 
        PrintResult(result);

        // 3. 编码输出
        if (is_draw && save_path != ""){
        // if(pic_hw_encoder_->is_support(pic_path)){  // 硬件支持可支持该图片编码格式，否则改为OpenCV软解

        // }else{
        //     //采用OpenCV编码
        // }
            auto pic_paths = std::vector<std::string>{pic_path};
            auto surfs = std::vector<gddeploy::BufSurfWrapperPtr>{};
            auto in_mats = std::vector<cv::Mat>{};

            DrawBbox(result, pic_paths, in_mats, surfs, save_path);
        }
    }

    return 0;
}

int PicRunnerPriv::InferMultiPicSync(std::vector<std::string> pic_path, std::string save_path, bool is_draw)
{
    // 1. 图片解码
    // if(pic_hw_decoder_->is_support(pic_path)){  // 硬件支持可支持该图片编码格式，否则改为OpenCV软解

    // }else{
    //     //采用OpenCV解码
    // }
    int pic_num = pic_path.size();
    gddeploy::PackagePtr in = gddeploy::Package::Create(pic_num);
    std::vector<cv::Mat> in_mats;
    std::vector<gddeploy::BufSurfWrapperPtr> surfs;
    std::vector<std::string> pic_paths;

    for (int i = 0; i < pic_num; i++){
    #if 0
        NvjpegImageProcessor jpg_processor;
        BufSurface surface;
        std::vector<std::string> pic_files = {pic_path};

        jpg_processor.Decode(pic_files, &surface);

        gddeploy::BufSurfWrapperPtr surf = std::make_shared<gddeploy::BufSurfaceWrapper>(&surface, false);
    #else
        cv::Mat in_mat = cv::imread(pic_path[i]);

        // 2. 推理
        gddeploy::BufSurfWrapperPtr surf;
    #if WITH_BM1684
        bm_image img;
        cv::bmcv::toBMI(in_mat, &img, true);
        convertBmImage2BufSurface(img, surf, false);
    #else
        convertMat2BufSurface(in_mat, surf, true);
    #endif

    #endif
        
        in->data[i]->Set(surf);


        in_mats.push_back(in_mat);
        surfs.push_back(surf);
        pic_paths.push_back(pic_path[i]);
    }

    gddeploy::PackagePtr out = gddeploy::Package::Create(1);

    if (infer_api_v_.size() > 0){
        for (auto &infer_api : infer_api_v_){
            // 循环将上一模型结果传入下一模型的输入
            if (true == out->data[0]->HasMetaValue())
                in->data[0]->SetMetaData(out->data[0]->GetMetaData<gddeploy::InferResult>());

            infer_api.InferSync(in, out);
            if (false == out->data[0]->HasMetaValue())
                break;
            
            for (int i = 0; i < pic_num; i++){
                gddeploy::InferResult result = out->data[i]->GetMetaData<gddeploy::InferResult>();

                PrintResult(result);
                // 3. 编码输出
                if (is_draw && save_path != ""){
                // if(pic_hw_encoder_->is_support(pic_path)){  // 硬件支持可支持该图片编码格式，否则改为OpenCV软解

                // }else{
                //     //采用OpenCV编码
                // }
                    auto pic_paths_tmp = std::vector<std::string>{pic_paths[i]};
                    auto surfs_tmp = std::vector<gddeploy::BufSurfWrapperPtr>{surfs[i]};
                    auto in_mats_tmp = std::vector<cv::Mat>{in_mats[i]};

                    std::string img_save_path = save_path + GetFilename(pic_paths[i]);

                    DrawBbox(result, pic_paths_tmp, in_mats_tmp, surfs_tmp, img_save_path);
                }
            }
        }
    } else {
        infer_api_.InferSync(in, out);

        if (false == out->data[0]->HasMetaValue())
        return -1;

        gddeploy::InferResult result = out->data[0]->GetMetaData<gddeploy::InferResult>(); 
        PrintResult(result);

        // 3. 编码输出
        if (is_draw && save_path != ""){
        // if(pic_hw_encoder_->is_support(pic_path)){  // 硬件支持可支持该图片编码格式，否则改为OpenCV软解

        // }else{
        //     //采用OpenCV编码
        // }
            auto pic_paths = std::vector<std::string>{pic_path};
            auto surfs = std::vector<gddeploy::BufSurfWrapperPtr>{};
            auto in_mats = std::vector<cv::Mat>{};

            DrawBbox(result, pic_paths, in_mats, surfs, save_path);
        }
    }
    return 0;
}

int PicRunnerPriv::InferAsync(std::string pic_path, std::string save_path, bool is_draw)
{
    //可一次解码多张图再推理，增大batch
    return 0;
}


PicRunner::PicRunner(){
    priv_ = std::make_shared<PicRunnerPriv>();
}

int PicRunner::Init(const std::string config, std::string model_path, std::string license)
{
    priv_->Init(config, model_path);

    return 0;
}

int PicRunner::Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license)
{
    priv_->Init(config, model_paths, license);

    return 0;
}

int PicRunner::InferSync(std::string pic_path, std::string save_path, bool is_draw)
{
    // 查询是否图片还是图片路径，获取全部图片
    std::vector<std::string> imgs_path;
    struct stat s_buf;
    stat(pic_path.c_str(), &s_buf);
    if (S_ISDIR(s_buf.st_mode)){
        struct dirent * ptr;
        DIR *dir = opendir((char *)pic_path.c_str()); //打开一个目录
        while((ptr = readdir(dir)) != NULL) //循环读取目录数据
        {
            if (std::string(ptr->d_name) == "." || std::string(ptr->d_name) == "..")
                continue;
            
            if (-1 == std::string(ptr->d_name).find(".jpg")
                && -1 == std::string(ptr->d_name).find(".png")) {   //目录
                continue;
            }
            // TODO: 判断后缀是否为可支持的图片格式，排除非图片文件
            imgs_path.push_back(pic_path+std::string(ptr->d_name));
        }
    }else{
        imgs_path.push_back(pic_path);
    }

    // 循环异步推理图片数据
    if (imgs_path.size() > 1){
        priv_->InferMultiPicSync(imgs_path, save_path, is_draw);
    } else {
        priv_->InferSync(imgs_path[0], save_path, is_draw);
    }

    return 0;
}

int PicRunner::InferAsync(std::string pic_path, std::string save_path, bool is_draw)
{
        // 查询是否图片还是图片路径，获取全部图片
    std::vector<std::string> imgs_path;
    struct stat s_buf;
    stat(pic_path.c_str(), &s_buf);
    if (S_ISDIR(s_buf.st_mode)){
        struct dirent * ptr;
        DIR *dir = opendir((char *)pic_path.c_str()); //打开一个目录
        while((ptr = readdir(dir)) != NULL) //循环读取目录数据
        {
            if (std::string(ptr->d_name) == "." || std::string(ptr->d_name) == "..")
                continue;
            imgs_path.push_back(pic_path+std::string(ptr->d_name));
        }
    }else{
        imgs_path.push_back(pic_path);
    }

    // 循环异步推理图片数据
    for (auto img_path : imgs_path){
        std::string img_save_path = save_path + GetFilename(img_path);

        priv_->InferAsync(img_save_path, save_path, is_draw);
    }
    return 0;
}

