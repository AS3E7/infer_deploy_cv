#pragma once

#include <string>
#include <vector>
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface_impl.h"

namespace gddeploy
{

// resize的算法类型，有INTER_NEAREST，INTER_LINEAR，INTER_BICUBIC
enum ResizeAlgorithmType
{
    INTER_NEAREST = 0,
    INTER_LINEAR,
    INTER_BICUBIC
};

// 单纯的resize参数
typedef struct {
    int dst_x;
    int dst_y;
    int dst_w;
    int dst_h;
    unsigned char padding_r;
    unsigned char padding_g;
    unsigned char padding_b;
    int if_memset;
} ResizeParam;

typedef struct {
    int start_x;
    int start_y;
    int crop_w;
    int crop_h;
} CropParam;

// crop和resize参数
typedef struct {
    std::vector<CropParam> crop_params;
    int out_x;
    int out_y;
    int out_w;
    int out_h;    
    ResizeAlgorithmType algorithm;
} CropResizeParam;


// CV算子抽象，一般包含resize、crop、normalize、csc等操作
// 输入输出以BufSurface为单位
class CV
{
public:
    static CV *Instance() noexcept
    {
        if (pInstance_ == nullptr)
        {
            pInstance_ = new CV();
        }
        return pInstance_;
    }

    int register_cv(std::string manufacturer, std::string chip, CV *predictor_creator) noexcept
    {
        std::unordered_map<std::string, CV *> chip_predictor_creator;
        chip_predictor_creator[chip] = predictor_creator;
        cv_map_[manufacturer] = chip_predictor_creator;

        return 0;
    }

    CV *GetCV(std::string manufacturer, std::string chip)
    {
        if (cv_map_.count(manufacturer) == 0)
            return nullptr;
        
        if (cv_map_[manufacturer].count("any") != 0)
            return cv_map_[manufacturer]["any"];

        return cv_map_[manufacturer][chip];
    }

    virtual int Resize(gddeploy::BufSurfWrapperPtr src, gddeploy::BufSurfWrapperPtr dst){ return 0; }

    virtual int Crop(gddeploy::BufSurfWrapperPtr src, std::vector<gddeploy::BufSurfWrapperPtr> dst, std::vector<CropParam> crop_params) { return 0; }

    virtual int CropAndResize(gddeploy::BufSurfWrapperPtr src, gddeploy::BufSurfWrapperPtr dst, CropResizeParam param){ return 0; }

    // virtual int Normalize(BufSurface &src, BufSurface &dst);

    // virtual int Csc(BufSurface &src, BufSurface &dst);

private:
    std::unordered_map<std::string, std::unordered_map<std::string, CV *>> cv_map_;

    static CV *pInstance_;
};

int register_cv_module();

}  // namespace gddeploy