#include <memory>
#include <string>
#include <math.h>
#include "rk_preproc.h"
#include "rk_common.h"

#include "core/mem/buf_surface.h"
#include "core/preprocess.h"
#include "opencv2/opencv.hpp"

#include "core/model.h"
#include "core/mem/buf_surface_util.h"
#include "core/result_def.h"

#include "common/logger.h"
#include "common/type_convert.h"

// #include "preproc/transform/transform.h"
#include "RgaUtils.h"
#include "im2d.hpp"
#include "im2d_common.h"
#include "im2d_buffer.h"
#include "rga.h"

using namespace gddeploy;

namespace gddeploy{   
class RkPreProcPriv{
public:
    RkPreProcPriv(ModelPtr model):model_(model){
        for (int i = 0; i < model->InputNum(); i++){
            auto shape = model_->InputShape(i);
            model_h_ = shape[2];
            model_w_ = shape[3];
            model_c_ = shape[1];
            batch_num_ = shape[0];
        }
    }
    ~RkPreProcPriv(){
        for (auto pool : pools_){
            // pool->DestroyPool();
            delete pool;
        }
        pools_.clear();
    }

    int Init(std::string config); 

    BufSurfWrapperPtr RequestBuffer(){
        BufSurfWrapperPtr buf_ptr = pools_[0]->GetBufSurfaceWrapper();

        return buf_ptr;
    }

    int GetModelWidth(){
        return model_w_;
    }

    int GetModelHeight(){
        return model_h_;
    }

    int PreProc(BufSurfaceParams *in_surf_param, BufSurfaceParams *out_surf_param, std::vector<InferResult> result);

private:
    ModelPtr model_;
    int model_h_;
    int model_w_;
    int model_c_;
    int batch_num_;
    
    std::vector<std::pair<std::string, gddeploy::any>> ops_;

    // 预分配中间过程内存
    std::shared_ptr<void> resize_cache_;
    std::shared_ptr<void> bgr2rgb_cache_;
    std::shared_ptr<void> float_cache_;
    std::shared_ptr<void> norn_cache_;
    std::shared_ptr<void> hwc2chw_cache_;

    std::vector<BufPool*> pools_;
};
}

static int CreatePool(ModelPtr model, BufPool *pool, BufSurfaceMemType mem_type, int block_count) {
    // 解析model，获取必要结构
    const DataLayout input_layout =  model->InputLayout(0);
    auto dtype = input_layout.dtype;
    auto order = input_layout.order;
    int data_size = 0;
    if (dtype == DataType::INT8 || dtype == DataType::UINT8){
        data_size = sizeof(uint8_t);
    }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
        data_size = sizeof(uint16_t);
    }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
        data_size = sizeof(uint32_t);
    }

    int model_h, model_w, model_c, model_b;
    auto shape = model->InputShape(0);
    if (order == DimOrder::NCHW){
        model_b = shape[0];
        model_h = shape[2];
        model_w = shape[3];
        model_c = shape[1];
    }else if (order == DimOrder::NHWC){
        model_b = shape[0];
        model_h = shape[1];
        model_w = shape[2];
        model_c = shape[3];
    }

    BufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(create_params));
    create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = model_b;
    create_params.size = model_h * model_w * model_c;
    create_params.size *= data_size;
    create_params.width = model_w;
    create_params.height = model_h;
    create_params.bytes_per_pix = data_size;

    ModelPropertiesPtr mp = model->GetModelInfoPriv();
    std::string net_type = mp->GetNetType();
    if (net_type == "yolox")
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER;
    else
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;

    if (pool->CreatePool(&create_params, block_count) < 0) {
        return -1;
    }
    return 0;
}


int RkPreProcPriv::Init(std::string config){
    // 预分配内存池
    size_t i_num = model_->InputNum();
    for (size_t i_idx = 0; i_idx < i_num; ++i_idx) {
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_BMNN, 3);
        pools_.emplace_back(pool);
    }

    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    
    // if (model_type == "classification" && net_type == "ofa"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
    //         .padding_num = 0,
    //     };
    //     gddeploy::transform::ComposeNormalizeParam norn_param = {
    //         .mean = {123.675, 116.28, 103.53},
    //         .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
    //     };
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
    //     if (order == DimOrder::NHWC){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //     }
    // } else if (model_type == "detection" && net_type == "yolo"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_CENTER,
    //         .padding_num = 114,
    //     };
    //     gddeploy::transform::ComposeNormalizeParam norn_param = {
    //         .mean = {0, 0, 0},
    //         .std = {1 / 255, 1 / 255, 1 / 255},
    //     };
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    // } else if (model_type == "pose" && net_type == "yolox"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_LEFT_TOP,
    //         .padding_num = 114,
    //     };
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
        
    //     if (product == "Tsingmicro"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //         ops_.insert(ops_.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
    //     } else if (product == "Cambricon"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //         ops_.emplace_back(std::pair<std::string, gddeploy::any>("float", true));
    //     }
    // } else if (model_type == "segmentation" && net_type == "OCRNet"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_LEFT_TOP,
    //         .padding_num = 114,
    //     };

    //     gddeploy::transform::ComposeNormalizeParam norn_param = {
    //         .mean = {123.675, 116.28, 103.53},
    //         .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
    //     };

    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
    //     if (product == "Tsingmicro"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //         ops_.insert(ops_.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
    //     } else if (product == "Cambricon"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //     }
    // // }else if (net_type == "action"){
    // //     Preprocyolov5(input_mat, output_mat, model_h_, model_w_);
    // } else if (model_type == "image-retrieval" && net_type == "dolg"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_CENTER,
    //         .padding_num = 114,
    //     };

    //     gddeploy::transform::ComposeNormalizeParam norn_param = {
    //         .mean = {123.675, 116.28, 103.53},
    //         .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
    //     };

    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
    //     if (product == "Tsingmicro"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //         ops_.insert(ops_.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
    //     } else if (product == "Cambricon"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //     }
    // } else if (model_type == "image-retrieval" && net_type == "arcface"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
    //         .padding_num = 0,
    //     };

    //     gddeploy::transform::ComposeNormalizeParam norn_param = {
    //         .mean = {127.5, 127.5, 127.5},
    //         .std = {1 / 127.5, 1 / 127.5, 1 / 127.5},
    //     };

    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
    //     if (product == "Tsingmicro"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //         ops_.insert(ops_.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
    //     } else if (product == "Cambricon"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //     }
    // } else if (model_type == "ocr" && net_type == "ocr_rec"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
    //         .padding_num = 114,
    //     };

    //     gddeploy::transform::ComposeNormalizeParam norn_param = {
    //         .mean = {103.53, 116.28, 123.675},
    //         .std = {1 / 57.375, 1 / 57.12, 1 / 58.395},
    //     };

    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
    //     if (product == "Tsingmicro"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //         ops_.insert(ops_.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
    //     } else if (product == "Cambricon"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //     }
    // } else if (model_type == "ocr" && net_type == "ocr_rec" || model_type == "ocr" && net_type == "resnet31v2ctc"){
    //     gddeploy::transform::ComposeResizeParam resize_param = {
    //         .in_w = input_mat.cols,
    //         .in_h = input_mat.rows,
    //         .out_w = model_w_,
    //         .out_h = model_h_,
    //         .type = gddeploy::transform::ResizeProcessType::RESIZE_PT_DEFAULT,
    //         .padding_num = 114,
    //     };

    //     gddeploy::transform::ComposeNormalizeParam norn_param = {
    //         .mean = {123.675, 116.28, 103.53},
    //         .std = {1 / 58.395, 1 / 57.12, 1 / 57.375},
    //     };

    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("bgr2rgb", true));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("resize", resize_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("normalize", norn_param));
    //     ops_.emplace_back(std::pair<std::string, gddeploy::any>("hwc2chw", true));
    //     if (product == "Tsingmicro"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //         ops_.insert(ops_.begin()+2, std::pair<std::string, gddeploy::any>("float", true));
    //     } else if (product == "Cambricon"){
    //         for (auto &iter : ops_){
    //             if (iter.first == "hwc2chw"){
    //                 iter.second = false;
    //             }
    //         }
    //     }
    // }
    
    // 预分配中间过程内存
    // int resize_cache_size = model_h_ * model_w_ * model_c_ * sizeof(uint8_t);
    // resize_cache_ = std::shared_ptr<void>(malloc(resize_cache_size));
    
    // int bgr2rgb_cache_size = model_h_ * model_w_ * model_c_ * sizeof(uint8_t);
    // bgr2rgb_cache_ = std::shared_ptr<void>(malloc(bgr2rgb_cache_size));
    
    // int float_cache_size = model_h_ * model_w_ * model_c_ * sizeof(float);
    // float_cache_ = std::shared_ptr<void>(malloc(float_cache_size));
    
    // int norn_cache_size = model_h_ * model_w_ * model_c_ * sizeof(float);
    // norn_cache_ = std::shared_ptr<void>(malloc(norn_cache_size));
    
    // int hwc2chw_cache_size = model_h_ * model_w_ * model_c_ * sizeof(float);
    // hwc2chw_cache_ = std::shared_ptr<void>(malloc(hwc2chw_cache_size));

    return 0;
}


int RkPreProcPriv::PreProc(BufSurfaceParams *in_surf_param, BufSurfaceParams *out_surf_param, std::vector<InferResult> result)
{
    // auto ret = gddeploy::transform::Compose(in_surf_param, out_surf_param, ops_);
    int width_align_bits = 16;
    
    int src_width = in_surf_param->width;
    int src_height = in_surf_param->height;
    int src_width_align = src_width / width_align_bits ? (src_width / width_align_bits + 1) * width_align_bits : src_width;
    int src_height_align = src_height / 2 ? (src_height / 2 + 1) * 2 : src_height;
    int src_format = convertSurfFormat2RKFormat(in_surf_param->color_format);
    int src_buf_size = src_width * src_height * get_bpp_from_format(src_format);
    void *src_buf = in_surf_param->data_ptr;

    int dst_width = out_surf_param->width;
    int dst_height = out_surf_param->height;
    int dst_width_align = dst_width / width_align_bits ? (dst_width / width_align_bits + 1) * width_align_bits : dst_width;
    int dst_height_align = dst_height / 2 ? (dst_height / 2 + 1) * 2 : dst_height;
    int dst_format = convertSurfFormat2RKFormat(out_surf_param->color_format);
    int dst_buf_size = dst_width * dst_height * get_bpp_from_format(dst_format);
    void *dst_buf = out_surf_param->data_ptr;

    rga_buffer_handle_t src_handle, dst_handle;
#if RKNN_RGA_VEWSION_1_3
    src_handle = wrapbuffer_virtualaddr(src_buf, src_width, src_height, src_format);
    dst_handle = wrapbuffer_virtualaddr(dst_buf, dst_width, dst_height, dst_format);
    if(src_handle.width == 0 || dst_handle.width == 0) {
        printf("wrapbuffer_virtualaddr error, %s, %s\n", __FUNCTION__, imStrError());
        goto release_buffer;
    }

    // ret = imresize(src_handle, dst_handle);
    // if (ret != IM_STATUS_SUCCESS) {
    //     printf("unning failed, %s\n", imStrError((IM_STATUS)ret));
    //     goto release_buffer;
    // }
#elif RKNN_RGA_VEWSION_1_9
    src_handle = importbuffer_virtualaddr(src_buf, src_buf_size);
    dst_handle = importbuffer_virtualaddr(dst_buf, dst_buf_size);
    if (src_handle == 0 || dst_handle == 0) {
        printf("importbuffer failed!\n");
        return -1;
    }
    rga_buffer_t src_buffer = wrapbuffer_handle(src_handle, src_width_align, src_height, src_format);
    rga_buffer_t dst_buffer = wrapbuffer_handle(dst_handle, dst_width_align, dst_height, dst_format);
#endif

    int ret = imcheck(src_buffer, dst_buffer, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }

    // 判断in_surf_param格式，如果是bgr或者nv12，需要转rgb
    if (in_surf_param->color_format != GDDEPLOY_BUF_COLOR_FORMAT_RGB){
        void *rgb_buf = malloc(src_width * src_height * 3);
        rga_buffer_handle_t rgb_handle = importbuffer_virtualaddr(rgb_buf, src_width * src_height * get_bpp_from_format(RK_FORMAT_RGB_888));
        rga_buffer_t rgb_buffer = wrapbuffer_handle(rgb_handle, src_width, src_height, RK_FORMAT_RGB_888);

        ret = imcvtcolor(src_buffer,
                     rgb_buffer,
                     src_format,
                     dst_format);
        if (IM_STATUS_NOERROR != ret) {
            printf("%d, imcvtcolor error! %s", __LINE__, imStrError((IM_STATUS)ret));
            return -1;
        }

        src_buffer = rgb_buffer;
    }

    // resize
    ret = imresize(src_buffer, dst_buffer);
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, imresize error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }
    
    return 0;
}

Status RkPreProc::Init(std::string config) noexcept
{ 
    printf("Rk Init\n");

    //TODO: 这里要补充解析配置，得到网络类型等新型
    if (false == HaveParam("model_info")){
        return gddeploy::Status::INVALID_PARAM;
    }
    ModelPtr model = GetParam<ModelPtr>("model_info");

    priv_ = std::make_shared<RkPreProcPriv>(model);

    priv_->Init(config);

    return gddeploy::Status::SUCCESS; 
}

Status RkPreProc::Init(ModelPtr model, std::string config)
{
    priv_ = std::make_shared<RkPreProcPriv>(model);

    priv_->Init(config);

    model_ = model;

    return gddeploy::Status::SUCCESS; 
}

Status RkPreProc::Process(PackagePtr pack) noexcept
{
    std::vector<InferResult> result;
    for (auto &data : pack->data){
        if (data->HasMetaValue())
            result.emplace_back(data->GetMetaData<gddeploy::InferResult>());
    }

    int batch_idx = 0;
    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    for (auto &data : pack->data){
        auto src_surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        BufSurface *src_surface = src_surf->GetBufSurface();
        BufSurfaceParams *src_param = src_surf->GetSurfaceParams(0);

        BufSurfWrapperPtr dst_buf = priv_->RequestBuffer();
        BufSurface *dst_surf = dst_buf->GetBufSurface();
        BufSurfaceParams *dst_param = dst_buf->GetSurfaceParams(0);

        // auto t0 = std::chrono::high_resolution_clock::now();
        int ret = priv_->PreProc(src_param, dst_param, result);
        if (ret != 0){
            GDDEPLOY_ERROR("[register] [rk preproc] PreProc error !!!");
            return gddeploy::Status::ERROR_BACKEND;
        }

        // auto t1 = std::chrono::high_resolution_clock::now();
        // printf("preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        
        infer_data->Set(std::move(dst_buf));
    }

    pack->predict_io =infer_data;

    return gddeploy::Status::SUCCESS; 
}