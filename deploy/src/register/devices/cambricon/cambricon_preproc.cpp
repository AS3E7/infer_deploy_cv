#include <memory>
#include <string>

#include "core/mem/buf_surface.h"
#include "core/preprocess.h"
#include "opencv2/opencv.hpp"

#include "core/model.h"
#include "core/mem/buf_surface_util.h"
#include "core/infer_server.h"
#include "common/type_convert.h"

#include  "transform_cncv/transform_cncv.hpp"
#include "transform_cncv/cncv_transform.h"
#include "cambricon_preproc.h"

#if 1
using namespace gddeploy;

namespace gddeploy{   
class CambriconPreProcPriv{
public:
    CambriconPreProcPriv(ModelPtr model):model_(model){
        for (int i = 0; i < model->InputNum(); i++){
            auto shape = model_->InputShape(i);
            model_h_ = shape[2];
            model_w_ = shape[3];
            model_c_ = shape[1];
            batch_num_ = shape[0];
        }

    }
    ~CambriconPreProcPriv(){
    }

    int Init(std::string config); 

    int PreProc(BufSurface* src, BufSurface* dst);

    int SetModel(ModelPtr model){
        if (model == nullptr){
            return -1;
        }else{
            model_ = model;
        }
        return 0;
    }

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

private:
    ModelPtr model_;
    int model_h_;
    int model_w_;
    int model_c_;
    int batch_num_;

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
    create_params.mem_type = mem_type;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = model_b;
    create_params.size = model_h * model_w * model_c;
    create_params.size *= data_size;
    create_params.width = model_w;
    create_params.height = model_h;
    create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;

    if (pool->CreatePool(&create_params, block_count) < 0) {
        return -1;
    }
    return 0;
}


int CambriconPreProcPriv::Init(std::string config){
    // 预分配内存池
    size_t i_num = model_->InputNum();
    for (size_t i_idx = 0; i_idx < i_num; ++i_idx) {
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_CAMBRICON, 3);
        pools_.emplace_back(pool);
    }

    return 0;
}


TransformRect KeepAspectRatio(int src_w, int src_h, int dst_w, int dst_h) {
    float src_ratio = static_cast<float>(src_w) / src_h;
    float dst_ratio = static_cast<float>(dst_w) / dst_h;
    TransformRect res;
    memset(&res, 0, sizeof(res));
    
    if (src_ratio < dst_ratio) {
        int pad_lenth = dst_w - src_ratio * dst_h;
        pad_lenth = (pad_lenth % 2) ? pad_lenth - 1 : pad_lenth;
        if (dst_w - pad_lenth / 2 < 0) 
            return res;
        res.width = dst_w - pad_lenth;
        res.left = pad_lenth / 2;
        res.top = 0;
        res.height = dst_h;
    } else if (src_ratio > dst_ratio) {
        int pad_lenth = dst_h - dst_w / src_ratio;
        pad_lenth = (pad_lenth % 2) ? pad_lenth - 1 : pad_lenth;
        if (dst_h - pad_lenth / 2 < 0) return res;
        res.height = dst_h - pad_lenth;
        res.top = pad_lenth / 2;
        res.left = 0;
        res.width = dst_w;
    } else {
        res.left = 0;
        res.top = 0;
        res.width = dst_w;
        res.height = dst_h;
    }
    return res;
}

int CambriconPreProcPriv::PreProc(BufSurface* src_surf, BufSurface* dst_surf)
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    const DataLayout input_layout =  model_->InputLayout(0);
    auto order = input_layout.order;

    int model_h, model_w, model_c, model_b;
    auto shape = model_->InputShape(0);
    // configur dst_desc
    TransformTensorDesc dst_desc;
    dst_desc.color_format = GDDEPLOY_TRANSFORM_COLOR_FORMAT_RGB;
    dst_desc.data_type = GDDEPLOY_TRANSFORM_UINT8;

    if (order == infer_server::DimOrder::NHWC) {
        dst_desc.shape.n = shape[0];
        dst_desc.shape.h = shape[1];
        dst_desc.shape.w = shape[2];
        dst_desc.shape.c = shape[3];
    } else if (order == infer_server::DimOrder::NCHW) {
        dst_desc.shape.n = shape[0];
        dst_desc.shape.c = shape[1];
        dst_desc.shape.h = shape[2];
        dst_desc.shape.w = shape[3];
    } 

    if (net_type == "yolo") {
        std::vector<TransformRect> src_rect(model_b);
        std::vector<TransformRect> dst_rect(model_b);
        TransformParams params;
        memset(&params, 0, sizeof(params));
        params.transform_flag = 0;
        if (src_rects.size()) {
            params.transform_flag |= GDDEPLOY_TRANSFORM_CROP_SRC;
            params.src_rect = src_rect.data();
            memset(src_rect.data(), 0, sizeof(TransformRect) * model_b);
            for (uint32_t i = 0; i < model_b; i++) {
                TransformRect *src_bbox = &src_rect[i];
                *src_bbox = src_rects[i];
                // validate bbox
                src_bbox->left -= src_bbox->left & 1;
                src_bbox->top -= src_bbox->top & 1;
                src_bbox->width -= src_bbox->width & 1;
                src_bbox->height -= src_bbox->height & 1;
                while (src_bbox->left + src_bbox->width > src_surf->surface_list[i].width) src_bbox->width -= 2;
                while (src_bbox->top + src_bbox->height > src_surf->surface_list[i].height) src_bbox->height -= 2;
            }
        }

        params.transform_flag |= GDDEPLOY_TRANSFORM_CROP_DST;
        params.dst_rect = dst_rect.data();
        memset(dst_rect.data(), 0, sizeof(TransformRect) * model_b);
        for (uint32_t i = 0; i < model_b; i++) {
            TransformRect *dst_bbox = &dst_rect[i];
            *dst_bbox = KeepAspectRatio(src_surf->surface_list[i].width, src_surf->surface_list[i].height, dst_desc.shape.w,
                                        dst_desc.shape.h);
            // validate bbox
            dst_bbox->left -= dst_bbox->left & 1;
            dst_bbox->top -= dst_bbox->top & 1;
            dst_bbox->width -= dst_bbox->width & 1;
            dst_bbox->height -= dst_bbox->height & 1;
            while (dst_bbox->left + dst_bbox->width > dst_desc.shape.w) dst_bbox->width -= 2;
            while (dst_bbox->top + dst_bbox->height > dst_desc.shape.h) dst_bbox->height -= 2;
        }

        params.dst_desc = &dst_desc;
    }

    CncvTransform(src_surf, dst_surf, TransformParams* transform_params);

    return 0;
}

Status CambriconPreProc::Init(std::string config) noexcept
{ 
    return gddeploy::Status::SUCCESS; 
}

Status CambriconPreProc::Init(ModelPtr model, std::string config)
{
    priv_ = std::make_shared<CambriconPreProcPriv>(model);

    priv_->Init(config);

    return gddeploy::Status::SUCCESS; 
}

Status CambriconPreProc::Process(PackagePtr pack) noexcept
{    
    BufSurfWrapperPtr dst_buf = priv_->RequestBuffer();
    BufSurface *dst_surf = dst_buf->GetBufSurface();

    BufSurface src_surf;
    src_surf.mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
    src_surf.batch_size = dst_surf->batch_size;
    src_surf.num_filled = 1;
    src_surf.is_contiguous = 0;    // AVFrame的两个plane地址不一定连续
    
    src_surf.surface_list = new BufSurfaceParams[dst_surf->batch_size];
    int batch_idx = 0;
    BufSurfaceMemType mem_type;

    for (auto &data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        BufSurface *surface = surf->GetBufSurface();
        BufSurfaceParams *src_param = surf->GetSurfaceParams(0);
        mem_type = surface->mem_type;

        if (surface->mem_type == GDDEPLOY_BUF_MEM_CAMBRICON){
            for (int i = 0; i < surface->batch_size; i++){
                src_surf.surface_list[batch_idx++] = *(src_param+i);
            }
        } else {    // 来着CPU，需要拷贝
            for (int i = 0; i < surface->batch_size; i++){
                // 图片大小不确定，无法预分配内存
                void *data_ptr = nullptr;
                cnrtMalloc(&data_ptr, src_param->data_size);
                cnrtMemcpy(data_ptr, src_param->data_ptr, src_param->data_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
                
                src_surf.surface_list[batch_idx] = *(src_param+i);
                src_surf.surface_list[batch_idx].data_ptr = data_ptr;
                batch_idx++;
            }
        }
    }
    priv_->PreProc(&src_surf, dst_surf);

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(dst_buf));
    
    // pack->data.clear();
    pack->predict_io =infer_data;

    return gddeploy::Status::SUCCESS; 
}
#endif