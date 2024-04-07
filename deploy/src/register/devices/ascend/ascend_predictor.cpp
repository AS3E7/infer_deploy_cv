#include <string>
#include <memory.h>
#include <vector>

#include "ascend_predictor.h"
#include "core/mem/buf_surface.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "common/logger.h"

#include "acl/acl.h"

using namespace gddeploy;
// 通过ort_model的priv_获取到session
// 拷贝输入PackPtr里面的数据到session  Ascend::Value input_tensor = Ascend::Value::CreateTensor
// session推理，得到的结果拷贝到输出的output PackPtr

namespace gddeploy
{
class AscendPredictorPrivate{
public:
    AscendPredictorPrivate() = default;
    AscendPredictorPrivate(ModelPtr model):model_(model){
        void *model_id_ptr = gddeploy::any_cast<std::shared_ptr<void>>(model->GetModel()).get();
        model_id_ = *(uint32_t *)model_id_ptr;

        aclrtCreateContext(&context_model_, model_id_);

        aclrtCreateStream(&stream_);
    }
    ~AscendPredictorPrivate(){
        aclmdlDestroyDataset(input_);
        aclmdlDestroyDataset(output_);

        aclrtDestroyStream(stream_);
        aclrtDestroyContext(context_model_);
    }

    BufSurfWrapperPtr RequestBuffer(int idx){
        BufSurfWrapperPtr buf = pools_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    BufSurfWrapperPtr RequestInputDevBuffer(int idx){
        BufSurfWrapperPtr buf = pools_input_dev_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    BufSurfWrapperPtr RequestOutputDevBuffer(int idx){
        BufSurfWrapperPtr buf = pools_output_dev_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    int Init(std::string config);
    int Process(std::vector<BufSurfWrapperPtr> &in_bufs, std::vector<BufSurfWrapperPtr> &out_bufs);

    size_t num_inputs_;
    size_t num_outputs_;
    std::vector<const char *> input_node_names_;
    std::vector<const char *> output_node_names_;

    ModelPtr model_;
    
    aclrtContext context_model_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t model_id_;

    aclmdlDataset *input_;
    aclmdlDataset *output_;

private:
    std::vector<BufPool*> pools_input_dev_;
    std::vector<BufPool*> pools_output_dev_;
    std::vector<BufPool*> pools_;
};
}

int AscendPredictorPrivate::Init(std::string config)
{
    size_t o_num = model_->OutputNum();
    for (size_t i_idx = 0; i_idx < o_num; ++i_idx) {
        const DataLayout input_layout =  model_->OutputLayout(i_idx);
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
        auto shape = model_->OutputShape(i_idx);
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
        create_params.mem_type = GDDEPLOY_BUF_MEM_ASCEND_RT;
        create_params.force_align_1 = 1;  // to meet mm's requirement
        create_params.device_id = 0;
        create_params.batch_size = model_b;
        create_params.size = model_h * model_w * model_c;
        create_params.size *= data_size;
        create_params.width = model_w;
        create_params.height = model_h;
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;

        BufPool *pool = new BufPool;
        if (pool->CreatePool(&create_params, 3) < 0) {
            return -1;
        }
        pools_output_dev_.emplace_back(pool);

        // 创建系统内存池
        BufSurfaceCreateParams create_params_sys;
        memset(&create_params_sys, 0, sizeof(create_params_sys));
        create_params_sys.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
        create_params_sys.force_align_1 = 1;  // to meet mm's requirement
        create_params_sys.device_id = 0;
        create_params_sys.batch_size = model_b;
        create_params_sys.size = model_h * model_w * model_c;
        create_params_sys.size *= data_size;
        create_params_sys.width = model_w;
        create_params_sys.height = model_h;
        create_params_sys.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;

        BufPool *pool_sys = new BufPool;
        if (pool_sys->CreatePool(&create_params_sys, 3) < 0) {
            return -1;
        }
        pools_.emplace_back(pool_sys);
    }

    // input dev memory pool
    size_t i_num = model_->InputNum();
    for (size_t i_idx = 0; i_idx < i_num; ++i_idx) {
        const DataLayout input_layout =  model_->InputLayout(i_idx);
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
        auto shape = model_->InputShape(i_idx);
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
        create_params.mem_type = GDDEPLOY_BUF_MEM_ASCEND_RT;
        create_params.force_align_1 = 1;  // to meet mm's requirement
        create_params.device_id = 0;
        create_params.batch_size = model_b;
        create_params.size = model_h * model_w * model_c;
        create_params.size *= data_size;
        create_params.width = model_w;
        create_params.height = model_h;
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;

        BufPool *pool = new BufPool;
        if (pool->CreatePool(&create_params, 3) < 0) {
            return -1;
        }
        pools_input_dev_.emplace_back(pool);
    }

    input_ = aclmdlCreateDataset();
    output_ = aclmdlCreateDataset();

    // 讲申请的dev内存提前放入output_中
    for (size_t i_idx = 0; i_idx < o_num; ++i_idx) {
        BufSurfWrapperPtr output_surf_ptr = RequestOutputDevBuffer(i_idx);

        BufSurface *surf = output_surf_ptr->GetBufSurface();

        aclDataBuffer* outputData = aclCreateDataBuffer(surf->surface_list[0].data_ptr, surf->surface_list[0].data_size * surf->batch_size);
        aclError ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_ERROR_NONE) {
            GDDEPLOY_ERROR("[register] [ascend predictor] aclmdlAddDatasetBuffer error !!!");
        }
    }

    return 0;
}

int AscendPredictorPrivate::Process(std::vector<BufSurfWrapperPtr> &in_bufs, std::vector<BufSurfWrapperPtr> &out_bufs)
{
    // 轮训in_bufs，逐个获取data_ptr，拷贝到input_中
    for (int i = 0; i < in_bufs.size(); i++){
        BufSurfWrapperPtr in_buf = in_bufs[i];
        BufSurface *surf = in_buf->GetBufSurface();
        int batch_size = surf->batch_size;  

        auto src_param = in_buf->GetSurfaceParams();

        //1. 创建模型推理的输入数据
        int ret = -1;
        if (surf->is_contiguous){
            if (surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP){
                aclDataBuffer* inputData = aclCreateDataBuffer(surf->surface_list[0].data_ptr, surf->surface_list[0].data_size * batch_size);
                aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
                if (ret != ACL_ERROR_NONE) {
                    GDDEPLOY_ERROR("[register] [ascend predictor] aclmdlAddDatasetBuffer error !!!");
                }
            } else {
                // 从pools_input_dev_分配一块内存，然后调用h2d拷贝数据
                BufSurfWrapperPtr input_surf_ptr = RequestInputDevBuffer(i);
                BufSurface *input_surf = input_surf_ptr->GetBufSurface();
                for (int i = 0; i < batch_size; i++){
                    ret = aclrtMemcpy(input_surf->surface_list[i].data_ptr, input_surf->surface_list[i].data_size, surf->surface_list[i].data_ptr, surf->surface_list[i].data_size, ACL_MEMCPY_HOST_TO_DEVICE);
                    if (ret == -1){
                        GDDEPLOY_ERROR("[register] [ascend predictor] copy s2d error !!!");
                    }
                }
            }
        } else {
            if (surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP){
                int img_size = src_param->data_size;

                for (int i = 0; i < batch_size; i++){
                    aclDataBuffer* inputData = aclCreateDataBuffer(surf->surface_list[i].data_ptr, surf->surface_list[i].data_size);
                    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
                    if (ret != ACL_ERROR_NONE) {
                        GDDEPLOY_ERROR("[register] [ascend predictor] aclmdlAddDatasetBuffer error !!!");
                    }
                }
            } else {
                // 从pools_input_dev_分配一块内存，然后调用h2d拷贝数据
                BufSurfWrapperPtr input_surf_ptr = RequestInputDevBuffer(i);
                BufSurface *input_surf = input_surf_ptr->GetBufSurface();
                for (int i = 0; i < batch_size; i++){
                    ret = aclrtMemcpy(input_surf->surface_list[i].data_ptr, input_surf->surface_list[i].data_size, surf->surface_list[i].data_ptr, surf->surface_list[i].data_size, ACL_MEMCPY_HOST_TO_DEVICE);
                    if (ret == -1){
                        GDDEPLOY_ERROR("[register] [ascend predictor] copy s2d error !!!");
                    }
                }
            }
        }
    }

    // 2. 推理
    auto ret = aclmdlExecute(model_id_, input_, output_);
    if (ret != ACL_ERROR_NONE) {
        GDDEPLOY_ERROR("[register] [ascend predictor] aclmdlExecute error !!!");
        return -1;
    }

    // 拷贝数据到系统内存
    for (unsigned int i = 0; i < model_->OutputNum(); ++i){
        BufSurfWrapperPtr buf = out_bufs[i];
        BufSurface *surf = buf->GetBufSurface();
        int batch_size = surf->batch_size;  

        auto src_param = buf->GetSurfaceParams();

        aclDataBuffer *output_databuf = aclmdlGetDatasetBuffer(output_, i);
        void *output_data_ptr = aclGetDataBufferAddr(output_databuf);
        uint32_t output_data_size = aclGetDataBufferSize(output_databuf);

        //1. 创建模型推理的输入数据
        int ret = -1;
        if (surf->is_contiguous){
            ret = aclrtMemcpy(surf->surface_list[0].data_ptr, surf->surface_list[0].data_size * batch_size, output_data_ptr, output_data_size, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret == -1){
                GDDEPLOY_ERROR("[register] [ascend predictor] copy d2s error !!!");
            }
        } else {
            
        }
    }

    return 0;
}

Status AscendPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<AscendPredictorPrivate>(model);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}


Status AscendPredictor::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    std::vector<BufSurfWrapperPtr> in_bufs = {in_buf};
    std::vector<BufSurfWrapperPtr> out_bufs;
    // 1. 后处理，画图
    for (int i = 0; i < model_->OutputNum(); i++){
        // 请求申请一块CPU内存
        BufSurfWrapperPtr buf = priv_->RequestBuffer(i);

        out_bufs.emplace_back(buf);
    }
    
    if (priv_->Process(in_bufs, out_bufs) < 0){
        return gddeploy::Status::ERROR_BACKEND;
    }

    return gddeploy::Status::SUCCESS; 
}
