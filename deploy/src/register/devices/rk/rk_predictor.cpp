#include <string>
#include <memory.h>
#include <vector>

#include "rk_predictor.h"
#include "core/mem/buf_surface.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "common/logger.h"

#include "rknn_api.h"

using namespace gddeploy;

namespace gddeploy
{
class RkPredictorPrivate{
public:
    RkPredictorPrivate() = default;
    RkPredictorPrivate(ModelPtr model, int core_id):model_(model){
        auto ctx_tmp = gddeploy::any_cast<rknn_context>(model->GetModel());

        int ret = rknn_dup_context(&ctx_tmp, &ctx_);
        if (ret < 0)
        {
            printf("rknn_dup_context error ret=%d\n", ret);
        }
        
        rknn_core_mask core_mask = RKNN_NPU_CORE_0;
        if (core_id == 0){
            core_mask = RKNN_NPU_CORE_0;
        } else if (core_id == 1){
            core_mask = RKNN_NPU_CORE_1;
        } else if (core_id == 2){
            core_mask = RKNN_NPU_CORE_2;
        } else{
            core_mask = RKNN_NPU_CORE_0_1_2;
        }
        rknn_set_core_mask(ctx_, RKNN_NPU_CORE_AUTO);

        rknn_input_output_num io_num;
        ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret < 0)
        {
            printf("rknn_query in/out num error ret=%d\n", ret);
        }
        
        for (int i = 0; i < io_num.n_input; i++){
            rknn_tensor_attr input_attr;
            ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attr),
                    sizeof(rknn_tensor_attr));
            if (ret < 0)
            {
                printf("rknn_query input attr error ret=%d\n", ret);
            }
            input_attr_.emplace_back(input_attr);
        }
        
        for (int i = 0; i < io_num.n_output; i++){
            rknn_tensor_attr output_attr;
            output_attr.index = i;
            ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attr),
                    sizeof(rknn_tensor_attr));
            if (ret < 0)
            {
                printf("rknn_query input attr error ret=%d\n", ret);
            }

            output_attr_.emplace_back(output_attr);
        }
    }
    ~RkPredictorPrivate(){
        for (auto pool : pools_){
            // pool->DestroyPool();
            delete pool;
        }
        pools_.clear();
    }

    BufSurfWrapperPtr RequestBuffer(int idx){
        BufSurfWrapperPtr buf = pools_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    int Init(std::string config);
    ModelPtr model_;
    
    rknn_context ctx_;
    std::vector<rknn_tensor_attr> input_attr_;
    std::vector<rknn_tensor_attr> output_attr_;

private:
    std::vector<BufPool*> pools_;
};
}

int RkPredictorPrivate::Init(std::string config)
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
        create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
        create_params.force_align_1 = 1;  // to meet mm's requirement
        create_params.device_id = 0;
        create_params.batch_size = model_b;
        create_params.size = model_h * model_w * model_c;
        create_params.size *= data_size;
        create_params.width = model_w;
        create_params.height = model_h;
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;

        BufPool *pool = new BufPool;
        if (pool->CreatePool(&create_params, 6) < 0) {
            return -1;
        }
        pools_.emplace_back(pool);
    }

    return 0;
}

Status RkPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<RkPredictorPrivate>(model, dev_id_);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}


Status RkPredictor::Process(PackagePtr pack) noexcept
{
    // auto t0 = std::chrono::high_resolution_clock::now();
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    BufSurface *surf = in_buf->GetBufSurface();
    int batch_size = surf->batch_size;  

    auto src_param = in_buf->GetSurfaceParams();

    int ret = -1;
    // 1. 拷贝数据
    if (surf->mem_type == GDDEPLOY_BUF_MEM_RK_RGA){
        int img_size = src_param->data_size;

        for (int i = 0; i < batch_size; i++){
            // imgs[i] = *((bm_image *)surf->surface_list[i].data_ptr);
        }

    } else {
        uint32_t img_size = src_param->data_size;
        
    }
    rknn_input inputs[1];
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = priv_->input_attr_[0].size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = surf->surface_list[0].data_ptr;

    ret = rknn_inputs_set(priv_->ctx_, 1, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set error ret=%d\n", ret);
        return gddeploy::Status::ERROR_BACKEND;
    }

    // rknn_set_io_mem(priv_->ctx_, rknn_tensor_mem *mem, priv_->input_attr_.data());

    
    // 2. 推理
    auto t0 = std::chrono::high_resolution_clock::now();
    if ((ret = rknn_run(priv_->ctx_, NULL)) < 0){
        GDDEPLOY_ERROR("[register] [rk predictor] rknn_run error");
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("##################inference time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    int out_num = priv_->output_attr_.size();
    rknn_output outputs[out_num];
    memset(outputs, 0, sizeof(outputs));
    std::vector<BufSurfWrapperPtr> out_bufs;
    for (int i = 0; i < out_num; i++)
    {
        outputs[i].want_float = 0;
        outputs[i].is_prealloc = 1;

        // 请求申请一块CPU内存
        BufSurfWrapperPtr buf = priv_->RequestBuffer(i);
        outputs[i].buf = buf->GetData(0, 0);

        out_bufs.emplace_back(buf);
    }
    ret = rknn_outputs_get(priv_->ctx_, out_num, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get error ret=%d\n", ret);
        return gddeploy::Status::ERROR_BACKEND;
    }

    
    // for (unsigned int i = 0; i < out_num; ++i){
    //     // 请求申请一块CPU内存
    //     BufSurfWrapperPtr buf = priv_->RequestBuffer(i);

    //     auto output_shape = priv_->model_->OutputShape(i);
    //     auto data_count  = output_shape.BatchDataCount();
    
    //     memcpy((void *)buf->GetData(0, 0), outputs[i].buf, priv_->output_attr_[i].size);

    //     out_bufs.emplace_back(buf);
    // }    

    pack->predict_io->Set(std::move(out_bufs));

    rknn_outputs_release(priv_->ctx_, out_num, outputs);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("infer time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    return gddeploy::Status::SUCCESS; 
}
