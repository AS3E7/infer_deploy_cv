#include "cambricon_predictor.h"
#include <string>
#include <memory.h>
#include <vector>

#include "core/mem/buf_surface.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface_impl.h"
#include "core/model.h"
#include "cnrt.h"

using namespace gddeploy;
// 通过ort_model的priv_获取到session
// 拷贝输入PackPtr里面的数据到session  Cambricon2xx::Value input_tensor = Cambricon2xx::Value::CreateTensor
// session推理，得到的结果拷贝到输出的output PackPtr
#define CHECK_CNRT_RET(ret, msg, val)              \
  do {                                             \
    if ((ret) != CNRT_RET_SUCCESS) {               \
      std::cout << msg << " error code: " << ret << std::endl; \
      return ;                                  \
    }                                              \
  } while (0)

static cnrtDataType convertMyDataType2CambriconDataType(gddeploy::DataType in){
    cnrtDataType out;

    switch (in)
    {
    case DataType::INT8:
        out = cnrtDataType::CNRT_INT8;
        break;
    case DataType::UINT8:
        out = cnrtDataType::CNRT_UINT8;
        break;
    case DataType::FLOAT32:
        out = cnrtDataType::CNRT_FLOAT32;
        break;
    case DataType::FLOAT16:
        out = cnrtDataType::CNRT_FLOAT16;
        break;
    case DataType::UINT16:
        out = cnrtDataType::CNRT_UINT16;
        break;
    case DataType::INT16:
        out = cnrtDataType::CNRT_INT16;
        break;
    case DataType::INT32:
        out = cnrtDataType::CNRT_INT32; //cnrtDataType::u32
        break;
    default:
        out = cnrtDataType::CNRT_INVALID;
        break;
    }
    return out;
}

static int getDataSize(gddeploy::DataType dtype)
{
    int data_size = 0;
    if (dtype == DataType::INT8 || dtype == DataType::UINT8){
        data_size = sizeof(uint8_t);
    }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
        data_size = sizeof(uint16_t);
    }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
        data_size = sizeof(uint32_t);
    }

    return data_size;
}

namespace gddeploy
{
class Cambricon2xxPredictorPrivate{
public:
    Cambricon2xxPredictorPrivate() = default;
    Cambricon2xxPredictorPrivate(ModelPtr model):model_(model){
        cnrtInit(0);

        function_ = gddeploy::any_cast<std::shared_ptr<cnrtFunction_t>>(model->GetModel());
        // cnrt_model_ = *((cnrtModel_t*)cnrt_model_ptr_.get());

        // auto error_code = cnrtCreateFunction(&function_);
        // error_code = cnrtExtractFunction(&function_, cnrt_model_, func_name_.c_str());

        // int model_parallelism;
        // error_code = cnrtQueryModelParallelism(cnrt_model_, &model_parallelism);

        cnrtSetCurrentChannel((cnrtChannelType_t)dev_channel_);

        input_num_ = model->InputNum();
        output_num_ = model->OutputNum();
        cnrtRet_t ret = cnrtCreateRuntimeContext(&ctx_, *function_, NULL);
        CHECK_CNRT_RET(ret, "[EasyDK InferServer] [ModelRunner] Create runtime context failed!", false);

        cnrtChannelType_t channel = CNRT_CHANNEL_TYPE_NONE;
        // ret = cnrtSetRuntimeContextChannel(ctx_, channel);
        // CHECK_CNRT_RET(ret, "[EasyDK InferServer] [ModelRunner] Set Runtime Context Channel failed!", false);
        // ret = cnrtSetRuntimeContextDeviceId(ctx_, device_id_);
        // CHECK_CNRT_RET(ret, "[EasyDK InferServer] [ModelRunner] Set Runtime Context Device Id failed!", false);
        ret = cnrtInitRuntimeContext(ctx_, NULL);
        CHECK_CNRT_RET(ret, "[EasyDK InferServer] [ModelRunner] Init runtime context failed!", false);

        // VLOG(1) << "[EasyDK InferServer] [ModelRunner] Create CNRT queue from runtime context";
        ret = cnrtRuntimeContextCreateQueue(ctx_, &task_queue_);
        CHECK_CNRT_RET(ret, "[EasyDK InferServer] [ModelRunner] Runtime Context Create Queue failed", false);

        params_ = new void*[input_num_ + output_num_];
    }
    ~Cambricon2xxPredictorPrivate(){
        SetCurrentDevice(device_id_);
        if (task_queue_) {
            cnrtDestroyQueue(task_queue_);
            task_queue_ = nullptr;
        }
        if (ctx_) {
            cnrtDestroyRuntimeContext(ctx_);
            ctx_ = nullptr;
        }
        if (params_) {
            delete[] params_;
            params_ = nullptr;
        }

        // cnrtDestroyFunction(*function_);
        // cnrtUnloadModel(cnrt_model_);

        cnrtDestroy();
    }

    BufSurfWrapperPtr RequestBuffer(int idx){
        BufSurfWrapperPtr buf = pools_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    BufSurfWrapperPtr RequestInputDevBuffer(int idx){
        BufSurfWrapperPtr buf = input_dev_pools_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    BufSurfWrapperPtr RequestOutputDevBuffer(int idx){
        BufSurfWrapperPtr buf = output_dev_pools_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    int Init(std::string config);

    std::shared_ptr<cnrtFunction_t> function_;
    // cnrtModel_t cnrt_model_;
    cnrtRuntimeContext_t ctx_{nullptr};
    void** params_{nullptr};
    cnrtQueue_t task_queue_{nullptr};

    ModelPtr model_;

    uint32_t input_num_{0};
    uint32_t output_num_{0};
private:
    // std::shared_ptr<void> cnrt_model_ptr_;

    std::vector<BufPool*> pools_;
    std::vector<BufPool*> input_dev_pools_;
    std::vector<BufPool*> output_dev_pools_;

    std::string func_name_ = "subnet0";

    int dev_channel_{0};
    int device_id_{0};
};
}

int Cambricon2xxPredictorPrivate::Init(std::string config)
{

    // 分配input显存空间
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
        create_params.mem_type = GDDEPLOY_BUF_MEM_CAMBRICON;
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
        input_dev_pools_.emplace_back(pool);
    }

    // 分配output显存空间
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
        create_params.mem_type = GDDEPLOY_BUF_MEM_CAMBRICON;
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
        output_dev_pools_.emplace_back(pool);
    }

    // 输出的主内存空间
    for (size_t i_idx = 0; i_idx < o_num; ++i_idx) {
        const DataLayout input_layout =  model_->OutputLayout(i_idx);
        auto dtype = input_layout.dtype;
        auto order = input_layout.order;
        int data_size = sizeof(uint32_t);
        // if (dtype == DataType::INT8 || dtype == DataType::UINT8){
        //     data_size = sizeof(uint8_t);
        // }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
        //     data_size = sizeof(uint16_t);
        // }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
        //     data_size = sizeof(uint32_t);
        // }

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
        if (pool->CreatePool(&create_params, 3) < 0) {
            return -1;
        }
        pools_.emplace_back(pool);
    }

    return 0;
}

namespace gddeploy
{
Status Cambricon2xxPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<Cambricon2xxPredictorPrivate>(model);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}


Status Cambricon2xxPredictor::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    BufSurface *surf = in_buf->GetBufSurface();
    BufSurfaceParams* src_param = in_buf->GetSurfaceParams();

    float* tmp_host = nullptr;
    // 1. 拷贝数据
    if (surf->mem_type == GDDEPLOY_BUF_MEM_CAMBRICON){
        priv_->params_[0] = in_buf->GetData(0, 0);
    } else {
        BufSurface *host_surf = surf;
        BufSurfWrapperPtr dev_surf_ptr = priv_->RequestInputDevBuffer(0);
        BufSurface *dev_surf = dev_surf_ptr->GetBufSurface();

        // 这里的前输入格式一般是fp32，而模型输入可能是fp32/fp16，需要手动转换
        const DataLayout input_layout =  priv_->model_->InputLayout(0);
        auto dtype = input_layout.dtype;
        if (dtype != DataType::FLOAT32 && dtype != DataType::INT32){
            BufSurface *tmp = new BufSurface();
            *tmp = *host_surf;
            tmp->surface_list = reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * host_surf->batch_size)); 
            memcpy(tmp->surface_list, surf->surface_list, sizeof(BufSurfaceParams) * host_surf->batch_size);

            tmp->surface_list[0].data_ptr = malloc(dev_surf->surface_list[0].data_size);
            tmp->surface_list[0].data_size = dev_surf->surface_list[0].data_size;

            cnrtCastDataType(reinterpret_cast<void *>(host_surf->surface_list[0].data_ptr), CNRT_FLOAT32, 
                    reinterpret_cast<void *>(tmp->surface_list[0].data_ptr),//转换后的目的地址，分配的cpu内存
                    convertMyDataType2CambriconDataType(dtype),
                    tmp->surface_list[0].data_size / getDataSize(dtype), //转换的数据个数
                    NULL); 

            host_surf = tmp;
        }

        auto mem_allocator = CreateMemAllocator(GDDEPLOY_BUF_MEM_CAMBRICON);
        mem_allocator->Copy(host_surf, dev_surf);       

        priv_->params_[0] = dev_surf_ptr->GetData(0, 0);

        if (dtype != DataType::FLOAT32 && dtype != DataType::INT32){
            free(host_surf->surface_list[0].data_ptr);
            free(host_surf->surface_list);
        }
    }
    

    std::vector<BufSurfWrapperPtr> out_bufs;
    for (size_t i = 0; i < priv_->output_num_; ++i) {
        BufSurfWrapperPtr surf = priv_->RequestOutputDevBuffer(i);
        priv_->params_[priv_->input_num_ + i] = surf->GetData(0, 0);

        out_bufs.emplace_back(surf);
    }

    unsigned int affinity = 1 << 0;//设置通道亲和性,使用指定的MLU·cluster做推理
    cnrtInvokeParam_t invokeParam;//invoke参数              
                                                           
    //设置invoke的参数                                                          
    invokeParam.invoke_param_type = CNRT_INVOKE_PARAM_TYPE_0;                   
    invokeParam.cluster_affinity.affinity = &affinity;   
    
    // 2. 推理
    cnrtInvokeRuntimeContext(priv_->ctx_, priv_->params_, priv_->task_queue_, &invokeParam);
    cnrtSyncQueue(priv_->task_queue_);

    // copy to host memory
    std::vector<BufSurfWrapperPtr> out_host_bufs;
    auto mem_allocator = CreateMemAllocator(GDDEPLOY_BUF_MEM_CAMBRICON);
    for (size_t i = 0; i < priv_->output_num_; ++i) {
        BufSurfWrapperPtr host_surf_ptr = priv_->RequestBuffer(i);
        BufSurface* host_surf = host_surf_ptr->GetBufSurface();
        BufSurface* host_surf_tmp = host_surf;

        BufSurfWrapperPtr dev_surf_ptr = out_bufs[i];
        BufSurface* dev_surf = dev_surf_ptr->GetBufSurface();

        const DataLayout output_layout =  priv_->model_->OutputLayout(i);
        auto dtype = output_layout.dtype;
        if (dtype != DataType::FLOAT32 && dtype != DataType::INT32){    //TODO:考虑到数据size不一样，需要拷贝
            BufSurface *tmp = new BufSurface();
            *tmp = *dev_surf;

            tmp->surface_list = reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * dev_surf->batch_size)); 
            memcpy(tmp->surface_list, dev_surf->surface_list, sizeof(BufSurfaceParams) * dev_surf->batch_size);

            tmp->surface_list[0].data_ptr = malloc(dev_surf->surface_list[0].data_size);
            tmp->surface_list[0].data_size = dev_surf->surface_list[0].data_size;
            tmp->mem_type = GDDEPLOY_BUF_MEM_SYSTEM;

            host_surf_tmp = tmp;
        }

        mem_allocator->Copy(dev_surf, host_surf_tmp);

        if (dtype != DataType::FLOAT32 && dtype != DataType::INT32){
            cnrtCastDataType(reinterpret_cast<void *>(host_surf_tmp->surface_list[0].data_ptr), convertMyDataType2CambriconDataType(dtype), 
                    reinterpret_cast<void *>(host_surf->surface_list[0].data_ptr),//转换后的目的地址，分配的cpu内存
                    CNRT_FLOAT32,
                    host_surf_tmp->surface_list[0].data_size / getDataSize(dtype), //转换的数据个数
                    NULL); 
        }

        out_host_bufs.emplace_back(host_surf_ptr);

        if (dtype != DataType::FLOAT32 && dtype != DataType::INT32){
            free(host_surf_tmp->surface_list[0].data_ptr);
            free(host_surf_tmp->surface_list);
        }
    }

    pack->predict_io->Set(std::move(out_host_bufs));

    return gddeploy::Status::SUCCESS; 
}
}