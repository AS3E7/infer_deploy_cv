#include "nv_predictor.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface.h"
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <NvInferRuntime.h>
#include "common/logger.h"
#include "logging.h"
#include "nv_common.h"

using namespace gddeploy;
using namespace nvinfer1;
using namespace std;
// 通过ort_model的priv_获取到session
// 拷贝输入PackPtr里面的数据到session  Ort::Value input_tensor = Ort::Value::CreateTensor
// session推理，得到的结果拷贝到输出的output PackPtr

namespace gddeploy
{
    class NvPredictorPrivate
    {
    public:
        NvPredictorPrivate() = default;
        NvPredictorPrivate(ModelPtr model) : model_(model)
        {
            model_ = model;
        }
        ~NvPredictorPrivate();
        ModelPtr model_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<IExecutionContext> context_;
        std::unique_ptr<cudaStream_t> stream_;
        Status Init();

        BufSurfWrapperPtr RequestBuffer(int idx = 0)
        {
            BufSurfWrapperPtr buf = pools_[idx]->GetBufSurfaceWrapper();
            return buf;
        }

        BufSurfWrapperPtr RequestInDevBuffer(int idx = 0)
        {
            BufSurfWrapperPtr buf = pools_dev_in_[idx]->GetBufSurfaceWrapper();
            return buf;
        }

        BufSurfWrapperPtr RequestOutDevBuffer(int idx = 0)
        {
            BufSurfWrapperPtr buf = pools_dev_out_[idx]->GetBufSurfaceWrapper();
            return buf;
        }

    private:
        std::vector<BufPool *> pools_;
        std::vector<BufPool *> pools_dev_in_;
        std::vector<BufPool *> pools_dev_out_;
    };
}


static int CreatePool(ModelPtr model, BufPool *pool, BufSurfaceMemType mem_type, int idx, int dir, int block_count) {
    // 解析model，获取必要结构
    DataLayout layout;
    if (dir == 0){
        layout =  model->InputLayout(idx);
    } else {
        layout = model->OutputLayout(idx);
    }
    auto dtype = layout.dtype;
    auto order = layout.order;
    int data_size = 0;
    if (dtype == gddeploy::DataType::INT8 || dtype == gddeploy::DataType::UINT8){
        data_size = sizeof(uint8_t);
    }else if (dtype == gddeploy::DataType::FLOAT16 || dtype == gddeploy::DataType::UINT16 || dtype == gddeploy::DataType::INT16){
        data_size = sizeof(uint16_t);
    }else if (dtype == gddeploy::DataType::FLOAT32 || dtype == gddeploy::DataType::INT32){
        data_size = sizeof(uint32_t);
    }

    int model_h, model_w, model_c, model_b;
    gddeploy::Shape shape;
    if (dir == 0){
        shape = model->InputShape(idx);
    } else {
        shape = model->OutputShape(idx);
    }
    
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
    create_params.bytes_per_pix = data_size;
    create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;

    if (pool->CreatePool(&create_params, block_count) < 0) {
        return -1;
    }
    return 0;
}

Status NvPredictorPrivate::Init()
{
    engine_ = gddeploy::any_cast<std::shared_ptr<nvinfer1::ICudaEngine>>(model_->GetModel());
    context_ = std::unique_ptr<IExecutionContext>(engine_->createExecutionContext());
    stream_ = std::make_unique<cudaStream_t>();
    CHECK_AND_RET(cudaStreamCreate(stream_.get()), Status::INVALID_MODEL);
    CHECK_AND_RET(context_ == nullptr, Status::INVALID_MODEL);

    size_t o_num = model_->OutputNum();
    size_t i_num = model_->InputNum();

    // 分配设备显存池
    for (int i = 0; i < i_num; i++){
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_NVIDIA, i, 0, 3);
        pools_dev_in_.emplace_back(pool);
    }

    for (int i = 0; i < o_num; i++){
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_NVIDIA, i, 1, 3);
        pools_dev_out_.emplace_back(pool);
    }

    for (int i = 0; i < o_num; i++){
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_SYSTEM, i, 1, 3);
        pools_.emplace_back(pool);
    }

    return Status::SUCCESS;
}

NvPredictorPrivate::~NvPredictorPrivate()
{
    if (stream_ != nullptr)
    {
        printf("destroy stream");
        cudaStreamDestroy(*(stream_.get()));
    }
}

Status NvPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept
{
    priv_ = std::make_shared<NvPredictorPrivate>(model);
    model_ = model;
    if (priv_->Init() != Status::SUCCESS)
        return gddeploy::Status::INVALID_MODEL;
    return gddeploy::Status::SUCCESS;
}

Status NvPredictor::Process(PackagePtr pack) noexcept
{
    void *buffers_[256] = {NULL};

    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    BufSurface *surface = in_buf->GetBufSurface();

    BufSurfaceMemType mem_type = surface->mem_type;
    int batch_size = surface->batch_size;

    BufSurfWrapperPtr in_ptr = nullptr;
    
    if (mem_type == GDDEPLOY_BUF_MEM_NVIDIA){
        BufSurface *src_surf = new BufSurface;
        src_surf->mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
        src_surf->batch_size = batch_size;
        src_surf->num_filled = 1;
        src_surf->is_contiguous = 0;    // AVFrame的两个plane地址不一定连续
        
        src_surf->surface_list = new BufSurfaceParams[batch_size];
        
        int batch_idx = 0;
        for (int i = 0; i < batch_size; i++){
            BufSurfaceParams *src_param = in_buf->GetSurfaceParams(i);
          
            for (int i = 0; i < surface->batch_size; i++){
                src_surf->surface_list[batch_idx++] = *(src_param+i);
            }
        }
        in_ptr = std::make_shared<BufSurfaceWrapper>(src_surf, false);
    } else {    // 来着CPU，需要拷贝
        in_ptr = priv_->RequestInDevBuffer();   //从内存池获取
        int batch_idx = 0;
        
        for (auto &data : pack->data){
            for (int i = 0; i < surface->batch_size; i++){
                cudaMemcpy(in_ptr->GetData(0, batch_idx+i), in_buf->GetData(0, i), in_buf->GetSurfaceParams(i)->data_size, cudaMemcpyHostToDevice);
            }
             batch_idx += surface->batch_size;
             // TODO:这里如果超过in_ptr的batchsize，需要阶段再分配一块显存继续存储；
             // 比如ocr第二模型，拆分了多个小目标，很容易就生成超过batch大小的数量
        }
    }
    buffers_[0] = in_ptr->GetData(0, 0);

    // 从预设显存池获取空间赋值
    std::vector<BufSurfWrapperPtr> out_dev_ptrs;
    for (size_t i = 0; i < model_->OutputNum(); i++){
        BufSurfWrapperPtr out_dev_ptr = priv_->RequestOutDevBuffer(i);
        buffers_[i + model_->InputNum()] = out_dev_ptr->GetData(0, 0);

        out_dev_ptrs.emplace_back(out_dev_ptr);
    }

    // 根据batch size重写设置一下input dims
    gddeploy::Shape input_shape = model_->InputShape(0);
    Dims dims;
    for (int i = 0; i < input_shape.Size(); i++)
    {
        // printf("input_shape[%d] = %d\n", i, input_shape[i]);
        dims.d[i] = input_shape[i];
    }
    dims.d[0] = batch_size;
    dims.nbDims = input_shape.Size();
    CHECK_AND_RET(priv_->context_->setBindingDimensions(0, dims) == 0, Status::INVALID_MODEL);

    // inference    
    priv_->context_->enqueueV2(buffers_, nullptr, nullptr);
    // cudaStreamSynchronize(*(priv_->stream_.get()));

    std::vector<BufSurfWrapperPtr> out_bufs;
    for (size_t i = 0; i < model_->OutputNum(); i++)
    {
        BufSurfWrapperPtr buf = priv_->RequestBuffer(i);
        auto output_shape = priv_->model_->OutputShape(i);
        auto data_count = output_shape.BatchDataCount();
        CHECK_AND_RET(cudaMemcpy((void *)buf->GetData(0, 0), buffers_[i + model_->InputNum()], data_count * sizeof(float) * surface->batch_size, cudaMemcpyDeviceToHost) != cudaSuccess, Status::ERROR_MEMORY);
        out_bufs.push_back(buf);
    }
    pack->predict_io->Set(std::move(out_bufs));

    if (mem_type == GDDEPLOY_BUF_MEM_NVIDIA){
        auto surf = in_ptr->GetBufSurface();
        delete surf->surface_list;
        delete surf;
    }

    return gddeploy::Status::SUCCESS;
}

// REGISTER_PREDICTOR_CREATOR("ort", "cpu", NvPredictorCreator)