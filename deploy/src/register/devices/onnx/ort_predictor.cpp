#include "ort_predictor.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface.h"

#include <onnxruntime_cxx_api.h>

using namespace gddeploy;
// 通过ort_model的priv_获取到session
// 拷贝输入PackPtr里面的数据到session  Ort::Value input_tensor = Ort::Value::CreateTensor
// session推理，得到的结果拷贝到输出的output PackPtr

namespace gddeploy
{
class OrtPredictorPrivate{
public:
    OrtPredictorPrivate() = default;
    OrtPredictorPrivate(ModelPtr model):model_(model){
        session_ = gddeploy::any_cast<std::shared_ptr<Ort::Session>>(model->GetModel());
        Ort::AllocatorWithDefaultOptions allocator;
        num_inputs_ = session_->GetInputCount();
        input_node_names_.resize(num_inputs_);
        for (size_t i = 0; i < num_inputs_; ++i)
        {
            input_node_names_[i] = session_->GetInputName(i, allocator);
        }
        num_outputs_ = session_->GetOutputCount();
        output_node_names_.resize(num_outputs_);
        for (unsigned int i = 0; i < num_outputs_; ++i)
        {
            output_node_names_[i] = session_->GetOutputName(i, allocator);
        }
    }
    OrtPredictorPrivate(std::shared_ptr<Ort::Session> sess):session_(sess){
        Ort::AllocatorWithDefaultOptions allocator;
        num_inputs_ = session_->GetInputCount();
        input_node_names_.resize(num_inputs_);
        for (size_t i = 0; i < num_inputs_; ++i)
        {
            input_node_names_[i] = session_->GetInputName(i, allocator);
        }
        num_outputs_ = session_->GetOutputCount();
        output_node_names_.resize(num_outputs_);
        for (unsigned int i = 0; i < num_outputs_; ++i)
        {
            output_node_names_[i] = session_->GetOutputName(i, allocator);
        }

    }
    std::shared_ptr<Ort::Session> GetSession() { return session_; }

    BufSurfWrapperPtr RequestBuffer(int idx){
        return pools_[idx]->GetBufSurfaceWrapper();
    }

    Ort::Env env_;
    std::shared_ptr<Ort::Session> session_;
    // Ort::MemoryInfo memory_info_handler_;

    int Init(std::string config);

    size_t num_inputs_;
    size_t num_outputs_;
    std::vector<const char *> input_node_names_;
    std::vector<const char *> output_node_names_;

    ModelPtr model_;
private:
    std::vector<BufPool*> pools_;
};
}

int OrtPredictorPrivate::Init(std::string config)
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

        int model_h = 0, model_w = 0, model_c = 0, model_b = 0;
        int data_num = 0;
        auto shape = model_->OutputShape(i_idx);
        if (shape.Size() == 4){
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
            data_num = model_h * model_w * model_c;
        } else if (shape.Size() == 2){
            model_b = shape[0];
            model_c = shape[1];
            data_num = model_c;
        }

        BufSurfaceCreateParams create_params;
        memset(&create_params, 0, sizeof(create_params));
        create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
        create_params.force_align_1 = 1;  // to meet mm's requirement
        create_params.device_id = 0;
        create_params.batch_size = model_b;
        create_params.size = data_num;
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

Status OrtPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<OrtPredictorPrivate>(model);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}


Status OrtPredictor::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    Status s = Status::SUCCESS;

    // // 1. 拷贝数据
    Ort::MemoryInfo memory_info_handler_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    gddeploy::Shape input_shape = model_->InputShape(0);

    std::vector<int64_t> input_node1_dims;
    for (int i = 0; i < input_shape.Size(); i++){
        input_node1_dims.emplace_back(input_shape[i]);
    }
    float * in_data_ptr = (float *)in_buf->GetHostData(0);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info_handler_, in_data_ptr,
                                           in_buf->GetSurfaceParams(0)->data_size/sizeof(float), 
                                           input_node1_dims.data(), input_node1_dims.size());
    
    
    
    // // 2. 推理
    std::vector<Ort::Value> output_tensors = priv_->session_->Run(Ort::RunOptions{nullptr}, 
                priv_->input_node_names_.data(), &input_tensor, 
                priv_->num_inputs_,  priv_->output_node_names_.data(), priv_->num_outputs_);


    // // 1. 后处理，画图
    std::vector<BufSurfWrapperPtr> out_bufs;
    for (unsigned int i = 0; i < output_tensors.size(); ++i){
        auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();

        BufSurfWrapperPtr buf = priv_->RequestBuffer(i);
    
        auto output_shape = priv_->model_->OutputShape(i);
        auto data_count  = output_shape.BatchDataCount();
        void *data = output_tensors.at(i).GetTensorMutableData<float>();
        memcpy(buf->GetData(0, 0), data, data_count * sizeof(float));

        out_bufs.emplace_back(buf);
    }
    pack->predict_io->Set(out_bufs);

    return gddeploy::Status::SUCCESS; 
}


// REGISTER_PREDICTOR_CREATOR("ort", "cpu", OrtPredictorCreator)