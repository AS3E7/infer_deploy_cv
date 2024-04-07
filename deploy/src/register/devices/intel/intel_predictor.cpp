#include "intel_predictor.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface.h"
#include "openvino/openvino.hpp"
#include "intel_common.h"

using namespace gddeploy;
// 通过ort_model的priv_获取到session
// 拷贝输入PackPtr里面的数据到session  Intel::Value input_tensor = Intel::Value::CreateTensor
// session推理，得到的结果拷贝到输出的output PackPtr


static ov::element::Type_t convertMyDataType2IntelDataType(gddeploy::DataType in){
    ov::element::Type_t out;

    switch (in)
    {
    case DataType::INT8:
        out = ov::element::Type_t::i8;
        break;
    case DataType::UINT8:
        out = ov::element::Type_t::u8;
        break;
    case DataType::FLOAT32:
        out = ov::element::Type_t::f32;
        break;
    case DataType::FLOAT16:
        out = ov::element::Type_t::f16;
        break;
    case DataType::UINT16:
        out = ov::element::Type_t::u16;
        break;
    case DataType::INT16:
        out = ov::element::Type_t::i16;
        break;
    case DataType::INT32:
        out = ov::element::Type_t::i32; //ov::element::Type_t::u32
        break;
    default:
        out = ov::element::Type_t::undefined;
        break;
    }
    return out;
}

namespace gddeploy
{
class IntelPredictorPrivate{
public:
    IntelPredictorPrivate() = default;
    IntelPredictorPrivate(ModelPtr model):model_(model){
        std::shared_ptr<MultiCompiledModel> multi_compiled_model = gddeploy::any_cast<std::shared_ptr<MultiCompiledModel>>(model->GetModel());
        if (multi_compiled_model->gpu_request_num){
            compiled_model_ = multi_compiled_model->compiled_model_gpu_;
            multi_compiled_model->gpu_request_num--;
            // printf("gpu_request_num: %d\n", multi_compiled_model->gpu_request_num);
        } else if (multi_compiled_model->cpu_request_num - 1 > 0){
            compiled_model_ = multi_compiled_model->compiled_model_cpu_;
            multi_compiled_model->cpu_request_num--;
            // printf("cpu_request_num: %d\n", multi_compiled_model->cpu_request_num);
        } else{
            compiled_model_ = multi_compiled_model->compiled_model_cpu_;
            // printf("no request num\n");
        }

        // ov_model_ = gddeploy::any_cast<std::shared_ptr<ov::Model>>(model->GetModel());

        // compiled_model_ = std::make_shared<ov::CompiledModel>();
        // *compiled_model_ = core_.compile_model(ov_model_, "GPU", {{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}});

        auto inputs = compiled_model_->inputs();
        num_inputs_ = inputs.size();
        for (size_t i = 0; i < num_inputs_; ++i)
        {
            input_node_names_.emplace_back(inputs[i].get_any_name());
            input_shape_.emplace_back(inputs[i].get_shape());

            auto input_data_layout = model_->InputLayout(i);
            input_type_ = convertMyDataType2IntelDataType(input_data_layout.dtype);
        }

        auto outputs = compiled_model_->outputs();
        num_outputs_ = outputs.size();
        for (unsigned int i = 0; i < num_outputs_; ++i)
        {
            output_node_names_.emplace_back(outputs[i].get_any_name());
            output_shape_.emplace_back(outputs[i].get_shape());

            auto output_data_layout = model_->OutputLayout(i);
            output_type_ = convertMyDataType2IntelDataType(output_data_layout.dtype);
        }
    }

    BufSurfWrapperPtr RequestBuffer(int idx){
        return pools_[idx]->GetBufSurfaceWrapper();
    }

    ov::Core core_; 
    std::shared_ptr<ov::Model> ov_model_;
    std::shared_ptr<ov::CompiledModel> compiled_model_;

    int Init(std::string config);

    size_t num_inputs_;
    size_t num_outputs_;
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;

    std::vector<ov::Shape> input_shape_;
    std::vector<ov::Shape> output_shape_;

    ov::element::Type_t input_type_;
    ov::element::Type_t output_type_;

    ModelPtr model_;
    ov::InferRequest infer_request_;
private:
    std::vector<BufPool*> pools_;
};
}

int IntelPredictorPrivate::Init(std::string config)
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
        if (pool->CreatePool(&create_params, 6) < 0) {
            return -1;
        }
        pools_.emplace_back(pool);
    }

    infer_request_ = compiled_model_->create_infer_request();

    return 0;
}

Status IntelPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<IntelPredictorPrivate>(model);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}


Status IntelPredictor::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    Status s = Status::SUCCESS;

    // // 1. 拷贝数据
    auto input_tensor = ov::Tensor(priv_->input_type_, priv_->input_shape_[0], in_buf->GetData(0));
    priv_->infer_request_.set_tensor(priv_->input_node_names_[0], input_tensor);
    // ov::Tensor input_tensor = priv_->infer_request_.get_input_tensor();
    // float* input_blob_data_ptrs = input_tensor.data<float>();

    // auto input_shape = priv_->model_->InputShape(0);
    // auto data_count  = input_shape.BatchDataCount();
    // float*input_data = (float*)in_buf->GetData(0);
    // memcpy((uint8_t *)input_blob_data_ptrs, input_data, data_count * sizeof(float));

    // 2. 设置输出buf
    std::vector<BufSurfWrapperPtr> out_bufs;
    for (unsigned int i = 0; i < priv_->num_outputs_; ++i){
        BufSurfWrapperPtr buf = priv_->RequestBuffer(i);
        out_bufs.emplace_back(buf);

        auto output_tensor = ov::Tensor(priv_->output_type_, priv_->output_shape_[i], buf->GetData(0, 0));
        priv_->infer_request_.set_tensor(priv_->output_node_names_[i], output_tensor);
    }
    
    std::condition_variable cv;
    std::mutex mutex;
    int status = 0;
    
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex);
        if (status == 1) {
            cv.wait(lock);
            break;
        }
        // // 3. 推理
        
        if (status == 0){
            // auto t0 = std::chrono::high_resolution_clock::now();
            priv_->infer_request_.set_callback([&, &mutex, &cv](std::exception_ptr ex) {
                std::unique_lock<std::mutex> lock(mutex);
                // InferResult result;
                // postprocess(output_blob_data_ptrs, result);

                // observer_->Response(frame_id, &result, user_data);
                cv.notify_one();
                // auto t1 = std::chrono::high_resolution_clock::now();
                // printf("!!!!!intel predictor time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            });

            priv_->infer_request_.start_async();
            // priv_->infer_request_.wait();
            status = 1;
        }
    }
    
    // priv_->infer_request_.infer();
    
    // priv_->infer_request_.start_async();
    // priv_->infer_request_.wait();

    // std::vector<BufSurfWrapperPtr> out_bufs;
    // for (unsigned int i = 0; i < priv_->num_outputs_; ++i){
    //     BufSurfWrapperPtr buf = priv_->RequestBuffer(i);
    //     out_bufs.emplace_back(buf);

    //     auto output_shape = priv_->model_->OutputShape(i);
    //     auto data_count  = output_shape.BatchDataCount();
    //     ov::Tensor output_tensor = priv_->infer_request_.get_output_tensor(i);
    //     float *data = output_tensor.data<float>();

    //     memcpy(buf->GetData(0, 0), data, data_count * sizeof(float));
    // }
    

    // 4. out
    pack->predict_io->Set(std::move(out_bufs));

    return gddeploy::Status::SUCCESS; 
}