#include "intel_model.h"

#include <iostream>
#include <string.h>
#include "openvino/openvino.hpp"
#include "ie/ie_core.hpp"

#include "core/infer_server.h"
#include "intel_common.h"
using namespace gddeploy;

namespace gddeploy
{
class IntelModelPrivate{
public:
    IntelModelPrivate(const std::string& model_path, const std::string& key){

        SetInputOuputInfo();

        md5_ = key;
    }
    IntelModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key){
        std::string file_size_str(reinterpret_cast<const char*>(mem_ptr), 4);

        int file_size;
        memcpy(&file_size, file_size_str.c_str(), 4);

        std::string strModel((char *)mem_ptr+4, (char *)mem_ptr+file_size+4);
        std::string weightData((char *)mem_ptr+file_size+4, (char *)mem_ptr+mem_size);

        raw_model_ = strModel;
        raw_weight_ = weightData;

        model_ = core_.read_model(raw_model_, ov::Tensor(ov::element::u8, {raw_weight_.size()}, (char *)raw_weight_.c_str()));

        compiled_model_cpu_ = std::make_shared<ov::CompiledModel>();
        *compiled_model_cpu_ = core_.compile_model(model_, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        uint32_t nireq_cpu = compiled_model_cpu_->get_property(ov::optimal_number_of_infer_requests);

        compiled_model_gpu_ = std::make_shared<ov::CompiledModel>();
        *compiled_model_gpu_ = core_.compile_model(model_, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        uint32_t nireq_gpu = compiled_model_gpu_->get_property(ov::optimal_number_of_infer_requests);

        // printf("nireq cpu: %d, gpu: %d\n", nireq_cpu, nireq_gpu);

        multi_compiled_model_ = std::make_shared<MultiCompiledModel>();
        multi_compiled_model_->cpu_request_num = nireq_cpu;
        multi_compiled_model_->gpu_request_num = nireq_gpu;
        multi_compiled_model_->compiled_model_cpu_ = compiled_model_cpu_;
        multi_compiled_model_->compiled_model_gpu_ = compiled_model_gpu_;

        // ov::hint::num_requests(4);
        SetInputOuputInfo();

        // InferenceEngine::Core ie_;
        // ov::runtime::Core::get_available_devices()
        // std::vector<std::string> devices = ie_.GetAvailableDevices();
        // for (auto& device : devices) {
        //     std::cout << "device: " << device << std::endl;
        // }

        md5_ = key;
    }

    int SetInputOuputInfo();

    std::vector<const char *> GetInputNodeName() {return input_node_names_;}
    std::vector<std::vector<int64_t>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<ov::element::Type_t> GetInputDataType() {return input_data_type_;}

    std::vector<const char *> GetOutputNodeName() {return output_node_names_;}
    std::vector<std::vector<int64_t>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<ov::element::Type_t> GetOutputDataType() {return output_data_type_;}

    std::shared_ptr<MultiCompiledModel> GetModel() { 
        // std::shared_ptr<ov::Model> tmp = std::dynamic_pointer_cast<ov::Model>(model_); 
        // return gddeploy::any_cast<std::shared_ptr<void>>(model_);
        return multi_compiled_model_;
    }

    std::string GetMD5() { return md5_;  }

private:
    
    std::shared_ptr<MultiCompiledModel> multi_compiled_model_;

    std::string raw_model_;
    std::string raw_weight_;

    std::vector<const char *> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<ov::element::Type_t> input_data_type_;

    std::vector<const char *> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<ov::element::Type_t> output_data_type_;

    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    std::shared_ptr<ov::CompiledModel> compiled_model_cpu_;
    std::shared_ptr<ov::CompiledModel> compiled_model_gpu_;

    std::string md5_;
};
}

static DataType convertIntelDataType2MyDataType(ov::element::Type_t in){
    DataType out;

    switch (in)
    {
    case ov::element::Type_t::i8:
        out = DataType::INT8;
        break;
    case ov::element::Type_t::u8:
        out = DataType::UINT8;
        break;
    case ov::element::Type_t::f32:
        out = DataType::FLOAT32;
        break;
    case ov::element::Type_t::f16:
        out = DataType::FLOAT16;
        break;
    case ov::element::Type_t::u16:
        out = DataType::UINT16;
        break;
    case ov::element::Type_t::i16:
        out = DataType::INT16;
        break;
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        out = DataType::INT32;
        break;
    default:
        out = DataType::INVALID;
        break;
    }
    return out;
}


int IntelModelPrivate::SetInputOuputInfo()
{
    std::vector<ov::Output<ov::Node>> inputs = model_->inputs();
    int input_num = inputs.size();

    std::vector<ov::Output<ov::Node>> outputs = model_->outputs();
    int output_num = outputs.size();

    // ov::InferRequest infer_request = compiled_model_->create_infer_request();

    // input info
    for (int i_idx = 0; i_idx < input_num; ++i_idx) {
        const ov::Shape input_shape = inputs[i_idx].get_shape();

        std::vector<int64_t> i_shape;
        for (int i = 0; i < input_shape.size(); i++)
            i_shape.emplace_back(input_shape[i]);
        input_node_dims_.emplace_back(i_shape);

        size_t tensor_size = 1;
        for (auto dim : i_shape){
            tensor_size *= dim;
        }
        input_tensor_sizes_.emplace_back(tensor_size);

        auto type = inputs[i_idx].get_element_type();

        // ov::Tensor input_tensor = infer_request.get_input_tensor();

        input_data_type_.emplace_back(type);
    }

    // output info
    for (int i_idx = 0; i_idx < output_num; ++i_idx) {
        const ov::Shape output_shape = outputs[i_idx].get_shape();

        std::vector<int64_t> o_shape;
        for (int i = 0; i < output_shape.size(); i++)
            o_shape.emplace_back(output_shape[i]);
        output_node_dims_.emplace_back(o_shape);

        size_t tensor_size = 1;
        for (auto dim : o_shape){
            tensor_size *= dim;
        }
        output_tensor_sizes_.emplace_back(tensor_size);
        
        // ov::Tensor output_tensor = infer_request.get_output_tensor(i_idx);
        auto type = outputs[i_idx].get_element_type();
        output_data_type_.emplace_back(type);
    }

    return 0;
}

int IntelModel::Init(const std::string& model_path, const std::string& param)
{
    intel_model_priv_ = std::make_shared<IntelModelPrivate>(model_path, param);

    auto input_dims = intel_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = intel_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertIntelDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = intel_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = intel_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertIntelDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int IntelModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    intel_model_priv_ = std::make_shared<IntelModelPrivate>(mem_ptr, mem_size, param);

    auto input_dims = intel_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = intel_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertIntelDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = intel_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = intel_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertIntelDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }
    
    return 0;
}

/**
 * @brief Get input shape
 *
 * @param index index of input
 * @return const Shape& shape of specified input
 */
const Shape& IntelModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& IntelModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int IntelModel::FixedOutputShape() noexcept
{
    if (outputs_shape_.size() == 0){
        return -1;
    }

    return 0;
}


/**
 * @brief Get input layout on MLU
 *
 * @param index index of input
 * @return const DataLayout& data layout of specified input
 */
const DataLayout& IntelModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& IntelModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t IntelModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t IntelModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}

std::vector<float> IntelModel::InputScale() const noexcept
{
    return std::vector<float>{};
}

std::vector<float> IntelModel::OutputScale() const noexcept
{
    return std::vector<float>{};
}

/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t IntelModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string IntelModel::GetKey() const noexcept
{
    return intel_model_priv_->GetMD5();
}


any IntelModel::GetModel()
{
    return intel_model_priv_->GetModel();
}


// REGISTER_MODEL_CREATOR("any", "any", IntelModelCreator) 