#include "ort_model.h"

#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <string.h>

#include "core/infer_server.h"
using namespace gddeploy;

namespace gddeploy
{
class OrtModelPrivate{
public:
    OrtModelPrivate(const std::string& model_path, const std::string& key){
        session_ = std::shared_ptr<Ort::Session>(new Ort::Session(env_, model_path.c_str(), Ort::SessionOptions{nullptr}));
        SetInputShape();
        SetOutputShape();
    }
    OrtModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key){
        session_ = std::shared_ptr<Ort::Session>(new Ort::Session(env_, mem_ptr, mem_size,  Ort::SessionOptions{nullptr}));
        SetInputShape();
        SetOutputShape();
    }

    int SetInputShape();
    int SetOutputShape();

    std::vector<const char *> GetInputNodeName() {return input_node_names_;}
    std::vector<std::vector<int64_t>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<ONNXTensorElementDataType> GetInputDataType() {return input_data_type_;}

    std::vector<const char *> GetOutputNodeName() {return output_node_names_;}
    std::vector<std::vector<int64_t>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<ONNXTensorElementDataType> GetOutputDataType() {return output_data_type_;}

    std::shared_ptr<Ort::Session> GetModel() { return session_; }

private:
    std::vector<const char *> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<ONNXTensorElementDataType> input_data_type_;

    std::vector<const char *> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<ONNXTensorElementDataType> output_data_type_;

    Ort::Env env_;
    std::shared_ptr<Ort::Session> session_;

};
}

static DataType convertOrtDataType2MyDataType(ONNXTensorElementDataType in){
    DataType out;

    switch (in)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        out = DataType::INT8;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        out = DataType::UINT8;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        out = DataType::FLOAT32;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        out = DataType::FLOAT16;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        out = DataType::UINT16;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        out = DataType::INT16;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        out = DataType::INT32;
        break;
    default:
        out = DataType::INVALID;
        break;
    }
    return out;
}

int OrtModelPrivate::SetInputShape()
{
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    input_node_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i)
    {
        input_node_names_[i] = session_->GetInputName(i, allocator);

        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = tensor_info.GetShape();
        
        input_node_dims_.push_back(input_dims);
        input_data_type_.push_back(tensor_info.GetElementType());

        size_t tensor_size = 1;
        for (unsigned int j = 0; j < input_dims.size(); ++j)
            tensor_size *= input_dims.at(j);

        input_tensor_sizes_.push_back(tensor_size);
    }

    return 0;
}

int OrtModelPrivate::SetOutputShape()
{
    Ort::AllocatorWithDefaultOptions allocator;
    int num_outputs = session_->GetOutputCount();
    output_node_names_.resize(num_outputs);
    for (unsigned int i = 0; i < num_outputs; ++i)
    {
        output_node_names_[i] = session_->GetOutputName(i, allocator);
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims_.push_back(output_dims);
        
        output_data_type_.push_back(output_tensor_info.GetElementType());
    }

    return 0;
}

int OrtModel::Init(const std::string& model_path, const std::string& param)
{
    ort_model_priv_ = std::make_shared<OrtModelPrivate>(model_path, param);

    auto input_dims = ort_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = ort_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertOrtDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = ort_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = ort_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertOrtDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int OrtModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    ort_model_priv_ = std::make_shared<OrtModelPrivate>(mem_ptr, mem_size, param);

    auto input_dims = ort_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = ort_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertOrtDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = ort_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = ort_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertOrtDataType2MyDataType(dti);
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
const Shape& OrtModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& OrtModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int OrtModel::FixedOutputShape() noexcept
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
const DataLayout& OrtModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& OrtModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t OrtModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t OrtModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}


/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t OrtModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string OrtModel::GetKey() const noexcept
{
    return "Ort";
}


any OrtModel::GetModel()
{
    return ort_model_priv_->GetModel();
}


// REGISTER_MODEL_CREATOR("any", "any", OrtModelCreator) 