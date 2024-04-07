#include "mnn_model.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string.h>
#include <vector>

#include "core/infer_server.h"

#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>


using namespace gddeploy;

namespace gddeploy
{
class MnnModelPrivate{
public:
    MnnModelPrivate(const std::string& model_path, const std::string& key){
        net_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
        if (nullptr == net_) {
            std::cout << "[MNN] Parse model fail!!!" << std::endl;
        }
        sess_config_.numThread = 4;
        sess_config_.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
        // backendConfig.precision =  MNN::PrecisionMode Precision_Normal; // static_cast<PrecisionMode>(Precision_Normal);
        sess_config_.backendConfig = &backendConfig;
        sess_ = std::shared_ptr<MNN::Session>(net_->createSession(sess_config_), [this](MNN::Session *session){
            this->net_->releaseSession(session);
        });

        SetInputShape();
        SetOutputShape();
    }
    MnnModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key){
        net_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(mem_ptr, mem_size));
        if (nullptr == net_) {
            std::cout << "[MNN] Parse model fail!!!" << std::endl;
        }
        sess_config_.numThread = 4;
        sess_config_.type      = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
        // backendConfig.precision =  MNN::PrecisionMode Precision_Normal; // static_cast<PrecisionMode>(Precision_Normal);
        sess_config_.backendConfig = &backendConfig;
        sess_ = std::shared_ptr<MNN::Session>(net_->createSession(sess_config_), [this](MNN::Session *session){
            this->net_->releaseSession(session);
        });

        SetInputShape();
        SetOutputShape();
    }

    int SetInputShape();
    int SetOutputShape();

    std::vector<std::string> GetInputNodeName() {return input_node_names_;}
    std::vector<std::vector<int>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<int> GetInputDataType() {return input_data_type_;}

    std::vector<std::string> GetOutputNodeName() {return output_node_names_;}
    std::vector<std::vector<int>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<int> GetOutputDataType() {return output_data_type_;}

    std::shared_ptr<MNN::Interpreter> GetModel() { return net_; }

private:
    std::vector<std::string> input_node_names_;
    std::vector<std::vector<int>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<int> input_data_type_;

    std::vector<std::string> output_node_names_;
    std::vector<std::vector<int>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<int> output_data_type_;

    std::shared_ptr<MNN::Interpreter> net_;
    MNN::ScheduleConfig sess_config_;
    std::shared_ptr<MNN::Session> sess_;

};
}

static DataType convertMnnDataType2MyDataType(int in){
    DataType out;

    switch (in)
    {
    case 1:
        out = DataType::INT8;
        break;
    case 4:
        out = DataType::FLOAT32;
        break;
    case 2:
        out = DataType::INT16;
        break;
    default:
        out = DataType::INVALID;
        break;
    }
    return out;
}

int MnnModelPrivate::SetInputShape()
{
    auto inputTensors = net_->getSessionInputAll(sess_.get());

    for (auto iter : inputTensors){
        input_node_names_.emplace_back(iter.first);
        auto input_tensor = iter.second;
        // 保存shape
        auto input_dims = input_tensor->shape();
        input_node_dims_.push_back(input_dims);

        // 保存类型
        auto data_type = input_tensor->elementSize();
        input_data_type_.push_back(data_type);

        // 保存Tensor Size
        size_t tensor_size = 1;
        for (unsigned int j = 0; j < input_dims.size(); ++j)
            tensor_size *= input_dims.at(j);

        input_tensor_sizes_.push_back(tensor_size);
    }

    return 0;
}

int MnnModelPrivate::SetOutputShape()
{
    auto outputTensors = net_->getSessionOutputAll(sess_.get());

    for (auto iter : outputTensors){
        output_node_names_.emplace_back(iter.first);
        auto input_tensor = iter.second;
        // 保存shape
        auto output_dims = input_tensor->shape();
        output_node_dims_.push_back(output_dims);

        // 保存类型
        auto data_type = input_tensor->elementSize();
        output_data_type_.push_back(data_type);

        // 保存Tensor Size
        size_t tensor_size = 1;
        for (unsigned int j = 0; j < output_dims.size(); ++j)
            tensor_size *= output_dims.at(j);

        output_tensor_sizes_.push_back(tensor_size);
    }
    return 0;
}

int MnnModel::Init(const std::string& model_path, const std::string& param)
{
    mnn_model_priv_ = std::make_shared<MnnModelPrivate>(model_path, param);

    auto input_dims = mnn_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims){
        std::vector<int64_t> in_dims;
        std::transform(dims.begin(), dims.end(), std::back_inserter(in_dims), [](int value){ return (int64_t)value; });
        inputs_shape_.push_back(Shape(in_dims));
    }
    
    auto input_data_type = mnn_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertMnnDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = mnn_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims){
        std::vector<int64_t> out_dims;
        std::transform(dims.begin(), dims.end(), std::back_inserter(out_dims), [](int value){ return (int64_t)value; });
        outputs_shape_.push_back(Shape(out_dims));
    }

    auto output_data_type = mnn_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertMnnDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int MnnModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    mnn_model_priv_ = std::make_shared<MnnModelPrivate>(mem_ptr, mem_size, param);

    auto input_dims = mnn_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims){
        std::vector<int64_t> in_dims;
        std::transform(dims.begin(), dims.end(), std::back_inserter(in_dims), [](int value){ return (int64_t)value; });
        inputs_shape_.push_back(Shape(in_dims));
    }
    
    auto input_data_type = mnn_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertMnnDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = mnn_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims){
        std::vector<int64_t> out_dims;
        std::transform(dims.begin(), dims.end(), std::back_inserter(out_dims), [](int value){ return (int64_t)value; });
        outputs_shape_.push_back(Shape(out_dims));
    }

    auto output_data_type = mnn_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertMnnDataType2MyDataType(dti);
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
const Shape& MnnModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& MnnModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int MnnModel::FixedOutputShape() noexcept
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
const DataLayout& MnnModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& MnnModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t MnnModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t MnnModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}


/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t MnnModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string MnnModel::GetKey() const noexcept
{
    return "Mnn";
}


any MnnModel::GetModel()
{
    return mnn_model_priv_->GetModel();
}


// REGISTER_MODEL_CREATOR("any", "any", MnnModelCreator) 