#include "cambricon_model.h"

#include <iostream>
#include <string.h>

#include "common/logger.h"
#include "core/infer_server.h"

#include "cnrt.h"

using namespace gddeploy;

namespace gddeploy
{
class Cambricon3xxModelPrivate{
public:
    Cambricon3xxModelPrivate(const std::string& model_path,const std::string& key);
    Cambricon3xxModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key);
    ~Cambricon3xxModelPrivate();

    int SetInputOuputInfo();

    std::vector<std::vector<int64_t>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<cnrtDataType_t> GetInputDataType() {return input_data_type_;}

    std::vector<std::vector<int64_t>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<cnrtDataType_t> GetOutputDataType() {return output_data_type_;}

    std::shared_ptr<void> GetModel() { return std::shared_ptr<void>(&function_); }

    std::string GetMD5() { return md5_;  }

private:
    cnrtFunction_t function_;
    cnrtModel_t model_{nullptr};
    std::string func_name_ = "subnet0";

    cnrtDataType_t* input_dtypes_ = nullptr;
    cnrtDataType_t* output_dtypes_ = nullptr;

    std::vector<const char *> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<cnrtDataType_t> input_data_type_;

    std::vector<const char *> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<cnrtDataType_t> output_data_type_;

    std::string md5_;
};
}

static DataType convertCambricon3xxDataType2MyDataType(cnrtDataType_t in){
    DataType out;

    switch (in)
    {
    // case BM_INT8:
    //     out = DataType::INT8;
    //     break;
    // case BM_UINT8:
    //     out = DataType::UINT8;
    //     break;
    // case BM_FLOAT32:
    //     out = DataType::FLOAT32;
    //     break;
    // case BM_FLOAT16:
    //     out = DataType::FLOAT16;
    //     break;
    // case BM_UINT16:
    //     out = DataType::UINT16;
    //     break;
    // case BM_INT16:
    //     out = DataType::INT16;
    //     break;
    // case BM_INT32:
    // case BM_UINT32:
    //     out = DataType::INT32;
    //     break;
    // default:
    //     out = DataType::INVALID;
    //     break;
    }
    return out;
}

#define CHECK_CNRT_RET(err_code, msg)                                                                  \
  if (CNRT_RET_SUCCESS != err_code) {                                                                  \
    printf(Exception::MEMORY, std::string(msg) + " error code: " + std::to_string(err_code)); \
  }

Cambricon3xxModelPrivate::Cambricon3xxModelPrivate(const std::string& model_path, const std::string& key)
{
    cnrtRet_t error_code = cnrtLoadModel(&model_, model_path.c_str());

    error_code = cnrtCreateFunction(&function_);
    // printf(error_code, "[EasyDK InferServer] [Model] Create function failed.", false);
    error_code = cnrtExtractFunction(&function_, model_, func_name_.c_str());
    // printf(error_code, "[EasyDK InferServer] [Model] Extract function failed.", false);
    int model_parallelism;
    error_code = cnrtQueryModelParallelism(model_, &model_parallelism);

    SetInputOuputInfo();

    md5_ = key;
}

Cambricon3xxModelPrivate::Cambricon3xxModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key)
{
    cnrtRet_t error_code = cnrtLoadModelFromMem(&model_, reinterpret_cast<char*>(mem_ptr));
    error_code = cnrtCreateFunction(&function_);
    // printf(error_code, "[EasyDK InferServer] [Model] Create function failed.", false);
    error_code = cnrtExtractFunction(&function_, model_, func_name_.c_str());
    // printf(error_code, "[EasyDK InferServer] [Model] Extract function failed.", false);
    int model_parallelism;
    error_code = cnrtQueryModelParallelism(model_, &model_parallelism);

    SetInputOuputInfo();

    md5_ = key;
}

Cambricon3xxModelPrivate::~Cambricon3xxModelPrivate()
{
    cnrtRet_t error_code = cnrtDestroyFunction(function_);
    error_code = cnrtUnloadModel(model_);
}

int Cambricon3xxModelPrivate::SetInputOuputInfo()
{
    cnrtRet_t error_code;
    int64_t* input_sizes = nullptr;
    int input_num = 0;
    error_code = cnrtGetInputDataSize(&input_sizes, &input_num, function_);

    int64_t* output_sizes = nullptr;
    int output_num = 0;
    error_code = cnrtGetOutputDataSize(&output_sizes, &output_num, function_);

    // input info
    int* input_dim_values = nullptr;
    int dim_num = 0;
    for (int i_idx = 0; i_idx < input_num; ++i_idx) {
        error_code = cnrtGetInputDataShape(&input_dim_values, &dim_num, i_idx, function_);
        // printf(error_code, "[EasyDK InferServer] [Model] Get input data size failed.", false);
        // nhwc shape
        std::vector<int64_t> i_shape;
        std::transform(input_dim_values, input_dim_values + dim_num, std::back_inserter(i_shape),
                    [](int v) -> Shape::value_type { return v; });
        input_node_dims_.emplace_back(i_shape);

        size_t tensor_size = 1;
        for (auto dim : i_shape){
            tensor_size *= dim;
        }
        input_tensor_sizes_.emplace_back(tensor_size);

        free(input_dim_values);

        error_code = cnrtGetInputDataType(&input_dtypes_, &input_num, function_);
        input_data_type_.emplace_back(input_dtypes_[i_idx]);
    }
    // output info
    int* output_dim_values = nullptr;
    dim_num = 0;
    error_code = cnrtGetInputDataType(&output_dtypes_, &output_num, function_);
    for (int i_idx = 0; i_idx < input_num; ++i_idx) {
        error_code = cnrtGetOutputDataShape(&output_dim_values, &dim_num, i_idx, function_);
        // printf(error_code, "[EasyDK InferServer] [Model] Get input data size failed.", false);
        // nhwc shape
        std::vector<int64_t> o_shape;
        std::transform(output_dim_values, output_dim_values + dim_num, std::back_inserter(o_shape),
                    [](int v) -> Shape::value_type { return v; });
        output_node_dims_.emplace_back(o_shape);

        size_t tensor_size = 1;
        for (auto dim : o_shape){
            tensor_size *= dim;
        }
        output_tensor_sizes_.emplace_back(tensor_size);

        free(output_dim_values);
           
        
        output_data_type_.emplace_back(output_dtypes_[i_idx]);
    }

    return 0;
}

int Cambricon3xxModel::Init(const std::string& model_path, const std::string& param)
{
    cambricon3_model_priv_ = std::make_shared<Cambricon3xxModelPrivate>(model_path, param);

    auto input_dims = cambricon3_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims){

        inputs_shape_.push_back(Shape(dims));
    }
    
    auto input_data_type = cambricon3_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertCambricon3xxDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = cambricon3_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = cambricon3_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertCambricon3xxDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int Cambricon3xxModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    cambricon3_model_priv_ = std::make_shared<Cambricon3xxModelPrivate>(mem_ptr, mem_size, param);

    auto input_dims = cambricon3_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = cambricon3_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertCambricon3xxDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = cambricon3_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = cambricon3_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertCambricon3xxDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
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
const Shape& Cambricon3xxModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& Cambricon3xxModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int Cambricon3xxModel::FixedOutputShape() noexcept
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
const DataLayout& Cambricon3xxModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& Cambricon3xxModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t Cambricon3xxModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t Cambricon3xxModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}


/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t Cambricon3xxModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string Cambricon3xxModel::GetKey() const noexcept
{
    return cambricon3_model_priv_->GetMD5();
}


any Cambricon3xxModel::GetModel()
{
    return cambricon3_model_priv_->GetModel();
}


// REGISTER_MODEL_CREATOR("any", "any", Cambricon3xxModelCreator) 
