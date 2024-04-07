#include "ascend_model.h"
#include "ascend_common.h"

#include <iostream>
#include <string.h>

#include "common/logger.h"
#include "core/infer_server.h"

#include "acl/acl.h"

using namespace gddeploy;

namespace gddeploy
{
class AscendModelPrivate{
public:
    AscendModelPrivate() = default;
    int LoadModel(const std::string& model_path,const std::string& key);
    int LoadModel(void* mem_ptr,  size_t mem_size, const std::string& key);
    ~AscendModelPrivate();

    int SetInputOuputInfo();

    std::vector<const char *> GetInputNodeName() {return input_node_names_;}
    std::vector<std::vector<int64_t>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<aclDataType> GetInputDataType() {return input_data_type_;}
    std::vector<float> GetInputScale() {return input_scale_;}

    std::vector<const char *> GetOutputNodeName() {return output_node_names_;}
    std::vector<std::vector<int64_t>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<aclDataType> GetOutputDataType() {return output_data_type_;}
    std::vector<float> GetOutputScale() {return output_scale_;}

    std::shared_ptr<void> GetModel() { return std::shared_ptr<void>(&model_id_); }

    std::string GetMD5() { return md5_;  }

private:
    std::vector<const char *> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<aclDataType> input_data_type_;
    std::vector<float> input_scale_;

    std::vector<const char *> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<aclDataType> output_data_type_;
    std::vector<float> output_scale_;

    size_t model_dev_ptr_size_;
    size_t weight_dev_ptr_size_;

    void *model_dev_ptr_;
    void *weight_dev_ptr_;

    uint32_t model_id_;
    aclrtContext context_model_ = nullptr;
    std::shared_ptr<aclmdlDesc> model_desc_ = nullptr;

    bool is_loaded_ = false;

    std::string md5_;
};
}

static DataType convertAscendDataType2MyDataType(aclDataType in){
    DataType out;
    
    switch (in)
    {
    case ACL_INT8:
        out = DataType::INT8;
        break;
    case ACL_UINT8:
        out = DataType::UINT8;
        break;
    case ACL_FLOAT:
        out = DataType::FLOAT32;
        break;
    case ACL_FLOAT16:
        out = DataType::FLOAT16;
        break;
    case ACL_UINT16:
        out = DataType::UINT16;
        break;
    case ACL_INT16:
        out = DataType::INT16;
        break;
    case ACL_INT32:
    case ACL_UINT32:
        out = DataType::INT32;
        break;
    default:
        out = DataType::INVALID;
        break;
    }
    return out;
}

int AscendModelPrivate::LoadModel(const std::string& model_path, const std::string& key)
{
    APP_ERROR ret = aclmdlQuerySize(model_path.c_str(), &model_dev_ptr_size_, &weight_dev_ptr_size_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclmdlQuerySizeFromMem failed, ret[{}].", ret);
        return ret;
    }

    ret = aclrtMalloc(&model_dev_ptr_, model_dev_ptr_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclrtMalloc dev_ptr failed, ret[{}].", ret);
        return ret;
    }
    ret = aclrtMalloc(&weight_dev_ptr_, weight_dev_ptr_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclrtMalloc weight_ptr failed, ret[{}] ({})", ret, GetAppErrCodeInfo(ret));
        return ret;
    }
    ret = aclmdlLoadFromFileWithMem(model_path.c_str(), &model_id_, model_dev_ptr_, model_dev_ptr_size_,
        weight_dev_ptr_, weight_dev_ptr_size_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclmdlLoadFromMemWithMem failed, ret[{}].", ret);
        return ret;
    }
    ret = aclrtGetCurrentContext(&context_model_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclrtMalloc weight_ptr failed, ret[{}].", ret);
        return ret;
    }
    // get input and output size
    aclmdlDesc *model_desc = aclmdlCreateDesc();
    if (model_desc == nullptr) {
        GDDEPLOY_ERROR("aclmdlCreateDesc failed.");
        return ret;
    }
    ret = aclmdlGetDesc(model_desc, model_id_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclmdlGetDesc ret fail, ret:{}.", ret);
        return ret;
    }
    model_desc_.reset(model_desc, aclmdlDestroyDesc);
    SetInputOuputInfo();

    md5_ = key;

    is_loaded_ = true;
}

int AscendModelPrivate::LoadModel(void* mem_ptr,  size_t mem_size, const std::string& key)
{
    APP_ERROR ret = aclmdlQuerySizeFromMem(mem_ptr, mem_size, &model_dev_ptr_size_, &weight_dev_ptr_size_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclmdlQuerySizeFromMem failed, ret[{}].", ret);
        return ret;
    }

    ret = aclrtMalloc(&model_dev_ptr_, model_dev_ptr_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclrtMalloc dev_ptr failed, ret[{}].", ret);
        return ret;
    }
    ret = aclrtMalloc(&weight_dev_ptr_, weight_dev_ptr_size_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclrtMalloc weight_ptr failed, ret[{}] ({}}", ret, GetAppErrCodeInfo(ret));
        return ret;
    }
    ret = aclmdlLoadFromMemWithMem(mem_ptr, mem_size, &model_id_, model_dev_ptr_, model_dev_ptr_size_,
        weight_dev_ptr_, weight_dev_ptr_size_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclmdlLoadFromMemWithMem failed, ret[{}].", ret);
        return ret;
    }
    ret = aclrtGetCurrentContext(&context_model_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclrtMalloc weight_ptr failed, ret[{}].", ret);
        return ret;
    }
    // get input and output size
    aclmdlDesc *model_desc = aclmdlCreateDesc();
    if (model_desc == nullptr) {
        GDDEPLOY_ERROR("aclmdlCreateDesc failed.");
        return ret;
    }
    ret = aclmdlGetDesc(model_desc, model_id_);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("aclmdlGetDesc ret fail, ret:{}.");
        return ret;
    }
    model_desc_.reset(model_desc, aclmdlDestroyDesc);
    SetInputOuputInfo();

    md5_ = key;

    is_loaded_ = true;
}

AscendModelPrivate::~AscendModelPrivate()
{
    if (is_loaded_) {
        aclmdlUnload(model_id_);
        aclrtFree(model_dev_ptr_);
        aclrtFree(weight_dev_ptr_);
    }
}

int AscendModelPrivate::SetInputOuputInfo()
{
    // input info
    size_t input_num = aclmdlGetNumInputs(model_desc_.get());
    input_node_names_.resize(input_num);
    for (size_t i = 0; i < input_num; ++i)
    {
        aclmdlIODims dims;
        aclmdlGetInputDims(model_desc_.get(), 0, &dims);
        aclFormat format = aclmdlGetInputFormat(model_desc_.get(), i);
        const char *input_name = aclmdlGetInputNameByIndex(model_desc_.get(), i);
        aclDataType data_type = aclmdlGetInputDataType(model_desc_.get(), i);

        input_node_names_[i] = input_name;

        size_t tensor_size = 1;
        std::vector<int64_t> input_dims;
        for (size_t j = 0; j < dims.dimCount; j++){
            input_dims.emplace_back(dims.dims[j]);
            tensor_size *= dims.dims[j];
        }
        input_node_dims_.emplace_back(input_dims);
        input_tensor_sizes_.emplace_back(tensor_size);

        input_data_type_.emplace_back(data_type);
        input_scale_.emplace_back(1.0);
    }

    // output info
    size_t output_num = aclmdlGetNumOutputs(model_desc_.get());
    output_node_names_.resize(output_num);
    for (size_t i = 0; i < output_num; ++i)
    {
        aclmdlIODims dims;
        aclmdlGetOutputDims(model_desc_.get(), 0, &dims);
        aclFormat format = aclmdlGetOutputFormat(model_desc_.get(), i);
        const char *output_name = aclmdlGetOutputNameByIndex(model_desc_.get(), i);
        aclDataType data_type = aclmdlGetOutputDataType(model_desc_.get(), i);

        output_node_names_[i] = output_name;

        size_t tensor_size = 1;
        std::vector<int64_t> output_dims;
        for (size_t j = 0; j < dims.dimCount; j++){
            output_dims.emplace_back(dims.dims[j]);
            tensor_size *= dims.dims[j];
        }
        output_node_dims_.emplace_back(output_dims);
        output_tensor_sizes_.emplace_back(tensor_size);

        output_data_type_.emplace_back(data_type);
        output_scale_.emplace_back(1.0);
    }

    return 0;
}

int AscendModel::Init(const std::string& model_path, const std::string& param)
{
    ascend_model_priv_ = std::make_shared<AscendModelPrivate>();
    ascend_model_priv_->LoadModel(model_path, param);

    auto input_dims = ascend_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims){

        inputs_shape_.push_back(Shape(dims));
    }
    
    auto input_data_type = ascend_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertAscendDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = ascend_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = ascend_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertAscendDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int AscendModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    ascend_model_priv_ = std::make_shared<AscendModelPrivate>();
    ascend_model_priv_->LoadModel(mem_ptr, mem_size, param);

    auto input_dims = ascend_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = ascend_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertAscendDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = ascend_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = ascend_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertAscendDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }

    input_scale_ = ascend_model_priv_->GetInputScale();
    output_scale_ = ascend_model_priv_->GetOutputScale();
    
    return 0;
}

/**
 * @brief Get input shape
 *
 * @param index index of input
 * @return const Shape& shape of specified input
 */
const Shape& AscendModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& AscendModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int AscendModel::FixedOutputShape() noexcept
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
const DataLayout& AscendModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& AscendModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t AscendModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t AscendModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
std::vector<float> AscendModel::InputScale() const noexcept
{
    return input_scale_;
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
std::vector<float> AscendModel::OutputScale() const noexcept
{
    return output_scale_;
}


/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t AscendModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string AscendModel::GetKey() const noexcept
{
    return ascend_model_priv_->GetMD5();
}


any AscendModel::GetModel()
{
    return ascend_model_priv_->GetModel();
}


// REGISTER_MODEL_CREATOR("any", "any", AscendModelCreator) 