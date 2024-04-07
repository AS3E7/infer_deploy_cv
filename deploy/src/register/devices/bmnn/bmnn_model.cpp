#include "bmnn_model.h"

#include <iostream>
#include <string.h>

#include "common/logger.h"
#include "core/infer_server.h"

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"
// #include "bm_wrapper.hpp"

using namespace gddeploy;

namespace gddeploy
{
class BmnnModelPrivate{
public:
    BmnnModelPrivate(const std::string& model_path,const std::string& key);
    BmnnModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key);
    ~BmnnModelPrivate();

    int SetInputOuputInfo();

    std::vector<const char *> GetInputNodeName() {return input_node_names_;}
    std::vector<std::vector<int64_t>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<bm_data_type_t> GetInputDataType() {return input_data_type_;}
    std::vector<float> GetInputScale() {return input_scale_;}

    std::vector<const char *> GetOutputNodeName() {return output_node_names_;}
    std::vector<std::vector<int64_t>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<bm_data_type_t> GetOutputDataType() {return output_data_type_;}
    std::vector<float> GetOutputScale() {return output_scale_;}

    std::shared_ptr<void> GetModel() { return p_bmrt_; }

    std::string GetMD5() { return md5_;  }

private:
    bm_handle_t handle_;
    std::shared_ptr<void> p_bmrt_;
    const bm_net_info_t *net_info_;

    std::vector<const char *> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<bm_data_type_t> input_data_type_;
    std::vector<float> input_scale_;

    std::vector<const char *> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<bm_data_type_t> output_data_type_;
    std::vector<float> output_scale_;

    std::string md5_;
};
}

static DataType convertBmnnDataType2MyDataType(bm_data_type_t in){
    DataType out;

    switch (in)
    {
    case BM_INT8:
        out = DataType::INT8;
        break;
    case BM_UINT8:
        out = DataType::UINT8;
        break;
    case BM_FLOAT32:
        out = DataType::FLOAT32;
        break;
    case BM_FLOAT16:
        out = DataType::FLOAT16;
        break;
    case BM_UINT16:
        out = DataType::UINT16;
        break;
    case BM_INT16:
        out = DataType::INT16;
        break;
    case BM_INT32:
    case BM_UINT32:
        out = DataType::INT32;
        break;
    default:
        out = DataType::INVALID;
        break;
    }
    return out;
}

BmnnModelPrivate::BmnnModelPrivate(const std::string& model_path, const std::string& key)
{
    if (bm_dev_query(0)){
        GDDEPLOY_ERROR("[Model] bmnn bm_dev_query fail\n");
    }

    bm_dev_request(&handle_, 0);
    void *p_bmrt = bmrt_create(handle_);
    p_bmrt_ = std::shared_ptr<void>(p_bmrt, [](void *p_bmrt){
        bmrt_destroy(p_bmrt);

        bm_handle_t bm_handle = (bm_handle_t)bmrt_get_bm_handle(p_bmrt);
        bm_dev_free(bm_handle);
    });

    bmrt_load_bmodel(p_bmrt_.get(), model_path.c_str());

    const char **net_names;
    bmrt_get_network_names(p_bmrt, &net_names);    

    net_info_ = bmrt_get_network_info(p_bmrt_.get(), net_names[0]);
    if (NULL == net_info_) {
        std::cout << "ERROR: get net-info failed!" << std::endl;
    }
    free(net_names);

    SetInputOuputInfo();

    md5_ = key;
}

BmnnModelPrivate::BmnnModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key)
{
    bm_dev_request(&handle_, 0);
    void *p_bmrt = bmrt_create(handle_);
    p_bmrt_ = std::shared_ptr<void>(p_bmrt, [](void *p_bmrt){
        bmrt_destroy(p_bmrt);

        bm_handle_t bm_handle = (bm_handle_t)bmrt_get_bm_handle(p_bmrt);
        // bm_dev_free(bm_handle);
    });

    bmrt_load_bmodel_data(p_bmrt_.get(), mem_ptr, mem_size);

    const char **net_names;
    bmrt_get_network_names(p_bmrt, &net_names);
    std::string net_name = net_names[0];
    free(net_names);

    net_info_ = bmrt_get_network_info(p_bmrt_.get(), net_name.c_str());
    if (NULL == net_info_) {
        std::cout << "ERROR: get net-info failed!" << std::endl;
    }


    SetInputOuputInfo();

    md5_ = key;
}

BmnnModelPrivate::~BmnnModelPrivate()
{
    bm_dev_free(handle_);
}

int BmnnModelPrivate::SetInputOuputInfo()
{
    // input info
    input_node_names_.resize(net_info_->input_num);
    for (size_t i = 0; i < net_info_->input_num; ++i)
    {
        input_node_names_[i] = net_info_->input_names[i];

        size_t tensor_size = 1;
        std::vector<int64_t> input_dims;
        for (size_t j = 0; j < (net_info_->stages->input_shapes+i)->num_dims; j++){
            input_dims.emplace_back((net_info_->stages->input_shapes+i)->dims[j]);
            tensor_size *= (net_info_->stages->input_shapes+i)->dims[j];
        }
        input_node_dims_.emplace_back(input_dims);
        input_tensor_sizes_.emplace_back(tensor_size);

        input_data_type_.emplace_back(*net_info_->input_dtypes);
        input_scale_.emplace_back(*(net_info_->input_scales+i));
    }

    // output info
    output_node_names_.resize(net_info_->output_num);
    for (size_t i = 0; i < net_info_->output_num; ++i)
    {
        output_node_names_[i] = net_info_->output_names[i];

        size_t tensor_size = 1;
        std::vector<int64_t> output_dims;
        for (size_t j = 0; j < (net_info_->stages->output_shapes+i)->num_dims; j++){
            output_dims.emplace_back((net_info_->stages->output_shapes+i)->dims[j]);
            tensor_size *= (net_info_->stages->output_shapes+i)->dims[j];
        }
        output_node_dims_.emplace_back(output_dims);
        output_tensor_sizes_.emplace_back(tensor_size);

        output_data_type_.emplace_back(*net_info_->output_dtypes);
        output_scale_.emplace_back(*(net_info_->output_scales+i));
    }

    return 0;
}

int BmnnModel::Init(const std::string& model_path, const std::string& param)
{
    bmnn_model_priv_ = std::make_shared<BmnnModelPrivate>(model_path, param);

    auto input_dims = bmnn_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims){

        inputs_shape_.push_back(Shape(dims));
    }
    
    auto input_data_type = bmnn_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertBmnnDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = bmnn_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = bmnn_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertBmnnDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int BmnnModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    bmnn_model_priv_ = std::make_shared<BmnnModelPrivate>(mem_ptr, mem_size, param);

    auto input_dims = bmnn_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = bmnn_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertBmnnDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = bmnn_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = bmnn_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertBmnnDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }

    input_scale_ = bmnn_model_priv_->GetInputScale();
    output_scale_ = bmnn_model_priv_->GetOutputScale();
    
    return 0;
}

/**
 * @brief Get input shape
 *
 * @param index index of input
 * @return const Shape& shape of specified input
 */
const Shape& BmnnModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& BmnnModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int BmnnModel::FixedOutputShape() noexcept
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
const DataLayout& BmnnModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& BmnnModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t BmnnModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t BmnnModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
std::vector<float> BmnnModel::InputScale() const noexcept
{
    return input_scale_;
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
std::vector<float> BmnnModel::OutputScale() const noexcept
{
    return output_scale_;
}


/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t BmnnModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string BmnnModel::GetKey() const noexcept
{
    return bmnn_model_priv_->GetMD5();
}


any BmnnModel::GetModel()
{
    return bmnn_model_priv_->GetModel();
}


// REGISTER_MODEL_CREATOR("any", "any", BmnnModelCreator) 