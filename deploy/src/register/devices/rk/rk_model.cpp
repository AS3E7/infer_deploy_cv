#include "rk_model.h"

#include <iostream>
#include <string.h>

#include "common/logger.h"
#include "core/infer_server.h"

#include "rknn_api.h"

using namespace gddeploy;

namespace gddeploy
{
class RkModelPrivate{
public:
    RkModelPrivate(const std::string& model_path,const std::string& key);
    RkModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key);
    ~RkModelPrivate();

    int SetInputOuputInfo();

    std::vector<const char *> GetInputNodeName() {return input_node_names_;}
    std::vector<std::vector<int64_t>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<rknn_tensor_type> GetInputDataType() {return input_data_type_;}
    std::vector<float> GetInputScale() {return input_scale_;}
    std::vector<int32_t> GetInputZps() {return input_zps_;}

    std::vector<const char *> GetOutputNodeName() {return output_node_names_;}
    std::vector<std::vector<int64_t>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<rknn_tensor_type> GetOutputDataType() {return output_data_type_;}
    std::vector<float> GetOutputScale() {return output_scale_;}
    std::vector<int32_t> GetOutputZps() {return output_zps_;}

    rknn_context GetModel() { return ctx_; }

    std::string GetMD5() { return md5_;  }

private:
    rknn_context ctx_;
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    std::vector<const char *> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<rknn_tensor_type> input_data_type_;
    std::vector<float> input_scale_;
    std::vector<int32_t> input_zps_;

    std::vector<const char *> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<rknn_tensor_type> output_data_type_;
    std::vector<float> output_scale_;
    std::vector<int32_t> output_zps_;

    std::string md5_;
};
}

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
    unsigned char* data = NULL;
    int            ret;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char*)malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }

    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
    FILE* fp = NULL;
    unsigned char* data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static DataType convertRkDataType2MyDataType(rknn_tensor_type in){
    DataType out;

    switch (in)
    {
    case RKNN_TENSOR_INT8:
        out = DataType::INT8;
        break;
    case RKNN_TENSOR_UINT8:
        out = DataType::UINT8;
        break;
    case RKNN_TENSOR_FLOAT32:
        out = DataType::FLOAT32;
        break;
    case RKNN_TENSOR_FLOAT16:
        out = DataType::FLOAT16;
        break;
    case RKNN_TENSOR_UINT16:
        out = DataType::UINT16;
        break;
    case RKNN_TENSOR_INT16:
        out = DataType::INT16;
        break;
    case RKNN_TENSOR_INT32:
    case RKNN_TENSOR_UINT32:
        out = DataType::INT32;
        break;
    default:
        out = DataType::INVALID;
        break;
    }
    return out;
}

RkModelPrivate::RkModelPrivate(const std::string& model_path, const std::string& key)
{
    int model_size = 0;
    auto model_data = load_model(model_path.c_str(), &model_size);

#if DEBUG_PERF
    int ret = rknn_init(&ctx_, model_data, model_size, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
#else
    int ret = rknn_init(&ctx_, model_data, model_size, 0, NULL);
#endif
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
    }

    SetInputOuputInfo();

    md5_ = key;
}

RkModelPrivate::RkModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key)
{
#if DEBUG_PERF
    int ret = rknn_init(&ctx_, model_data, model_size, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
#else
    int ret = rknn_init(&ctx_, mem_ptr, mem_size, 0, NULL);
#endif
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
    }

    SetInputOuputInfo();

    md5_ = key;
}

RkModelPrivate::~RkModelPrivate()
{
    int ret = rknn_destroy (ctx_);
    if (ret < 0)
    {
        printf("rknn_destroy error ret=%d\n", ret);
    }
}

int RkModelPrivate::SetInputOuputInfo()
{
    // 获取模型输入输出信息
    rknn_sdk_version version;
    auto ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_query error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query in/out num error ret=%d\n", ret);
        return -1;
    }
    
    input_attrs_.resize(io_num.n_input);
    input_node_names_.resize(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query input attr error ret=%d\n", ret);
            return -1;
        }
        input_scale_.push_back(input_attrs_[i].scale);
        input_zps_.push_back(input_attrs_[i].zp);

        input_node_names_[i] = input_attrs_[i].name;

        size_t tensor_size = 1;
        std::vector<int64_t> input_dims;
        for (size_t j = 0; j < input_attrs_[i].n_dims; j++){
            input_dims.emplace_back(input_attrs_[i].dims[j]);
            tensor_size *= input_attrs_[i].dims[j];
        }
        input_node_dims_.emplace_back(input_dims);
        input_tensor_sizes_.emplace_back(tensor_size);

        input_data_type_.emplace_back(input_attrs_[i].type);
    }

    output_attrs_.resize(io_num.n_output);
    output_node_names_.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]),
                   sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query input attr error ret=%d\n", ret);
            return -1;
        }

        output_scale_.push_back(output_attrs_[i].scale);
        output_zps_.push_back(output_attrs_[i].zp);

        output_node_names_[i] = output_attrs_[i].name;

        size_t tensor_size = 1;
        std::vector<int64_t> output_dims;
        for (size_t j = 0; j < output_attrs_[i].n_dims; j++){
            output_dims.emplace_back(output_attrs_[i].dims[j]);
            tensor_size *= output_attrs_[i].dims[j];
        }
        output_node_dims_.emplace_back(output_dims);
        output_tensor_sizes_.emplace_back(tensor_size);

        output_data_type_.emplace_back(output_attrs_[i].type);
    }

    return 0;
}

int RkModel::Init(const std::string& model_path, const std::string& param)
{
    rk_model_priv_ = std::make_shared<RkModelPrivate>(model_path, param);

    auto input_dims = rk_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims){

        inputs_shape_.push_back(Shape(dims));
    }
    
    auto input_data_type = rk_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertRkDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = rk_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = rk_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertRkDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int RkModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    rk_model_priv_ = std::make_shared<RkModelPrivate>(mem_ptr, mem_size, param);

    auto input_dims = rk_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = rk_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertRkDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = rk_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = rk_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertRkDataType2MyDataType(dti);
        dl.order = DimOrder::NCHW;
        output_data_layout_.push_back(dl);
    }

    input_scale_ = rk_model_priv_->GetInputScale();
    output_scale_ = rk_model_priv_->GetOutputScale();
    input_zp_ = rk_model_priv_->GetInputZps();
    output_zp_ = rk_model_priv_->GetOutputZps();
    
    return 0;
}

/**
 * @brief Get input shape
 *
 * @param index index of input
 * @return const Shape& shape of specified input
 */
const Shape& RkModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& RkModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int RkModel::FixedOutputShape() noexcept
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
const DataLayout& RkModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& RkModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t RkModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t RkModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
std::vector<float> RkModel::InputScale() const noexcept
{
    return input_scale_;
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
std::vector<float> RkModel::OutputScale() const noexcept
{
    return output_scale_;
}

std::vector<int> RkModel::InputZp() const noexcept
{
    return input_zp_;
}

std::vector<int> RkModel::OutputZp() const noexcept
{
    return output_zp_;
}


/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t RkModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string RkModel::GetKey() const noexcept
{
    return rk_model_priv_->GetMD5();
}


any RkModel::GetModel()
{
    return rk_model_priv_->GetModel();
}


// REGISTER_MODEL_CREATOR("any", "any", RkModelCreator) 