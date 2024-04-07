#include <iostream>
#include <string.h>
#include "core/infer_server.h"
#include "nv_model.h"
#include "logging.h"
#include "common/logger.h"
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <iostream>
#include <fstream>
#include "common/json.hpp"
#include "core/infer_server.h"
#include "NvOnnxParser.h"
#include "md5.h"
#include "nv_common.h"

using namespace gddeploy;
using namespace nvinfer1;
using namespace std;

static int OIDC[] = {8, 2, 6, 7, 9, 3, 5, 9, 2, 3};

namespace gddeploy
{
    class NvModelPrivate
    {
    public:
        typedef enum AccDeviceType
        {
            GPU = 0, // nano only support GPU
            DLA_0 = 1,
            DLA_1 = 2
        } AccDeviceType;

        NvModelPrivate()
        {
            batch_size_ = 32;
            acc_device_type_ = GPU;
        };
        ~NvModelPrivate();
        Status Init(void *mem_ptr, size_t mem_size, const std::string &param, ModelPropertiesPtr model_info_priv);
        std::vector<Shape> input_shapes_;
        std::vector<Shape> output_shapes_;
        std::vector<DataLayout> intput_data_layout_;
        std::vector<DataLayout> output_data_layout_;

        std::shared_ptr<ICudaEngine> engine_;

        std::string GetMD5() { return md5_;  }

    private:
        AccDeviceType acc_device_type_;
        std::unique_ptr<sample::Logger> logger_;
        std::unique_ptr<IRuntime> runtime_;
        unsigned int batch_size_;

        struct InferDeleter
        {
            template <typename T>
            void operator()(T *obj) const
            {
                // obj->destroy();
            }
        };

        std::string md5_;
    };
}

NvModelPrivate::~NvModelPrivate()
{
    //     if (engine_)
    //     {
    //         engine_->destroy();
    //     }
    //     if (runtime_)
    //     {
    //         runtime_->destroy();
    //     }
}

Status NvModelPrivate::Init(void *mem_ptr, size_t mem_size, const std::string &param, ModelPropertiesPtr model_info_priv)
{
    int batch_size = 1;
    unsigned char md5[16] = {0};
    char cache_hash_name[32 * 2 + 1] = {0};
    size_t size = 0;
    char *trt_model_stream = nullptr;

    // auto device_config = nlohmann::json::parse(param);
    // if (param.find("acc_device_type") != std::string::npos)
    // {
    //     acc_device_type_ = device_config["acc_device_type"];
    // }
    acc_device_type_ = GPU;

    logger_ = std::make_unique<sample::Logger>();
    IBuilder *builder = createInferBuilder(*logger_);
    builder->setMaxBatchSize(batch_size_);
    auto batch_flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(batch_flag);
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, *logger_);
    CHECK_AND_RET(parser == nullptr, Status::INVALID_MODEL);
    auto parsed = parser->parse(mem_ptr, mem_size);
    CHECK_AND_RET(!parsed, Status::INVALID_MODEL);
    MD5((const unsigned char *)mem_ptr, mem_size, md5);
    for (int i = 0; i < 16; i++)
    {
        sprintf(cache_hash_name + (i * 2), "%02x", md5[i]);
    }
    sprintf(cache_hash_name + 32, "_%d_%d", batch_size_, acc_device_type_);

    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    IOptimizationProfile *profile = builder->createOptimizationProfile();
    if (model_info_priv->GetQat() == false)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    else
    {
        config->setFlag(BuilderFlag::kINT8);
    }
    config->clearFlag(BuilderFlag::kTF32);
    config->setAvgTimingIterations(4);
    config->setMaxWorkspaceSize(1 << 30);
    Dims dims = network->getInput(0)->getDimensions();
    // for(int i = 0; i < dims.nbDims;i++)
    // {
    //     std::cout << "zzy debug inputdims " " dims: " << dims.d[i] << " i "<< i << std::endl;
    // }
    dims.d[0] = 1;
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, dims);
    dims.d[0] = 4;
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, dims);
    dims.d[0] = 32;
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);
    if (acc_device_type_ != GPU)
    {
        if (builder->getNbDLACores() == 0)
        {
            GDDEPLOY_ERROR("Trying to use DLA core {} on the platform,but doesn't have any DLA cores\n", acc_device_type_ - 1);
            return Status::INVALID_MODEL;
        }
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
        config->setFlag(BuilderFlag::kFP16);
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(acc_device_type_ - 1);
    }
    runtime_ = std::unique_ptr<IRuntime>(createInferRuntime(*logger_));
    // runtime_ = createInferRuntime(*logger_);
    CHECK_AND_RET(runtime_ == nullptr, Status::INVALID_MODEL);
    std::string trt_filename((char *)cache_hash_name);
    std::ifstream in_trt_file(trt_filename, std::ios::binary);
    bool build_engine = false;
    if (in_trt_file.good())
    {
        GDDEPLOY_INFO("read cache trt.\n");
        in_trt_file.seekg(0, in_trt_file.end);
        size = in_trt_file.tellg();
        in_trt_file.seekg(0, in_trt_file.beg);
        trt_model_stream = new char[size];
        if (trt_model_stream == nullptr)
        {
            GDDEPLOY_ERROR("malloc trt_model_stream error.\n");
            return Status::ERROR_MEMORY;
        }
        in_trt_file.read(trt_model_stream, size);
        in_trt_file.close();
        for (size_t i = 0; i < std::min((size_t)5000, size); i++)
        {
            trt_model_stream[i] = trt_model_stream[i] ^ OIDC[i % 10];
        }

        // engine_ = runtime_->deserializeCudaEngine(trt_model_stream, size, nullptr);
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(trt_model_stream, size), InferDeleter());
        delete[] trt_model_stream;
        if (engine_ == nullptr)
        {
            GDDEPLOY_ERROR("read cache model error,rebuild\n");
            build_engine = true;
        }
    }
    else
    {
        build_engine = true;
    }
    if (true == build_engine)
    {
        // engine_ = builder->buildCudaEngine(*network);
        // engine_ = builder->buildEngineWithConfig(*network, *config);
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), InferDeleter());
        CHECK_AND_RET(engine_ == nullptr, Status::INVALID_MODEL);
        IHostMemory *serialized_engine = engine_->serialize();
        char *engine_buf = (char *)(serialized_engine->data());
        for (size_t i = 0; i < 5000; i++)
        {
            engine_buf[i] = engine_buf[i] ^ OIDC[i % 10];
        }
        std::ofstream trt_file(trt_filename, std::ios::binary | std::ios_base::trunc);
        trt_file.write(engine_buf, serialized_engine->size());
        trt_file.close();
        serialized_engine->destroy();
    }
    for (int i = 0; i < network->getNbInputs(); i++)
    {
        Dims dims = network->getInput(i)->getDimensions();
        std::vector<Shape::value_type> dims_list;
        for (int j = 0; j < dims.nbDims; j++)
        {
            // std::cout << "zzy debug inputdims " << i << " dims: " << dims.d[j] << " j " << j << std::endl;
            if (dims.d[j] == -1)
                dims.d[j] = batch_size;
            dims_list.push_back(dims.d[j]);
        }
        Shape shape(dims_list);
        input_shapes_.push_back(shape);
        DataLayout dlayout;
        dlayout.dtype = DataType::FLOAT32;
        dlayout.order = DimOrder::NCHW;
        intput_data_layout_.push_back(dlayout);
    }
    for (int i = 0; i < network->getNbOutputs(); i++)
    {
        Dims dims = network->getOutput(i)->getDimensions();

        std::vector<Shape::value_type> dims_list;
        for (int j = 0; j < dims.nbDims; j++)
        {
            if (dims.d[j] == -1)
                dims.d[j] = batch_size;
            dims_list.push_back(dims.d[j]);
        }
        Shape shape(dims_list);
        output_shapes_.push_back(shape);
        DataLayout dlayout;
        dlayout.dtype = DataType::FLOAT32;
        dlayout.order = DimOrder::NCHW;
        output_data_layout_.push_back(dlayout);
    }

    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    md5_ = param;
    return Status::SUCCESS;
}

int NvModel::Init(const std::string &model_path, const std::string &param)
{
    nv_modl_priv_ = std::make_shared<NvModelPrivate>();
    std::ifstream in_trt_file(model_path, std::ios::binary);
    size_t mem_size = 0;
    char *onnx_model_stream = nullptr;
    if (in_trt_file.good())
    {
        GDDEPLOY_INFO("read cache trt.\n");
        in_trt_file.seekg(0, in_trt_file.end);
        mem_size = in_trt_file.tellg();
        in_trt_file.seekg(0, in_trt_file.beg);
        onnx_model_stream = new char[mem_size];
        if (onnx_model_stream == nullptr)
        {
            GDDEPLOY_ERROR("malloc onnx_model_stream error.\n");
            return -1;
        }
        in_trt_file.read(onnx_model_stream, mem_size);
        in_trt_file.close();
    }
    else
    {
        return -1;
    }
    ModelPropertiesPtr model_info_priv = this->GetModelInfoPriv();
    Status ret = nv_modl_priv_->Init((void *)onnx_model_stream, mem_size, param, model_info_priv);
    inputs_shape_ = nv_modl_priv_->input_shapes_;
    intput_data_layout_ = nv_modl_priv_->intput_data_layout_;
    outputs_shape_ = nv_modl_priv_->output_shapes_;
    output_data_layout_ = nv_modl_priv_->output_data_layout_;
    return int(ret);
}

int NvModel::Init(void *mem_ptr, size_t mem_size, const std::string &param)
{
    ModelPropertiesPtr model_info_priv = this->GetModelInfoPriv();
    nv_modl_priv_ = std::make_shared<NvModelPrivate>();
    Status ret = nv_modl_priv_->Init(mem_ptr, mem_size, param, model_info_priv);
    inputs_shape_ = nv_modl_priv_->input_shapes_;
    intput_data_layout_ = nv_modl_priv_->intput_data_layout_;
    outputs_shape_ = nv_modl_priv_->output_shapes_;
    output_data_layout_ = nv_modl_priv_->output_data_layout_;
    return int(ret);
}

/**
 * @brief Get input shape
 *
 * @param index index of input
 * @return const Shape& shape of specified input
 */
const Shape &NvModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape &NvModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int NvModel::FixedOutputShape() noexcept
{
    if (outputs_shape_.size() == 0)
    {
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
const DataLayout &NvModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];
}

/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout &NvModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];
}

/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t NvModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}

/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t NvModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}

/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t NvModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}

std::vector<float> NvModel::InputScale() const noexcept
{
    return std::vector<float>{};
}

std::vector<float>  NvModel::OutputScale() const noexcept
{
    return std::vector<float>{};
}

/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string NvModel::GetKey() const noexcept
{
    return nv_modl_priv_->GetMD5();
}

any NvModel::GetModel()
{
    return nv_modl_priv_->engine_;
}

// REGISTER_MODEL_CREATOR("any", "any", NvModelCreator)