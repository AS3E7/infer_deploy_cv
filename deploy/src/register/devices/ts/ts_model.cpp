#include "ts_model.h"

#include <fstream>
#include <iostream>
#include <string.h>
#include <ts_rne_c_api.h>
#include <ts_rne_log.h>
#include <ts_rne_version.h>
#include <ts_rne_device.h>

#include "core/infer_server.h"
using namespace gddeploy;

namespace gddeploy
{
#define TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM (4)
static TS_S32 TS_MPI_TRP_RNE_InitNetsGraphAndParams(RNE_NET_S **net, TS_U8 **paramStride, TS_S32 *paramSize, TS_S32 nNet)
{
    /* 打开RNE设备 */
    // TS_S32 ret = TS_MPI_TRP_RNE_OpenDevice(NULL, NULL);
    // if (ret) {
    //     TS_MPI_TRP_RNE_Error("open device error!\n");
    //     return ret;
    // }
    /* 4 初始化多网络模型，并在每次初始化网络配置后，进行网络OnceLoad
     */
    TS_S32 ret = 0;
    for (TS_S32 i = 0; i < nNet; ++i) {
    /* 量化和权重数据需要4byte对齐
     * 如果未在头文件4byte对齐，可执行W_ALIGN_BYTES_NUM内代码，进行4字节对齐
     */
// #ifdef TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM
// #if 1
//         {
//             paramStride[i] = (TS_U8 *)TS_MPI_TRP_RNE_AllocLinearMem(paramSize[i] + TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM);
// #else
//         if (((TS_SIZE_T)net[i]->u8pParams & (TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM - 1)) != 0) {
//             paramStride[i] = (TS_U8 *)TS_MPI_TRP_RNE_Alloc(paramSize[i] + TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM);
// #endif
//             if (NULL == paramStride[i]) {
//                 TS_MPI_TRP_RNE_Error("insufficient memory!\n");
//                 return ret;
//             }
//             TS_SIZE_T addr = (TS_SIZE_T)paramStride[i];
//             addr += TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM - 1;
//             addr &= ~(TS_MPI_TRP_RNE_W_ALIGN_BYTES_NUM - 1);
//             memcpy((TS_VOID *)addr, net[i]->u8pParams, paramSize[i]);
//             net[i]->u8pParams = (TS_U8 *)addr;
//         }
// #endif
        /* 初始化单个网路
         */
        ret = TS_MPI_TRP_RNE_LoadModel(net[i]);
        if (ret) {
            TS_MPI_TRP_RNE_Error("load model error!\n");
            return ret;
        }
        /* net once load
         * 仅有网络模型配置为once load情况下，内部才真正执行once load
         */
        ret = TS_MPI_TRP_RNE_OnceLoad(net[i]);
        if (ret) {
            TS_MPI_TRP_RNE_Error("once load error!\n");
            return ret;
        }
    }

    return ret;
}


class TsModelPrivate{
public:
    TsModelPrivate(const std::string& model_path, const std::string& key){
        // TS_S32 ret = TS_MPI_TRP_RNE_OpenDevice(NULL, NULL);
        // if (ret) {
        //     TS_MPI_TRP_RNE_Error("open device error!\n");
        //     return ret;
        // }
        
        // RNE_NET_S nnModel;
        // memset(&nnModel, 0, sizeof(RNE_NET_S));
        // // net_->u8pGraph = modelCfg;
        // net_->u8pParams = (TS_U8 *)modelWeight;
        // net_->eInputType = RNE_NET_INPUT_TYPE_INT8_HWC;

        // TS_S32 paramSize[] = {sizeof(modelWeight)};
        
        SetInputShape();
        SetOutputShape();

    }
    TsModelPrivate(void* mem_ptr,  size_t mem_size, const std::string& key){
        TS_MPI_TRP_RNE_SetLogLevel(RNE_LOG_INFO);
        TS_MPI_TRP_RNE_Info("current log level : %d\n", TS_MPI_TRP_RNE_GetLogLevel());
        TS_MPI_TRP_RNE_Info("current lib version :%s\n", TS_MPI_TRP_RNE_GetSdkVersion());
        TS_MPI_TRP_RNE_Info("main start...\n");

        TS_S32 ret = TS_MPI_TRP_RNE_OpenDevice(NULL, NULL);
        if (ret) {
            TS_MPI_TRP_RNE_Error("open device error!\n");
            return ;
        }
        TS_MPI_TRP_RNE_Info("main end...\n");

        char *model_mem_ptr = (char *)mem_ptr;

        /*----------------test----------------------*/
        // std::ifstream cfg_file("/root/gddeploy/data/models/helmet/_yolov5_8_r.cfg", std::ios::in|std::ios::binary);
        // std::ifstream weight_file("/root/gddeploy/data/models/helmet/_yolov5_8_r.weight", std::ios::in|std::ios::binary);

        // cfg_file.seekg(0, std::ios::end);
        // int cfg_length = cfg_file.tellg();   
        // cfg_file.seekg(0, std::ios::beg);    
        // char* cfg_buffer = new char[cfg_length];    
        // cfg_file.read(cfg_buffer, cfg_length); 

        // weight_file.seekg(0, std::ios::end);
        // int weight_length = weight_file.tellg();
        // weight_file.seekg(0, std::ios::beg);  
        // char* weight_buffer = new char[weight_length];
        // weight_file.read(weight_buffer, weight_length); 

        // net_ = new RNE_NET_S;
        // memset(net_, 0, sizeof(RNE_NET_S));
        // net_->u8pGraph = (TS_U8 *)cfg_buffer;
        // net_->u8pParams = (TS_U8 *)weight_buffer;
        // net_->eInputType = RNE_NET_INPUT_TYPE_INT8_HWC;

        // RNE_NET_S *net[] = {net_};
        // const TS_S32 num = sizeof(net) / sizeof(net[0]);
        // TS_S32 paramSize[] = {weight_length};
        // // TS_U8 *paramStride[num];
        // memset(paramStride_, 0, sizeof(paramStride_));
        /*-----------------------------------------*/
        int model_file_size;
        memcpy(&model_file_size, model_mem_ptr, 4);
        net_ = new RNE_NET_S;
        memset(net_, 0, sizeof(RNE_NET_S));
        net_->u8pGraph = (TS_U8 *)model_mem_ptr+4;
        net_->u8pParams = (TS_U8 *)model_mem_ptr+model_file_size+4;
        net_->eInputType = RNE_NET_INPUT_TYPE_FORMAT_0RGB;

        RNE_NET_S *net[] = {net_};
        const TS_S32 num = sizeof(net) / sizeof(net[0]);
        TS_S32 paramSize[] = {mem_size-model_file_size-4};
        TS_U8 *paramStride[num];
        memset(paramStride, 0, sizeof(paramStride));
        if (0 != TS_MPI_TRP_RNE_InitNetsGraphAndParams(net, paramStride_, paramSize, num)) {
            TS_MPI_TRP_RNE_Error("nets init error!\n");
            return;
        }

        SetInputShape();
        SetOutputShape();

        md5_ = key;
        /*--------------------test---------------------*/
        // delete weight_buffer;
        // delete cfg_buffer;
        /*-----------------------------------------*/
    }
    ~TsModelPrivate()
    {
        if (net_ != nullptr){
            // TS_MPI_TRP_RNE_Free(paramStride[n]);
            TS_MPI_TRP_RNE_UnloadModel(net_);
            TS_MPI_TRP_RNE_CloseDevice();
            delete net_;
            net_ = nullptr;
        }
    }

    int SetInputShape();
    int SetOutputShape();

    std::vector<const char *> GetInputNodeName() {return input_node_names_;}
    std::vector<std::vector<int64_t>> GetInputNodeDims() {return input_node_dims_;} // >=1 inputs.
    std::vector<size_t> GetInputTensorSizes() {return input_tensor_sizes_;}
    std::vector<TS_S32> GetInputDataType() {return input_data_type_;}

    std::vector<float> GetInputScale() { return input_scale_; }
    std::vector<float> GetOutputScale() { return output_scale_; }

    std::vector<const char *> GetOutputNodeName() {return output_node_names_;}
    std::vector<std::vector<int64_t>> GetOutputNodeDims() {return output_node_dims_;} // >=1 outputs
    std::vector<size_t> GetOutputTensorSizes() {return output_tensor_sizes_;}
    std::vector<TS_S32> GetOutputDataType() {return output_data_type_;}

    void* GetModel() { return (void *)(net_); }

    std::string GetMD5() { return md5_;  }

private:
    std::vector<const char *> input_node_names_;
    std::vector<std::vector<int64_t>> input_node_dims_; // >=1 inputs.
    std::vector<size_t> input_tensor_sizes_;
    std::vector<TS_S32> input_data_type_;

    std::vector<const char *> output_node_names_;
    std::vector<std::vector<int64_t>> output_node_dims_; // >=1 outputs
    std::vector<size_t> output_tensor_sizes_;
    std::vector<TS_S32> output_data_type_;

    RNE_NET_S *net_;

    std::vector<float> input_scale_;
    std::vector<float> output_scale_;

    std::string md5_;

    TS_U8 *paramStride_[1];
};
}

static DataType convertTsDataType2MyDataType(TS_S32 bit){
    DataType out;

    if (bit == 32){
        out = DataType::FLOAT32;
    } else if (bit == 16){
        out = DataType::FLOAT16;
    } else if ( bit == 8){
        out = DataType::INT8;
    } else {
        out = DataType::INVALID;
    }
    return out;
}

int TsModelPrivate::SetInputShape()
{
    RNE_BLOBS_S *blobs = TS_MPI_TRP_RNE_GetInputBlobs(net_);
    std::vector<int64_t> input_dims;

    for (TS_U32 blobIdx = 0; blobIdx < blobs->u32NBlob; ++blobIdx) {
        RNE_BLOB_S *stpBlob = &blobs->stpBlob[blobIdx];
        size_t tensor_size = 1;

        if (stpBlob->eFormat == RNE_BLOB_N_H_W_Cstride){
            // TS_S32 cStride = TS_MPI_TRP_RNE_CStride(stpBlob->s32C, stpBlob->s32BitNum, stpBlob->bIsJoined);
            // input_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, cStride};
            input_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, stpBlob->s32C};
        } else if (stpBlob->eFormat == RNE_BLOB_N_Cn_N_H_W_Cx){
            input_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, stpBlob->s32C};
        } else if (stpBlob->eFormat == RNE_BLOB_N_H_W_C){
            input_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, stpBlob->s32C};
        }
        input_node_dims_.push_back(input_dims);
        
        input_data_type_.push_back(stpBlob->s32BitNum);

        
        for (unsigned int j = 0; j < input_dims.size(); ++j)
            tensor_size *= input_dims.at(j);

        input_tensor_sizes_.push_back(tensor_size);

        input_scale_.push_back(stpBlob->fCoeff[0]);
    }

    return 0;
}

int TsModelPrivate::SetOutputShape()
{
    RNE_BLOBS_S *blobs = TS_MPI_TRP_RNE_GetResultBlobs(net_);
    std::vector<int64_t> output_dims;

    for (TS_U32 blobIdx = 0; blobIdx < blobs->u32NBlob; ++blobIdx) {
        RNE_BLOB_S *stpBlob = &blobs->stpBlob[blobIdx];
        size_t tensor_size = 1;

        if (stpBlob->eFormat == RNE_BLOB_N_H_W_Cstride){
            // TS_S32 cStride = TS_MPI_TRP_RNE_CStride(stpBlob->s32C, stpBlob->s32BitNum, stpBlob->bIsJoined);
            // output_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, cStride};
            output_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, stpBlob->s32C};
        } else if (stpBlob->eFormat == RNE_BLOB_N_Cn_N_H_W_Cx){
            output_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, stpBlob->s32C};
        } else if (stpBlob->eFormat == RNE_BLOB_N_H_W_C){
            output_dims = std::vector<int64_t>{stpBlob->s32N, stpBlob->s32H, stpBlob->s32W, stpBlob->s32C};
        }
        output_node_dims_.push_back(output_dims);
        
        output_data_type_.push_back(stpBlob->s32BitNum);

        
        for (unsigned int j = 0; j < output_dims.size(); ++j)
            tensor_size *= output_dims.at(j);

        output_tensor_sizes_.push_back(tensor_size);

        output_scale_.push_back(stpBlob->fCoeff[0]);
    }

    return 0;
}

int TsModel::Init(const std::string& model_path, const std::string& param)
{
    ts_model_priv_ = std::make_shared<TsModelPrivate>(model_path, param);

    auto input_dims = ts_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = ts_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertTsDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        intput_data_layout_.push_back(dl);
    }

    auto output_dims = ts_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = ts_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertTsDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        output_data_layout_.push_back(dl);
    }
    return 0;
}

int TsModel::Init(void* mem_ptr, size_t mem_size, const std::string& param)
{
    ts_model_priv_ = std::make_shared<TsModelPrivate>(mem_ptr, mem_size, param);

    auto input_dims = ts_model_priv_->GetInputNodeDims();
    for (auto dims : input_dims)
        inputs_shape_.push_back(Shape(dims));
    
    auto input_data_type = ts_model_priv_->GetInputDataType();
    for (auto dti : input_data_type){
        DataLayout dl;
        dl.dtype = convertTsDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        intput_data_layout_.push_back(dl);
        input_scale_ = ts_model_priv_->GetInputScale();
    }

    auto output_dims = ts_model_priv_->GetOutputNodeDims();
    for (auto dims : output_dims)
        outputs_shape_.push_back(Shape(dims));

    auto output_data_type = ts_model_priv_->GetOutputDataType();
    for (auto dti : output_data_type){
        DataLayout dl;
        dl.dtype = convertTsDataType2MyDataType(dti);
        dl.order = DimOrder::NHWC;
        output_data_layout_.push_back(dl);

        output_scale_ = ts_model_priv_->GetOutputScale();
    }
    
    return 0;
}

/**
 * @brief Get input shape
 *
 * @param index index of input
 * @return const Shape& shape of specified input
 */
const Shape& TsModel::InputShape(int index) const noexcept
{
    return inputs_shape_[index];
}

/**
 * @brief Get output shape
 *
 * @param index index of output
 * @return const Shape& shape of specified output
 */
const Shape& TsModel::OutputShape(int index) const noexcept
{
    return outputs_shape_[index];
}

/**
 * @brief Check if output shapes are fixed
 *
 * @return Returns true if all output shapes are fixed, otherwise returns false.
 */
int TsModel::FixedOutputShape() noexcept
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
const DataLayout& TsModel::InputLayout(int index) const noexcept
{
    return intput_data_layout_[index];    
}


/**
 * @brief Get output layout on MLU
 *
 * @param index index of output
 * @return const DataLayout& data layout of specified output
 */
const DataLayout& TsModel::OutputLayout(int index) const noexcept
{
    return output_data_layout_[index];    
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
uint32_t TsModel::InputNum() const noexcept
{
    return inputs_shape_.size();
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
uint32_t TsModel::OutputNum() const noexcept
{
    return outputs_shape_.size();
}


/**
 * @brief Get number of input
 *
 * @return uint32_t number of input
 */
std::vector<float> TsModel::InputScale() const noexcept
{
    return input_scale_;
}


/**
 * @brief Get number of output
 *
 * @return uint32_t number of output
 */
std::vector<float> TsModel::OutputScale() const noexcept
{
    return output_scale_;
}


/**
 * @brief Get model batch size
 *
 * @return uint32_t batch size
 */
uint32_t TsModel::BatchSize() const noexcept
{
    return inputs_shape_[0].BatchSize();
}


/**
 * @brief Get model key
 *
 * @return const std::string& model key
 */
std::string TsModel::GetKey() const noexcept
{
    return ts_model_priv_->GetMD5();
}


any TsModel::GetModel()
{
    return ts_model_priv_->GetModel();
}
