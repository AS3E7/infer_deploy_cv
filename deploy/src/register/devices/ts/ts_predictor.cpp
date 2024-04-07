#include "ts_predictor.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface.h"
#include "common/logger.h"

#include <ts_rne_c_api.h>
#include <ts_rne_log.h>
#include <ts_rne_version.h>
// #include "ts_comm_video.h"

using namespace gddeploy;

namespace gddeploy
{
class TsPredictorPrivate{
public:
    TsPredictorPrivate() = default;
    TsPredictorPrivate(ModelPtr model):model_(model){
        void *tmp = gddeploy::any_cast<void*>(model->GetModel());
        net_ = (RNE_NET_S*)tmp;
    }
    ~TsPredictorPrivate(){
        for (auto &pool : pools_){
            // pool->DestroyPool();
            delete pool;
        }
        pools_.clear();
    }

    BufSurfWrapperPtr RequestBuffer(int idx){
        return pools_[idx]->GetBufSurfaceWrapper();
    }

    RNE_NET_S* net_;
    // Ts::MemoryInfo memory_info_handler_;

    int Init(std::string config);

    ModelPtr model_;
private:
    std::vector<BufPool*> pools_;
};
}

int TsPredictorPrivate::Init(std::string config)
{
    size_t o_num = model_->OutputNum();
    for (size_t i_idx = 0; i_idx < o_num; ++i_idx) {
        const DataLayout input_layout =  model_->OutputLayout(i_idx);
        auto dtype = input_layout.dtype;
        auto order = input_layout.order;
        int data_size = sizeof(uint32_t);
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
            model_c = (model_c % 32 == 0) ? model_c : ((model_c / 32 + 1) * 32);
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
        if (pool->CreatePool(&create_params, 3) < 0) {
            return -1;
        }
        pools_.emplace_back(pool);
    }

    return 0;
}

Status TsPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<TsPredictorPrivate>(model);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}

#include "opencv2/opencv.hpp"
#include <chrono>

#define TS_MPI_TRP_RNE_MASK_BITS(m) ((1ll << (m)) - 1)
Status TsPredictor::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    Status s = Status::SUCCESS;
#if 1
    BufSurface *surface = in_buf->GetBufSurface();
    BufSurfaceParams *src_param = in_buf->GetSurfaceParams(0);
    // VIDEO_FRAME_INFO_S *frame_info_s = (VIDEO_FRAME_INFO_S *)src_param->data_ptr;
    // VIDEO_FRAME_INFO_S *pstFrameInfo = (VIDEO_FRAME_INFO_S *)src_param->data_ptr;
    // // 1. 拷贝数据
    // 这里需要修改，如果是CPU，需要拷贝对齐，如果是ts mem，则看是否前处理已经对齐
    // RNE_BLOBS_S *blobs = TS_MPI_TRP_RNE_GetInputBlobs(priv_->net_.get());
    // RNE_BLOB_S *stpBlob = &blobs->stpBlob[0];
    // priv_->net_->vpInput = (TS_U8 *)src_param->_reserved[0];//frame_info_s->stVFrame.u64VirAddr[0];
    // priv_->net_->vpInput = (TS_U8 *)in_buf->GetData(0);//frame_info_s->stVFrame.u64VirAddr[0];
    // cv::Mat output_mat(640, 640, CV_8UC3, (uint8_t *)priv_->net_->vpInput);
    // cv::imwrite("/root/gddeploy/preds/pre.jpg", output_mat);

    // if (0 != TS_MPI_TRP_RNE_SetInputBlobsAddr(priv_->net_.get(), (void *)TS_MPI_TRP_RNE_VirtualToPhysicalAddress((TS_SIZE_T)src_param->_reserved[0]), true)) {
    if (0 != TS_MPI_TRP_RNE_SetInputBlobsAddr(priv_->net_, (void *)src_param->_reserved[0], true)) {
        GDDEPLOY_ERROR("set input blobs addr error!");
        return gddeploy::Status::ERROR_BACKEND;
    }
    
    // // 2. 推理
    // auto t0 = std::chrono::high_resolution_clock::now();
    RNE_BLOBS_S *out_blobs = TS_MPI_TRP_RNE_Forward(priv_->net_);
    if (out_blobs == NULL) {
        GDDEPLOY_ERROR("net forward error!");
        return gddeploy::Status::ERROR_BACKEND;
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("inference time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    
    // // 3. 后处理，画图
    // t0 = std::chrono::high_resolution_clock::now();
    std::vector<BufSurfWrapperPtr> out_bufs;
    for (TS_U32 idx = 0; idx < out_blobs->u32NBlob; ++idx) {
        BufSurfWrapperPtr buf = priv_->RequestBuffer(idx);
        // float *out_float_ptr = (float*)buf->GetData(0, 0);
        // TS_U8 *out_int_ptr = (TS_U8 *)(out_blobs->stpBlob[idx].vpAddr);
        // TS_FLOAT fCoeff = *out_blobs->stpBlob[idx].fCoeff;

        TS_S32 num = out_blobs->stpBlob[idx].s32N;
        TS_S32 h = out_blobs->stpBlob[idx].s32H;
        TS_S32 w = out_blobs->stpBlob[idx].s32W;
        TS_S32 c = out_blobs->stpBlob[idx].s32C;
        TS_S32 cStride = TS_MPI_TRP_RNE_CStride(c, out_blobs->stpBlob[idx].s32BitNum, out_blobs->stpBlob[idx].bIsJoined);
        TS_S32 uSize = (out_blobs->stpBlob[idx].s32BitNum / CHAR_BIT);

        // for(TS_S32 n = 0; n < num; ++n) {
        //     for (TS_S32 i = 0; i < h; ++i) {
        //         for (TS_S32 j = 0; j < w; ++j) {
        //             for (TS_S32 k = 0; k < c; ++k) {
        //                 // TS_U8 *d = (TS_U8 *)(out + (((j + i * w) * cStride + k) * uSize));
        //                 TS_S32 *d = (TS_S32 *)(out_int_ptr + (((j + i * w) * cStride + k) * uSize));
        //                 TS_S32 data = *d & TS_MPI_TRP_RNE_MASK_BITS(out_blobs->stpBlob[idx].s32BitNum);

        //                 // *(out_ptr + ((h * w) * k + i * w + j)) = data * fCoeff;
        //                 *(out_float_ptr + n*h*w*c+i*w*c+j*c+k) = data * fCoeff;
        //                 // if (*(out_ptr + ((h * w) * k + i * w + j)) > 1 || *(out_ptr + ((h * w) * k + i * w + j)) < -1){
        //             }
        //         }
        //     }
        // }
        
        char *out_char_ptr = (char*)buf->GetData(0, 0);
        TS_U8 *out_int_ptr = (TS_U8 *)(out_blobs->stpBlob[idx].vpAddr);
        int size = num * h * w * cStride * uSize;
        memcpy((void *)out_char_ptr, out_int_ptr, size);
        
        out_bufs.emplace_back(buf);
    }
    // t1 = std::chrono::high_resolution_clock::now();
    // printf("post copy time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    
    pack->predict_io->Set(out_bufs);
#endif
    return gddeploy::Status::SUCCESS; 
}