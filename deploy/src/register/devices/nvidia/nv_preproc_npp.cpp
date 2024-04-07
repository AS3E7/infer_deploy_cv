#include <memory>
#include <string>
#include "nv_preproc_npp.h"

#include "core/mem/buf_surface.h"
#include "core/preprocess.h"
#include "core/model.h"
#include "core/mem/buf_surface_util.h"

#include "common/logger.h"
#include "common/type_convert.h"

#include "opencv2/opencv.hpp"

#include "cuda_runtime.h"
#include "npp.h"

namespace gddeploy{   
class NvPreProcNPPPriv{
public:
    NvPreProcNPPPriv(ModelPtr model):model_(model){
        for (int i = 0; i < model->InputNum(); i++){
            auto shape = model_->InputShape(i);
            model_h_ = shape[2];
            model_w_ = shape[3];
            model_c_ = shape[1];
            model_b_ = shape[0];
        }
    }
    ~NvPreProcNPPPriv(){
        cudaStreamDestroy(stream_);
    }

    int Init(std::string config);

    int PreProc(BufSurfWrapperPtr src, BufSurfWrapperPtr dst);

    int SetModel(ModelPtr model){
        if (model == nullptr){
            return -1;
        }else{
            model_ = model;
        }
        return 0;
    }

    BufSurfWrapperPtr RequestBuffer(){
        BufSurfWrapperPtr buf_ptr = pools_[0]->GetBufSurfaceWrapper();

        return buf_ptr;
    }

    int GetModelWidth(){
        return model_w_;
    }

    int GetModelHeight(){
        return model_h_;
    }


private:
    int Preproc_yolov5(BufSurfWrapperPtr src, BufSurfWrapperPtr dst);
    int Preproc_yolox(BufSurfWrapperPtr src, BufSurfWrapperPtr dst);
    int Preproc_classify(BufSurfWrapperPtr src, BufSurfWrapperPtr dst);
    int Preproc_seg(BufSurfWrapperPtr src, BufSurfWrapperPtr dst);

    ModelPtr model_;
    int model_h_;
    int model_w_;
    int model_c_;
    int model_b_;

    // nv
    cudaStream_t stream_;
    NppStreamContext npp_stream_context_;

    // mem poo
    std::vector<BufPool*> pools_;
};
}

using namespace gddeploy;
int NvPreProcNPPPriv::Init(std::string config){
    cudaStreamCreate(&stream_);
    nppSetStream(stream_);
    nppGetStreamContext(&npp_stream_context_);

    // nppStreamCreate(&npp_stream_context_.stream);
    // nppStreamGetCudaStream(npp_stream_context_.stream, stream_);

    // 解析model，获取必要结构
    const DataLayout input_layout =  model_->InputLayout(0);
    auto dtype = input_layout.dtype;
    auto order = input_layout.order;
    int data_size = 0;
    if (dtype == DataType::INT8 || dtype == DataType::UINT8){
        data_size = sizeof(uint8_t);
    }else if (dtype == DataType::FLOAT16 || dtype == DataType::UINT16 || dtype == DataType::INT16){
        data_size = sizeof(uint16_t);
    }else if (dtype == DataType::FLOAT32 || dtype == DataType::INT32){
        data_size = sizeof(uint32_t);
    }

    int model_h, model_w, model_c, model_b;
    auto shape = model_->InputShape(0);
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

    BufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(create_params));
    create_params.mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = model_b;
    create_params.size = model_h * model_w * model_c;
    create_params.size *= data_size;
    create_params.width = model_w;
    create_params.height = model_h;
    create_params.bytes_per_pix = data_size;
    create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;

    BufPool *pool = new BufPool;
    if (pool->CreatePool(&create_params, 3) < 0) {
        return -1;
    }

    pools_.emplace_back(pool);

    return 0;
}


int NvPreProcNPPPriv::Preproc_yolov5(BufSurfWrapperPtr src, BufSurfWrapperPtr dst)
{
    int batch_size = src->GetBatch();
    BufSurfaceColorFormat format = src->GetColorFormat();

    // 预分配resize和转fp32的显存
    int dst_w = dst->GetWidth();
    int dst_h = dst->GetHeight();

    int resize_mem_step = 0; 
    Npp8u * resize_mem = nppiMalloc_8u_C3(dst_w, dst_h, &resize_mem_step);

    int chw_mem_step;
    Npp8u * chw_mem = nppiMalloc_8u_C1(dst_w, dst_h * 3, &chw_mem_step);

    for (int i = 0; i < batch_size; i++){
        BufSurfaceParams *src_surf_param = src->GetSurfaceParams(i);
        int src_w = src_surf_param->width;
        int src_h = src_surf_param->height;

        void *src_data_ptr = src_surf_param->data_ptr;
        int src_data_size = src_surf_param->data_size;

        BufSurfaceParams *dst_surf_param = dst->GetSurfaceParams(i);
        int dst_w = dst_surf_param->width;
        int dst_h = dst_surf_param->height;

        void *dst_data_ptr = dst_surf_param->data_ptr;
        int dst_data_size = dst_surf_param->data_size;

        NppiSize src_size = {src_w, src_h};
        NppiRect src_roi  = {0, 0, src_w, src_h};

        int x = 0, y = 0, w = 0, h = 0;

        float ratio_w = (float) dst_w / src_w;
        float ratio_h = (float) dst_h / src_h;

        if (ratio_h > ratio_w){
            w = dst_w;
            h = src_h * ratio_w;
            y = (dst_h - h) / 2;
            x = 0;
        }else{
            w = src_w * ratio_h;
            h = dst_h;
            y = 0;
            x = (dst_w - w) / 2;
        }

        NppiSize dst_size = {dst_w, dst_h};
        NppiRect dst_roi  = {x, y, w, h};

        // 1. 预分配显存和设置114


        nppsSet_8u_Ctx(114, resize_mem, dst_w * dst_h * 3, npp_stream_context_);

        // 2. resize
        if (format == GDDEPLOY_BUF_COLOR_FORMAT_NV12 || format == GDDEPLOY_BUF_COLOR_FORMAT_NV21
            || format == GDDEPLOY_BUF_COLOR_FORMAT_YUV420){
            // nv12->bgr
            int bgr_mem_step = 0; 
            Npp8u * bgr_mem = nppiMalloc_8u_C3(src_w, src_h, &resize_mem_step);

            

            if (format == GDDEPLOY_BUF_COLOR_FORMAT_NV12 || format == GDDEPLOY_BUF_COLOR_FORMAT_NV21){
                const Npp8u* src_p3_ptr[2] = {(const Npp8u*)src_data_ptr, 
                        (const Npp8u*)(src_data_ptr+src_w*src_h)};

                nppiNV12ToBGR_8u_P2C3R_Ctx (src_p3_ptr, src_w, 
                    bgr_mem, src_w * 3, 
                    src_size, npp_stream_context_);
            }
            
            if (format == GDDEPLOY_BUF_COLOR_FORMAT_YUV420){
                const Npp8u* src_p3_ptr[3] = {(const Npp8u*)src_data_ptr, 
                        (const Npp8u*)(src_data_ptr+src_w*src_h),
                        (const Npp8u*)(src_data_ptr+src_w*src_h*5/4)};
                int rSrcStep[3] = {src_w, src_w / 2, src_w / 2};

                nppiYUV420ToBGR_8u_P3C3R_Ctx (src_p3_ptr, rSrcStep, 
                    bgr_mem, src_w * 3, 
                    src_size, npp_stream_context_);
            }

            // resize
            if (src_w < dst_w || src_h < dst_h){
                nppiResize_8u_C3R_Ctx((Npp8u*)bgr_mem, src_w * 3, src_size, src_roi,
                        resize_mem, dst_w * 3, dst_size, dst_roi, 
                        NPPI_INTER_NN, npp_stream_context_);
            } else {
                nppiResize_8u_C3R_Ctx((Npp8u*)bgr_mem, src_w * 3, src_size, src_roi,
                        resize_mem, dst_w * 3, dst_size, dst_roi, 
                        NPPI_INTER_LINEAR, npp_stream_context_);
            }
            // cv::Mat test(src_h, src_w, CV_8UC3);
            // cudaMemcpy(test.data, bgr_mem, src_w * src_h * 3, cudaMemcpyDeviceToHost);
            // cv::imwrite("/data/preds/nb212bgr.jpg", test);

            // 3. hwc2chw & bgr2rgb
            Npp8u* u8_rgb_planes[3] = {NULL,NULL,NULL};
            u8_rgb_planes[0] = chw_mem + dst_w * dst_h * 2;
            u8_rgb_planes[1] = chw_mem + dst_w * dst_h * 1;
            u8_rgb_planes[2] = chw_mem + dst_w * dst_h * 0;

            nppiCopy_8u_C3P3R_Ctx(resize_mem, dst_w * 3, u8_rgb_planes, dst_w, dst_size, npp_stream_context_);

            // 4. int8->fp32
            nppsConvert_8u32f_Ctx(chw_mem, (Npp32f *)dst_data_ptr, dst_data_size, npp_stream_context_);
        } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_BGR || format == GDDEPLOY_BUF_COLOR_FORMAT_RGB) {


            if (src_w < dst_w || src_h < dst_h){
                nppiResize_8u_C3R_Ctx((Npp8u*)src_data_ptr, src_w * 3, src_size, src_roi,
                        resize_mem, dst_w * 3, dst_size, dst_roi, 
                        NPPI_INTER_NN, npp_stream_context_);
            } else {
                nppiResize_8u_C3R_Ctx((Npp8u*)src_data_ptr, src_w * 3, src_size, src_roi,
                        resize_mem, dst_w * 3, dst_size, dst_roi, 
                        NPPI_INTER_LINEAR, npp_stream_context_);
            }

            // 3. hwc2chw & bgr2rgb
            Npp8u* u8_rgb_planes[3] = {NULL,NULL,NULL};
            if (format == GDDEPLOY_BUF_COLOR_FORMAT_BGR){
                u8_rgb_planes[0] = chw_mem + dst_w * dst_h * 2;
                u8_rgb_planes[1] = chw_mem + dst_w * dst_h * 1;
                u8_rgb_planes[2] = chw_mem + dst_w * dst_h * 0;
            } else {
                u8_rgb_planes[0] = chw_mem + dst_w * dst_h * 0;
                u8_rgb_planes[1] = chw_mem + dst_w * dst_h * 1;
                u8_rgb_planes[2] = chw_mem + dst_w * dst_h * 2;
            }

            nppiCopy_8u_C3P3R_Ctx(resize_mem, dst_w * 3, u8_rgb_planes, dst_w, dst_size, npp_stream_context_);

            // 4. int8->fp32
            nppsConvert_8u32f_Ctx(chw_mem, (Npp32f *)dst_data_ptr, dst_data_size, npp_stream_context_);
        } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER) {
            const Npp8u* src_p3_ptr[3] = {(const Npp8u*)src_data_ptr, 
                        (const Npp8u*)(src_data_ptr+src_w*src_h), 
                        (const Npp8u*)(src_data_ptr+src_w*src_h*2)};
            Npp8u* resize_mem_p3_ptr[3] = {resize_mem, resize_mem+dst_w*dst_h, resize_mem+dst_w*dst_h*2};
            if (src_w < dst_w || src_h < dst_h){
                nppiResize_8u_P3R_Ctx(src_p3_ptr, src_w, src_size, src_roi,
                        resize_mem_p3_ptr, dst_w, dst_size, dst_roi, 
                        NPPI_INTER_NN, npp_stream_context_);
            } else {
                nppiResize_8u_P3R_Ctx(src_p3_ptr, src_w, src_size, src_roi,
                        resize_mem_p3_ptr, dst_w, dst_size, dst_roi, 
                        NPPI_INTER_LINEAR, npp_stream_context_);
            }
            // cv::Mat test(dst_h, dst_w*3, CV_8UC1);
            // cudaMemcpy(test.data, resize_mem, dst_w * dst_h * 3, cudaMemcpyDeviceToHost);
            // cv::Mat channels[3] = { 
            //     cv::Mat(dst_h, dst_w, CV_8UC1, test.data+dst_w*dst_h*2),
            //     cv::Mat(dst_h, dst_w, CV_8UC1, test.data+dst_w*dst_h*1), 
            //     cv::Mat(dst_h, dst_w, CV_8UC1, test.data)
            //      };
            // cv::Mat img0;
            // cv::merge(channels, 3, img0);

            // cv::imwrite("/data/preds/resize.jpg", img0);

            // 4. int8->fp32
            nppsConvert_8u32f_Ctx(resize_mem, (Npp32f *)dst_data_ptr, dst_data_size, npp_stream_context_);
        } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER) {
            if (src_w < dst_w || src_h < dst_h){
                nppiResize_8u_P3R_Ctx((const Npp8u**)&src_data_ptr, src_w * 3, src_size, src_roi,
                        (Npp8u**)&resize_mem, dst_w * 3, dst_size, dst_roi, 
                        NPPI_INTER_NN, npp_stream_context_);
            } else {
                nppiResize_8u_P3R_Ctx((const Npp8u**)&src_data_ptr, src_w * 3, src_size, src_roi,
                        (Npp8u**)&resize_mem, dst_w * 3, dst_size, dst_roi, 
                        NPPI_INTER_LINEAR, npp_stream_context_);
            }

             // 4. int8->fp32
            nppsConvert_8u32f_Ctx(resize_mem, (Npp32f *)dst_data_ptr, dst_data_size, npp_stream_context_);
        }
        
        // cv::Mat test(dst_h, dst_w, CV_8UC3);
        // cudaMemcpy(test.data, dst_data_ptr, dst_w * dst_h * 3, cudaMemcpyDeviceToHost);
        // cv::imwrite("/data/preds/resize.jpg", test);

        // 5. -mean/std
        nppsDivC_32f_Ctx((Npp32f *)dst_data_ptr, 255.0, (Npp32f *)dst_data_ptr, dst_data_size, npp_stream_context_);

        cudaStreamSynchronize(stream_);
    }
    
    return 0;
}

int NvPreProcNPPPriv::Preproc_yolox(BufSurfWrapperPtr src, BufSurfWrapperPtr dst)
{
    return 0;
}

int NvPreProcNPPPriv::Preproc_classify(BufSurfWrapperPtr src, BufSurfWrapperPtr dst)
{
    return 0;
}

int NvPreProcNPPPriv::Preproc_seg(BufSurfWrapperPtr src, BufSurfWrapperPtr dst)
{
    return 0;
}

int NvPreProcNPPPriv::PreProc(BufSurfWrapperPtr src, BufSurfWrapperPtr dst)
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    int ret = 0;
    if (net_type == "classify"){
        // Preproc_classify(src, dst, model_h_, model_w_);
    } else if (net_type == "yolo"){
        ret = Preproc_yolov5(src, dst);
    } else if (net_type == "yolox"){
        ret = Preproc_yolox(src, dst);
    } else if (net_type == "OCRNet"){
        // Preproc_seg(src, dst, model_h_, model_w_);
    // }else if (net_type == "action"){
    //     Preproc_yolov5(src, dst, model_h_, model_w_);
    } else if (net_type == "dolg"){
        // Preproc_image_retrieval(src, dst, model_h_, model_w_);
    } else if (net_type == "arcface"){
        // Preproc_face_retrieval(src, dst, model_h_, model_w_);
    } else if (net_type == "ocr_det"){
        // Preproc_ocr_det(src, dst, model_h_, model_w_);
    } else if (net_type == "ocr_rec" || net_type == "resnet31v2ctc"){
        // Preproc_ocr_rec(src, dst, model_h_, model_w_);
    }

    return ret;
}

// Status Init() noexcept override;
Status NvPreProcNPP::Init(std::string config) noexcept 
{
    return gddeploy::Status::SUCCESS; 
}

Status NvPreProcNPP::Init(ModelPtr model, std::string config) 
{
    priv_ = std::make_shared<NvPreProcNPPPriv>(model);
    
    priv_->Init(config);

    return gddeploy::Status::SUCCESS; 
}

Status NvPreProcNPP::Process(PackagePtr pack) noexcept 
{
    int data_num = 0;
    for (auto &data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        BufSurface *surface = surf->GetBufSurface();
        data_num  += surface->batch_size;
    }   
    BufSurface src_surf;
    src_surf.mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
    src_surf.batch_size = data_num;
    src_surf.num_filled = 1;
    src_surf.is_contiguous = 0;    // AVFrame的两个plane地址不一定连续
    
    src_surf.surface_list = new BufSurfaceParams[data_num];
    int batch_idx = 0;
    BufSurfaceMemType mem_type;

    for (auto &data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        BufSurface *surface = surf->GetBufSurface();
        BufSurfaceParams *src_param = surf->GetSurfaceParams(0);
        mem_type = surface->mem_type;

        if (surface->mem_type == GDDEPLOY_BUF_MEM_NVIDIA){
            for (int i = 0; i < surface->batch_size; i++){
                src_surf.surface_list[batch_idx++] = *(src_param+i);
            }
        } else {    // 来着CPU，需要拷贝
            for (int i = 0; i < surface->batch_size; i++){
                // 图片大小不确定，无法预分配内存
                void *data_ptr = nullptr;
                cudaMalloc(&data_ptr, src_param->data_size);
                cudaMemcpy(data_ptr, src_param->data_ptr, src_param->data_size, cudaMemcpyHostToDevice);
                
                src_surf.surface_list[batch_idx] = *(src_param+i);
                src_surf.surface_list[batch_idx].data_ptr = data_ptr;
                batch_idx++;
            }
        }
    }

    BufSurfWrapperPtr in_ptr = std::make_shared<BufSurfaceWrapper>(&src_surf, false);

    BufSurfWrapperPtr out_ptr = priv_->RequestBuffer();

    int ret = priv_->PreProc(in_ptr, out_ptr);
    if (ret) {
        GDDEPLOY_ERROR("[register] [nv preproc npp] PreProc fail");
        return gddeploy::Status::ERROR_BACKEND;
    }

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(out_ptr));
    
    pack->predict_io = infer_data;

    if (mem_type != GDDEPLOY_BUF_MEM_NVIDIA){
        for (int i = 0; i < src_surf.batch_size; i++){
            cudaFree(src_surf.surface_list[i].data_ptr);
        }
    }

    return gddeploy::Status::SUCCESS; 
}