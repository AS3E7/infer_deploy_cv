#include <memory>
#include <string>
#include <math.h>
#include "ts_preproc.h"

#include "core/mem/buf_surface.h"
#include "core/preprocess.h"
#include "opencv2/opencv.hpp"

#include "core/model.h"
#include "core/mem/buf_surface_util.h"
#include "core/result_def.h"

#include "common/logger.h"
#include "common/type_convert.h"

#include "ts_common.h"
#include "ts_buffer.h"
#include "ts_comm_sys.h"
#include "ts_comm_vb.h"
#include "ts_comm_vdec.h"
#include "ts_defines.h"
#include "mpi_sys.h"
#include "mpi_vb.h"
#include "mpi_vdec.h"
#include "ts_math.h"
#include "semaphore.h"
#include "ts_type.h"
#include "mpi_vgs.h"

using namespace gddeploy;



namespace gddeploy{   
class TsPreProcPriv{
public:
    TsPreProcPriv(ModelPtr model):model_(model){
        for (int i = 0; i < model->InputNum(); i++){
            auto shape = model_->InputShape(i);
            model_h_ = shape[2];
            model_w_ = shape[3];
            model_c_ = shape[1];
            batch_num_ = shape[0];
        }
         
      

    }
    ~TsPreProcPriv(){
        for (auto pool : pools_){
            // pool->DestroyPool();
            delete pool;
        }
        pools_.clear();
    }

    int Init(std::string config); 

    int PreProc(BufSurfaceParams* in_img, BufSurfaceParams* out_img, std::vector<InferResult> result);

    int SetModel(ModelPtr model){
        if (model == nullptr){
            return -1;
        }else{
            model_ = model;
        }
        return 0;
    }

    ModelPtr GetModel(){
        return model_;
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
    int preproc_yolov5(BufSurfaceParams* in_img, BufSurfaceParams* out_img, std::vector<InferResult> result);

    ModelPtr model_;
    int model_h_;
    int model_w_;
    int model_c_;
    int batch_num_;

    std::vector<BufPool*> pools_;
};
}

static int CreatePool(ModelPtr model, BufPool *pool, BufSurfaceMemType mem_type, int block_count) {
    // 解析model，获取必要结构
    const DataLayout input_layout =  model->InputLayout(0);
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
    auto shape = model->InputShape(0);
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
    create_params.mem_type = GDDEPLOY_BUF_MEM_TS;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = model_b;
    create_params.size = model_h * model_w * model_c;
    create_params.size *= data_size;
    create_params.width = model_w;
    create_params.height = model_h;
    create_params.bytes_per_pix = data_size;
    create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_ARGB;

    if (pool->CreatePool(&create_params, block_count) < 0) {
        return -1;
    }
    return 0;
}


int TsPreProcPriv::Init(std::string config){
    // 预分配内存池
    size_t i_num = model_->InputNum();
    for (size_t i_idx = 0; i_idx < i_num; ++i_idx) {
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_TS, 3);
        pools_.emplace_back(pool);
    }

    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    if (net_type == "yolo" || net_type == "yolox" 
        || net_type == "classify" || net_type == "OCRNet"
        || net_type == "ocr_det" || net_type == "arcface"){
        int stride = ((model_w_ % 64 == 0) ? model_w_ : (model_w_ / 64 + 1) * 64);
    }

    return 0;
}

TS_S32 SAMPLE_YUV_RESIZE_ToArgb(VGS_TASK_ATTR_S *stVgsTaskAttr, VGS_SCALE_S *stVgsScale) 
{ 
    VGS_SCLCOEF_MODE_E penVgsSclCoefMode = VGS_SCLCOEF_NORMAL;
    TS_S32 s32Ret = TS_FAILURE;
    char time_start[60], time_end[60];
    VGS_HANDLE hHandle = -1;

    // step1:  Create VGS job
    s32Ret = TS_MPI_VGS_BeginJob(&hHandle);
    if (s32Ret != TS_SUCCESS) {
        printf("TS_MPI_VGS_BeginJob failed, s32Ret:0x%x", s32Ret);
        goto err;
    }
    // printf("#to call TS_MPI_VGS_BeginJob ok \n");

    // step2:  Add VGS task
    s32Ret = TS_MPI_VGS_AddScaleExTask(hHandle, stVgsTaskAttr, stVgsScale);
    if (s32Ret != TS_SUCCESS) {
        TS_MPI_VGS_CancelJob(hHandle);
        printf("TS_MPI_VGS_AddScaleTask failed, s32Ret:0x%x\n", s32Ret);
        goto err;
    }

    // step3:  Start VGS work
    s32Ret = TS_MPI_VGS_EndJob(hHandle);
    if (s32Ret != TS_SUCCESS) {
        // TS_MPI_VGS_CancelJob(hHandle);
        printf("TS_MPI_VGS_EndJob failed, s32Ret:0x%x\n", s32Ret);
        goto err;
    }	
    
    return s32Ret;
err:
	printf("resize error\n");   
	return s32Ret; 
}

static void save2file_render(void * buf, int len, char *fileName)
{
    int fd = -1;
    // unsigned long bytes;

    if (-1 == fd) {
        fd = open(fileName, O_WRONLY | O_CREAT | O_TRUNC,
                  S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        if (fd < 0) {
            printf("unable to create debug file.");
            fd = -2;
        }
    }

    if (fd > 0) {
        write(fd, buf, len);
        // SAMPLE_PRT(" file %s write bytes %ld image file closed .",fileName, bytes);
        close(fd);
        fd = -2;
    }

}
#include "ts_comm_vgs.h"
#define SAVE_ARGB_PIC 0
#define ALIGN(x, align) ((x % align == 0) ? x : (x / align + 1) * align)
#define ALIGN32(x) ALIGN(x, 32)
int TsPreProcPriv::preproc_yolov5(BufSurfaceParams* in_img, BufSurfaceParams* out_img, std::vector<InferResult> result)
{
        BufSurfaceParams *src_surf_param = in_img;
        BufSurfaceParams *dst_surf_param = out_img;


        VGS_TASK_ATTR_S vgs_task_attr;
        vgs_task_attr.stImgIn.stVFrame.u32Width = src_surf_param->width;
        vgs_task_attr.stImgIn.stVFrame.u32Height = ALIGN32(src_surf_param->height); //1088
        vgs_task_attr.stImgIn.stVFrame.enPixelFormat = PIXEL_FORMAT_NV_12;

        VIDEO_FRAME_INFO_S *pstFrameInfo = (VIDEO_FRAME_INFO_S *)src_surf_param->data_ptr;

        vgs_task_attr.stImgIn = *pstFrameInfo;
        // vgs_task_attr.stImgIn.stVFrame.u64VirAddr[0] = (int8_t *)src_surf_param->data_ptr;

// ret = av_image_copy_to_buffer((uint8_t*)vgs_task_attr.stImgIn.stVFrame.u64VirAddr[0], size, (const uint8_t *const *)frame->data,
// 					      (const int *)frame->linesize, frame->format, frame->width, frame->height,

        // vgs_task_attr.stImgOut.stVFrame.u32Width = dst_surf_param->width;
        // vgs_task_attr.stImgOut.stVFrame.u32Height = dst_surf_param->height;
        // vgs_task_attr.stImgOut.stVFrame.enPixelFormat = PIXEL_FORMAT_ARGB_8888;  
        vgs_task_attr.stImgOut = *(VIDEO_FRAME_INFO_S *)dst_surf_param->data_ptr;;

        int x = 0, y = 0, w = 0, h = 0;

        int net_w = dst_surf_param->width;
        int net_h = dst_surf_param->height;

        float ratio_w = (float) dst_surf_param->width / src_surf_param->width;
        float ratio_h = (float) dst_surf_param->height / src_surf_param->height;

        if (ratio_h > ratio_w){
            w = dst_surf_param->width;
            h = src_surf_param->height * ratio_w;
            y = (dst_surf_param->height - h) / 2;
            x = 0;
        }else{
            w = src_surf_param->width * ratio_h;
            h = dst_surf_param->height;
            y = 0;
            x = (dst_surf_param->width - w) / 2;
        }

        VGS_SCALE_S stVgsScale;
        stVgsScale.u32BgColor = 0X00000000;
        stVgsScale.stRect.s32X = x;
        stVgsScale.stRect.s32Y = y;
        stVgsScale.stRect.u32Width = w;
        stVgsScale.stRect.u32Height = h;

        // 调用gpu resize实现
        // printf("******************************************SAMPLE_YUV_RESIZE_ToArgb1\n");
        // auto t0 = std::chrono::high_resolution_clock::now();
        auto ret = SAMPLE_YUV_RESIZE_ToArgb(&vgs_task_attr, &stVgsScale);
        if (TS_SUCCESS != ret) {
            printf("SAMPLE_YUV_RESIZE_ToArgb err!\n");
            return -1;
        } 
        // usleep(5000);
        // printf("******************************************SAMPLE_YUV_RESIZE_ToArgb2\n");
        dst_surf_param->_reserved[0] = (void*)vgs_task_attr.stImgOut.stVFrame.u64PhyAddr[0];
        // auto t1 = std::chrono::high_resolution_clock::now();
        // printf("preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        // dst_surf_param->data_ptr = new char[640*640*3];

#if SAVE_ARGB_PIC
        char *tmp_data = new char[net_w*net_w*3];
        for (int i = 0; i < net_w; i++){
            for (int j = 0; j < net_w; j++){
                char *tmp = (char *)vgs_task_attr.stImgOut.stVFrame.u64VirAddr[0];
                char *tmp1 = (char *)tmp_data;
                *(tmp1+i*net_w*3+j*3+2) = *(tmp + i * net_w * 4 + j * 4 + 0);
                *(tmp1+i*net_w*3+j*3+1) = *(tmp + i * net_w * 4 + j * 4 + 1);
                *(tmp1+i*net_w*3+j*3+0) = *(tmp + i * net_w * 4 + j * 4 + 2);
            }
        }
        // memcpy(tmp_data, (void *)vgs_task_attr.stImgOut.stVFrame.u64VirAddr[0], 640*640*4);


		static int indexFile = 0;
        model_size = dst_surf_param->width;
		if (indexFile < 20) {
			char picName[64] = {0};
			sprintf(picName, "Img_%d_%d_%d_%d_%d.rgb", dst_surf_param->width, dst_surf_param->height, model_size, model_size, indexFile);
			int outsize = dst_surf_param->width*dst_surf_param->height*3;
			save2file_render((void*)tmp_data, outsize, picName);
		}
		indexFile++;
#endif

    return 0;
}

int TsPreProcPriv::PreProc(BufSurfaceParams* in_img, BufSurfaceParams* out_img, std::vector<InferResult> result)
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    int ret = 0;
    if (net_type == "yolo"){
        ret = preproc_yolov5(in_img, out_img, result);
    }
    
    return ret;
}

Status TsPreProc::Init(std::string config) noexcept
{ 
    ModelPtr model = GetParam<ModelPtr>("model_info");

    priv_ = std::make_shared<TsPreProcPriv>(model);

    priv_->Init(config);

    return gddeploy::Status::SUCCESS; 
}

Status TsPreProc::Init(ModelPtr model, std::string config)
{
    priv_ = std::make_shared<TsPreProcPriv>(model);

    priv_->Init(config);

    model_ = model;
    return gddeploy::Status::SUCCESS; 
}
#include "ts_buf_surface_impl.h"
Status TsPreProc::Process(PackagePtr pack) noexcept
{
    int batch_idx = 0;
    BufSurfWrapperPtr dst_buf = priv_->RequestBuffer();

    for (auto &data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        BufSurface *surface = surf->GetBufSurface();
        BufSurfaceParams *src_param = surf->GetSurfaceParams(0);

        // BufSurfaceParams src_param_data = *src_param;
        // VIDEO_FRAME_INFO_S *pstFrameInfo = nullptr;
        // if (surface->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
        //     pstFrameInfo = new VIDEO_FRAME_INFO_S;
        //     pstFrameInfo->stVFrame.u32Width = src_param_data.width;
        //     pstFrameInfo->stVFrame.u32Height = ALIGN_UP(src_param_data.height, 32); //1088
        //     pstFrameInfo->stVFrame.enPixelFormat = convertFormat(GDDEPLOY_BUF_COLOR_FORMAT_YUV420);
        //     SAMPLE_VGS_GetFrameVb(pstFrameInfo);

        //     // memcpy((char *)pstFrameInfo->stVFrame.u64VirAddr[0], src_param->data_ptr, src_param->data_size);

        //     src_param_data.data_ptr = pstFrameInfo;
        // }
        
        BufSurfaceParams *dst_param = dst_buf->GetSurfaceParams(0);

        if ( -1 == priv_->PreProc(src_param, dst_param, std::vector<InferResult>{})){
            return gddeploy::Status::NOT_IMPLEMENTED;
        }

        // if (surface->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
        //     SAMPLE_VGS_ReleaseFrameVb(pstFrameInfo);
        // }
    }
    
    // for (int i = 0; i < in_imgs.size(); i++){
    //     bm_image resize_img = in_imgs[i];
    //     std::string pic_name = "/gddeploy/preds/pre"+std::to_string(i)+".jpg";
    //     save_rgb_pic((char*)pic_name.c_str(), priv_->GetBmHandle(), resize_img, resize_img.width, resize_img.height);
    // }
    // auto t0 = std::chrono::high_resolution_clock::now();

    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(dst_buf));
    
    pack->predict_io =infer_data;

    return gddeploy::Status::SUCCESS; 
}