#include <memory>
#include <string>
#include <math.h>
#include "ascend_preproc.h"

#include "core/mem/buf_surface.h"
#include "core/preprocess.h"
#include "opencv2/opencv.hpp"

#include "core/model.h"
#include "core/mem/buf_surface_util.h"
#include "core/result_def.h"

#include "common/logger.h"
#include "common/type_convert.h"

#include "ascend_common.h"
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

using namespace gddeploy;

namespace gddeploy{   
class AscendPreProcPriv{
public:
    AscendPreProcPriv(ModelPtr model):model_(model){
        void *model_id_ptr = gddeploy::any_cast<std::shared_ptr<void>>(model->GetModel()).get();
        model_id_ = *(uint32_t *)model_id_ptr;
    }
    ~AscendPreProcPriv(){
        auto ret = acldvppDestroyChannel(dvpp_channel_desc_);
        if (ret != APP_ERR_OK) {
            GDDEPLOY_ERROR("Failed to destory dvpp channel, ret = {}.", ret);
            return ;
        }

        ret = acldvppDestroyChannelDesc(dvpp_channel_desc_);
        if (ret != APP_ERR_OK) {
            GDDEPLOY_ERROR("Failed to destroy dvpp channel description, ret = {}.", ret);
            return ;
        }

        aclrtDestroyStream(stream_);
        aclrtDestroyContext(context_model_);
    }

    int Init(std::string config); 

    int PreProc(BufSurface *src_surf, BufSurface *dst_surf);

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


private:
    int preproc_classify(BufSurface *input, BufSurface *output);
    int preproc_yolo(BufSurface *input, BufSurface *output);

    ModelPtr model_;
    
    aclrtContext context_model_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t model_id_;

    acldvppChannelDesc *dvpp_channel_desc_ = nullptr;
    aclrtStream dvpp_stream_ = nullptr;
    bool withSynchronize_ = true;

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
    create_params.mem_type = GDDEPLOY_BUF_MEM_ASCEND_DVPP;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = model_b;
    // create_params.size = model_h * model_w * model_c;create_params.size

    // Get output buffer size for resize output
    auto ret = GetVpcDataSize(model_w, model_h, convertFormat(GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER), create_params.size);
    if (ret != APP_ERR_OK) {
        return ret;
    }
    create_params.size *= data_size;
    create_params.width = model_w;
    create_params.height = model_h;
    create_params.bytes_per_pix = data_size;
    create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;

    if (pool->CreatePool(&create_params, block_count) < 0) {
        return -1;
    }
    return 0;
}


int AscendPreProcPriv::Init(std::string config){
    // 预分配内存池
    size_t i_num = model_->InputNum();
    for (size_t i_idx = 0; i_idx < i_num; ++i_idx) {
        BufPool *pool = new BufPool;
        CreatePool(model_, pool, GDDEPLOY_BUF_MEM_BMNN, 3);
        pools_.emplace_back(pool);
    }

    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType(); 

    aclrtCreateContext(&context_model_, model_id_);

    aclrtCreateStream(&stream_);

    dvpp_channel_desc_ = acldvppCreateChannelDesc();
    if (dvpp_channel_desc_ == nullptr) {
        return -1;
    }
    APP_ERROR ret = acldvppCreateChannel(dvpp_channel_desc_);
    if (ret != 0) {
        GDDEPLOY_ERROR("Failed to create dvpp channel: {}.", ret);
        acldvppDestroyChannelDesc(dvpp_channel_desc_);
        dvpp_channel_desc_ = nullptr;
        return ret;
    } 

    return 0;
}

// void GetCropRoi(const DvppDataInfo &input, const DvppDataInfo &output, VpcProcessType processType,
//                             CropRoiConfig &cropRoi) const
// {
//     // When processType is not VPC_PT_FILL, crop area is the whole input image
//     if (processType != VPC_PT_FILL) {
//         cropRoi.right = CONVERT_TO_ODD(input.width - ODD_NUM_1);
//         cropRoi.down = CONVERT_TO_ODD(input.height - ODD_NUM_1);
//         return;
//     }

//     bool widthRatioSmaller = true;
//     // The scaling ratio is based on the smaller ratio to ensure the smallest edge to fill the targe edge
//     float resizeRatio = static_cast<float>(input.width) / output.width;
//     if (resizeRatio > (static_cast<float>(input.height) / output.height)) {
//         resizeRatio = static_cast<float>(input.height) / output.height;
//         widthRatioSmaller = false;
//     }

//     const int halfValue = 2;
//     // The left and up must be even, right and down must be odd which is required by acl
//     if (widthRatioSmaller) {
//         cropRoi.left = 0;
//         cropRoi.right = CONVERT_TO_ODD(input.width - ODD_NUM_1);
//         cropRoi.up = CONVERT_TO_EVEN(static_cast<uint32_t>((input.height - output.height * resizeRatio) / halfValue));
//         cropRoi.down = CONVERT_TO_ODD(input.height - cropRoi.up - ODD_NUM_1);
//         return;
//     }

//     cropRoi.up = 0;
//     cropRoi.down = CONVERT_TO_ODD(input.height - ODD_NUM_1);
//     cropRoi.left = CONVERT_TO_EVEN(static_cast<uint32_t>((input.width - output.width * resizeRatio) / halfValue));
//     cropRoi.right = CONVERT_TO_ODD(input.width - cropRoi.left - ODD_NUM_1);
//     return;
// }

// void GetPasteRoi(const DvppDataInfo &input, const DvppDataInfo &output, VpcProcessType processType,
//                              CropRoiConfig &pasteRoi) const
// {
//     if (processType == VPC_PT_FILL) {
//         pasteRoi.right = CONVERT_TO_ODD(output.width - ODD_NUM_1);
//         pasteRoi.down = CONVERT_TO_ODD(output.height - ODD_NUM_1);
//         return;
//     }

//     bool widthRatioLarger = true;
//     // The scaling ratio is based on the larger ratio to ensure the largest edge to fill the targe edge
//     float resizeRatio = static_cast<float>(input.width) / output.width;
//     if (resizeRatio < (static_cast<float>(input.height) / output.height)) {
//         resizeRatio = static_cast<float>(input.height) / output.height;
//         widthRatioLarger = false;
//     }

//     // Left and up is 0 when the roi paste on the upper left corner
//     if (processType == VPC_PT_PADDING) {
//         pasteRoi.right = (input.width / resizeRatio) - ODD_NUM_1;
//         pasteRoi.down = (input.height / resizeRatio) - ODD_NUM_1;
//         pasteRoi.right = CONVERT_TO_ODD(pasteRoi.right);
//         pasteRoi.down = CONVERT_TO_ODD(pasteRoi.down);
//         return;
//     }

//     const int halfValue = 2;
//     // Left and up is 0 when the roi paste on the middler location
//     if (widthRatioLarger) {
//         pasteRoi.left = 0;
//         pasteRoi.right = output.width - ODD_NUM_1;
//         pasteRoi.up = (output.height - (input.height / resizeRatio)) / halfValue;
//         pasteRoi.down = output.height - pasteRoi.up - ODD_NUM_1;
//     } else {
//         pasteRoi.up = 0;
//         pasteRoi.down = output.height - ODD_NUM_1;
//         pasteRoi.left = (output.width - (input.width / resizeRatio)) / halfValue;
//         pasteRoi.right = output.width - pasteRoi.left - ODD_NUM_1;
//     }

//     // The left must be even and align to 16, up must be even, right and down must be odd which is required by acl
//     pasteRoi.left = DVPP_ALIGN_UP(CONVERT_TO_EVEN(pasteRoi.left), VPC_WIDTH_ALIGN);
//     pasteRoi.right = CONVERT_TO_ODD(pasteRoi.right);
//     pasteRoi.up = CONVERT_TO_EVEN(pasteRoi.up);
//     pasteRoi.down = CONVERT_TO_ODD(pasteRoi.down);
//     return;
// }


APP_ERROR SetDvppPicDescData(const BufSurfaceParams *surf_param, acldvppPicDesc &picDesc) 
{
    // 计算input output stride size
    uint32_t width_stride = 0;
    uint32_t height_stride = 0;
    APP_ERROR ret = GetVpcInputStrideSize(surf_param->width, surf_param->height, convertFormat(surf_param->color_format),
        width_stride, height_stride);
    if (ret != APP_ERR_OK) {
        return ret;
    }

    ret = acldvppSetPicDescData(&picDesc, surf_param->data_ptr);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("Failed to set data for dvpp picture description, ret = {}.", ret);
        return ret;
    }
    ret = acldvppSetPicDescSize(&picDesc, surf_param->data_size);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("Failed to set size for dvpp picture description, ret = {}.", ret);
        return ret;
    }
    ret = acldvppSetPicDescFormat(&picDesc, convertFormat(surf_param->color_format));
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("Failed to set format for dvpp picture description, ret = {}.", ret);
        return ret;
    }
    ret = acldvppSetPicDescWidth(&picDesc, surf_param->width);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("Failed to set width for dvpp picture description, ret = {}.", ret);
        return ret;
    }
    ret = acldvppSetPicDescHeight(&picDesc, surf_param->height);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("Failed to set height for dvpp picture description, ret = {}.", ret);
        return ret;
    }
    ret = acldvppSetPicDescWidthStride(&picDesc, width_stride);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("Failed to set aligned width for dvpp picture description, ret = {}.", ret);
        return ret;
    }
    ret = acldvppSetPicDescHeightStride(&picDesc, height_stride);
    if (ret != APP_ERR_OK) {
        GDDEPLOY_ERROR("Failed to set aligned height for dvpp picture description, ret = {}.", ret);
        return ret;
    }

    return APP_ERR_OK;
}

int AscendPreProcPriv::preproc_classify(BufSurface *input, BufSurface *output)
{
    // // 创建输入输出描述符和设置参数
    // acldvppPicDesc *inputDesc = acldvppCreatePicDesc();
    // acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
    // SetDvppPicDescData(input, inputDesc);
    // SetDvppPicDescData(output, outputDesc);

    // acldvppResizeConfig *resizeConfig = acldvppCreateResizeConfig();
    // if (resizeConfig == nullptr) {
    //     GDDEPLOY_ERROR("Failed to create dvpp resize config.";
    //     return APP_ERR_COMM_INVALID_POINTER;
    // }

    // APP_ERROR ret = acldvppVpcResizeAsync(dvpp_channel_desc_, &inputDesc, &outputDesc, resizeConfig, dvpp_stream_);
    // if (ret != APP_ERR_OK) {
    //     GDDEPLOY_ERROR("Failed to resize asynchronously, ret = {}.", ret);
    //     return ret;
    // }

    // if (withSynchronize_) {
    //     ret = aclrtSynchronizeStream(dvpp_stream_);
    //     if (ret != APP_ERR_OK) {
    //         GDDEPLOY_ERROR("Failed to synchronize stream, ret = {}.", ret);
    //         return ret;
    //     }
    // }

    return 0;
}

int AscendPreProcPriv::preproc_yolo(BufSurface *input, BufSurface *output)
{
    for (int i = 0; i < input->batch_size; i++){
        BufSurfaceParams surf_param = input->surface_list[i];
        BufSurfaceParams surf_param_out = output->surface_list[i];
        acldvppRoiConfig *cropRoiCfg = acldvppCreateRoiConfig(0, surf_param.width, 0, surf_param.height);
        if (cropRoiCfg == nullptr) {
            GDDEPLOY_ERROR("Failed to create dvpp roi config for corp area.");
            return APP_ERR_COMM_FAILURE;
        }

        int w, h, x, y;

        int input_w = surf_param.width;
        int input_h = surf_param.height;
        int model_w = surf_param_out.width;
        int model_h = surf_param_out.height;
        float r_w = model_h / (input_w * 1.0);
        float r_h = model_w / (input_h * 1.0);

        if (r_h > r_w)
        {
            w = model_h;
            h = r_w * input_h;
            x = 0;
            y = (model_w - h) / 2;
        }
        else
        {
            w = r_h * input_w;
            h = model_w;
            x = (model_h - w) / 2;
            y = 0;
        }

        acldvppRoiConfig *pastRoiCfg = acldvppCreateRoiConfig(x, x + w, y, y + h);
        if (pastRoiCfg == nullptr) {
            GDDEPLOY_ERROR("Failed to create dvpp roi config for paster area.");
            return APP_ERR_COMM_FAILURE;
        }

        acldvppPicDesc *inputDesc = acldvppCreatePicDesc();
        acldvppPicDesc *outputDesc = acldvppCreatePicDesc();
        SetDvppPicDescData(&surf_param, *inputDesc);
        SetDvppPicDescData(&surf_param_out, *outputDesc);

        APP_ERROR ret = acldvppVpcCropAndPasteAsync(dvpp_channel_desc_, inputDesc, outputDesc, cropRoiCfg,
                                                    pastRoiCfg, dvpp_stream_);
        if (ret != APP_ERR_OK) {
            // release resource.
            GDDEPLOY_ERROR("Failed to crop and paste asynchronously, ret = {}.", ret);
            return ret;
        }
        if (withSynchronize_) {
            ret = aclrtSynchronizeStream(dvpp_stream_);
            if (ret != APP_ERR_OK) {
                GDDEPLOY_ERROR("Failed tp synchronize stream, ret = {}.", ret);
                return ret;
            }
        }
    }

    return 0;
}


int AscendPreProcPriv::PreProc(BufSurface *src_surf, BufSurface *dst_surf)
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    int ret = 0;
    if (net_type == "classify"){
        ret = preproc_classify(src_surf, dst_surf);
    } else if (net_type == "yolo"){
        ret = preproc_yolo(src_surf, dst_surf);
    // } else if (net_type == "yolox"){
    //     ret = preproc_yolox(in_img, out_img, result);
    // } else if (net_type == "OCRNet"){
    //     preproc_seg(in_img, out_img, result);
    // // }else if (net_type == "action"){
    // //     preproc_yolov5(in_img, out_img, result);
    // } else if (net_type == "dolg"){
    //     preproc_image_retrieval(in_img, out_img, result);
    // } else if (net_type == "arcface"){
    //     preproc_face_retrieval(in_img, out_img, result);
    // } else if (net_type == "ocr_det"){
    //     preproc_ocr_det(in_img, out_img, result);
    // } else if (net_type == "ocr_rec" || net_type == "resnet31v2ctc"){
    //     preproc_ocr_rec(in_img, out_img, result);
    }
    
    return ret;
}

Status AscendPreProc::Init(std::string config) noexcept
{ 
    printf("Ascend Init\n");

    //TODO: 这里要补充解析配置，得到网络类型等新型
    if (false == HaveParam("model_info")){
        return gddeploy::Status::INVALID_PARAM;
    }
    ModelPtr model = GetParam<ModelPtr>("model_info");

    priv_ = std::make_shared<AscendPreProcPriv>(model);

    priv_->Init(config);

    return gddeploy::Status::SUCCESS; 
}

Status AscendPreProc::Init(ModelPtr model, std::string config)
{
    priv_ = std::make_shared<AscendPreProcPriv>(model);

    priv_->Init(config);

    model_ = model;

    return gddeploy::Status::SUCCESS; 
}

Status AscendPreProc::Process(PackagePtr pack) noexcept
{
    int data_num = 0;
    for (auto &data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        BufSurface *surface = surf->GetBufSurface();
        data_num  += surface->batch_size;
    }   
    BufSurface src_surf;
    src_surf.mem_type = GDDEPLOY_BUF_MEM_ASCEND_DVPP;
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

        if (surface->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP){
            for (int i = 0; i < surface->batch_size; i++){
                src_surf.surface_list[batch_idx++] = *(src_param+i);
            }
        } else {    // 来着CPU，需要拷贝 
            for (int i = 0; i < surface->batch_size; i++){
                // 图片大小不确定，无法预分配内存
                void *data_ptr = nullptr;
                auto ret = acldvppMalloc((void **)(&(data_ptr)), src_param->data_size);
                if (ret != APP_ERR_OK) {
                    GDDEPLOY_ERROR("Failed to malloc {} bytes on dvpp for resize, ret = {}.", src_param->data_size, ret);
                    return gddeploy::Status::ERROR_BACKEND;
                }

                aclrtMemset(data_ptr, src_param->data_size, 114, src_param->data_size);
                aclrtMemcpy(data_ptr, src_param->data_size, src_param->data_ptr, src_param->data_size, ACL_MEMCPY_HOST_TO_DEVICE);
                
                src_surf.surface_list[batch_idx] = *(src_param+i);
                src_surf.surface_list[batch_idx].data_ptr = data_ptr;
                batch_idx++;
            }
        }
    }

    // 申请输出内存
    BufSurfWrapperPtr buf = priv_->RequestBuffer();
    BufSurface *dst_surf = buf->GetBufSurface();

    // auto t0 = std::chrono::high_resolution_clock::now();
    int ret = priv_->PreProc(&src_surf, dst_surf);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    if (ret != 0){
        GDDEPLOY_ERROR("[register] [ascend preproc] PreProc error !!!");
        return gddeploy::Status::ERROR_BACKEND;
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(buf));
    
    pack->predict_io =infer_data;

    if (mem_type != GDDEPLOY_BUF_MEM_ASCEND_DVPP){
        for (int i = 0; i < src_surf.batch_size; i++){
            aclrtFree(src_surf.surface_list[i].data_ptr);
        }
    }

    return gddeploy::Status::SUCCESS; 
}