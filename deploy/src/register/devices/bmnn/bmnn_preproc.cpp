#include <memory>
#include <string>
#include <math.h>
#include "bmnn_preproc.h"

#include "core/mem/buf_surface.h"
#include "core/preprocess.h"
#include "opencv2/opencv.hpp"

#include "core/model.h"
#include "core/mem/buf_surface_util.h"
#include "core/result_def.h"

#include "common/logger.h"
#include "common/type_convert.h"

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"
// #include "bm_wrapper.hpp"
#include "bmcv_api_ext.h"
#include "bmnn_common.h"

using namespace gddeploy;


static int save_pic(char *pic_name, bm_handle_t  bm_handle, bm_image img)
{
    void *p_jpeg_data = NULL;
    size_t out_size = 0;
    int ret = bmcv_image_jpeg_enc(bm_handle, 1, &img, &p_jpeg_data, &out_size);
    if (ret != 0) {
        GDDEPLOY_ERROR("[register] [bmnn preproc] Can't save pic");
        return -1;
    }

    FILE *file_handle = fopen(pic_name, "wb+");
    if (file_handle != NULL) {
      fwrite(p_jpeg_data, out_size, 1, file_handle);
      fclose(file_handle);
    }

    if (p_jpeg_data) {
        free(p_jpeg_data);
    }

    return 0;
}
static int save_rgb_pic(char *pic_name, bm_handle_t  bm_handle, bm_image img, int w, int h)
{
    bm_image yuv_for_save_image;
    bmcv_rect yolo_rect = {0, 0, w, h};
    bm_image_create(bm_handle, h, w, FORMAT_YUV420P, DATA_TYPE_EXT_1N_BYTE, &yuv_for_save_image);

    if (img.image_format == FORMAT_NV12 || img.image_format == FORMAT_NV21 || img.image_format == FORMAT_YUV420P){
        yuv_for_save_image = img;
    } else {
        bmcv_image_vpp_convert(bm_handle, 1, img, &yuv_for_save_image, &yolo_rect);
    }
    save_pic(pic_name, bm_handle, yuv_for_save_image);
    bm_image_destroy(yuv_for_save_image);
    return 0;
}

namespace gddeploy{   
class BmnnPreProcPriv{
public:
    BmnnPreProcPriv(ModelPtr model):model_(model){
        for (int i = 0; i < model->InputNum(); i++){
            auto shape = model_->InputShape(i);
            model_h_ = shape[2];
            model_w_ = shape[3];
            model_c_ = shape[1];
            batch_num_ = shape[0];
        }
         
        p_bmrt_ = gddeploy::any_cast<std::shared_ptr<void>>(model->GetModel());

        const char **net_names;
        bmrt_get_network_names(p_bmrt_.get(), &net_names);
        std::string net_name(net_names[0]);
        free(net_names);

        net_info_ = bmrt_get_network_info(p_bmrt_.get(), net_name.c_str());
        if (NULL == net_info_) {
            GDDEPLOY_ERROR("[register] [bmnn preproc] ERROR: get net-info failed!");
        }

        // bm_handle_ = (bm_handle_t)bmrt_get_bm_handle(p_bmrt_.get());
        bm_dev_request(&bm_handle_, 0);
    }
    ~BmnnPreProcPriv(){
        for (int i = 0; i < batch_num_; i++){
            bm_image_destroy(*(resize_bmcv_.get()+i));
        }
        bm_dev_free(bm_handle_);

        for (auto pool : pools_){
            // pool->DestroyPool();
            delete pool;
        }
        pools_.clear();
    }

    int Init(std::string config); 

    int PreProc(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);

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

    std::vector<bm_image> RequestBMImages(){
        BufSurfWrapperPtr buf = pools_[0]->GetBufSurfaceWrapper();

        std::vector<bm_image> imgs = RequestBMImages(buf);

        return imgs;
    }

    std::vector<bm_image> RequestBMImages(BufSurfWrapperPtr &buf){
        // TODO: 这里要判断是否连续内存
        std::vector<bm_image> imgs;
        imgs.resize(batch_num_);

        bm_image_data_format_ext data_type = (BM_FLOAT32 == net_info_->input_dtypes[0]) ? DATA_TYPE_EXT_FLOAT32 : DATA_TYPE_EXT_1N_BYTE;

        int stride = ((model_w_ % 64 == 0) ? model_w_ : (model_w_/64+1) * 64);

        BufSurface *surf = buf->GetBufSurface();
        if (surf->is_contiguous){
            // bm_image_create_batch(bm_handle_, model_h_, model_w_, FORMAT_RGB_PLANAR, data_type, imgs.data(), batch_num_, &stride); 
            // for (int i = 0; i < batch_num_; i++){
            //     bm_image_create(bm_handle_, model_h_, model_w_, FORMAT_RGB_PLANAR, data_type, &imgs[i], &stride); 
            // }

            // bm_device_mem_t dev_mem;
            // bm_set_device_mem(&dev_mem, surf->batch_size * surf->surface_list[0].data_size, (unsigned long long)buf->GetData(0, 0));
            
            // if (BM_SUCCESS != bm_image_attach_contiguous_mem(batch_num_, imgs.data(), dev_mem)){
            //     GDDEPLOY_ERROR("[register] [bmnn preproc] bm_image_attach_contiguous_mem error !!!");
            // }
            for (int i = 0; i < batch_num_; i++){
                imgs[i] = *((bm_image *)surf->surface_list[i].data_ptr);
            }
        } else {
            for (int i = 0; i < batch_num_; i++){
                bm_image_create(bm_handle_, model_h_, model_w_, FORMAT_RGB_PLANAR, data_type, &imgs[i], &stride); 

                bm_device_mem_t dev_mem[surf->surface_list[0].plane_params.num_planes];
                for (int j = 0; j < surf->surface_list[0].plane_params.num_planes; j++){
                    bm_set_device_mem(&dev_mem[j], surf->surface_list[i].plane_params.psize[j], (unsigned long long)surf->surface_list[i].plane_params.data_ptr[j]);
                }
                // bm_set_device_mem(&dev_mem[j], surf->surface_list[i].data_size, (unsigned long long)surf->surface_list[i].data_ptr);
                
                if (BM_SUCCESS != bm_image_attach(imgs[i], dev_mem)){
                    GDDEPLOY_ERROR("[register] [bmnn preproc] bm_image_attach error !!!");
                }
            }
        }

        return imgs;

    }

    int GetModelWidth(){
        return model_w_;
    }

    int GetModelHeight(){
        return model_h_;
    }

    std::vector<bm_image> package2bmimage(PackagePtr pack);

    bm_handle_t GetBmHandle() { return bm_handle_; }

private:
    int preproc_yolov5(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);
    int preproc_yolox(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);
    int preproc_classify(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);
    int preproc_seg(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);
    int preproc_ocr_det(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);
    int preproc_ocr_rec(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);

    int preproc_image_retrieval(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);
    int preproc_face_retrieval(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result);

    ModelPtr model_;
    int model_h_;
    int model_w_;
    int model_c_;
    int batch_num_;

    bm_handle_t bm_handle_;
    std::shared_ptr<void> p_bmrt_;
    const bm_net_info_t *net_info_;

    std::vector<BufPool*> pools_;
    std::shared_ptr<bm_image> resize_bmcv_;
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
    create_params.mem_type = GDDEPLOY_BUF_MEM_BMNN;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = model_b;
    create_params.size = model_h * model_w * model_c;
    create_params.size *= data_size;
    create_params.width = model_w;
    create_params.height = model_h;
    create_params.bytes_per_pix = data_size;

    ModelPropertiesPtr mp = model->GetModelInfoPriv();
    std::string net_type = mp->GetNetType();
    if (net_type == "yolox")
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER;
    else
        create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;

    if (pool->CreatePool(&create_params, block_count) < 0) {
        return -1;
    }
    return 0;
}


int BmnnPreProcPriv::Init(std::string config){
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

    if (net_type == "yolo" || net_type == "yolox" 
        || net_type == "classify" || net_type == "OCRNet"
        || net_type == "ocr_det" || net_type == "arcface"){
        int stride = ((model_w_ % 64 == 0) ? model_w_ : (model_w_ / 64 + 1) * 64);
        bm_image_format_ext format = FORMAT_RGB_PLANAR;

        if (net_type == "yolox") {
            format = FORMAT_BGR_PLANAR;
        }

        resize_bmcv_ = std::shared_ptr<bm_image>(new bm_image[batch_num_], [](bm_image *p){delete [] p;});
        for (int i = 0; i < batch_num_; i++) {
            if(BM_SUCCESS != bm_image_create(bm_handle_,
                            model_h_,
                            model_w_,
                            format,
                            DATA_TYPE_EXT_1N_BYTE,
                            resize_bmcv_.get()+i,
                            &stride)){
                GDDEPLOY_ERROR("[MemAllocatorBmnn] bm create bm_image fail");
                return -1;     
            }
            if(BM_SUCCESS != bm_image_alloc_dev_mem(*(resize_bmcv_.get()+i))){
                GDDEPLOY_ERROR("[MemAllocatorBmnn] bm malloc device memory fail");
                return -1;                      
            }
        }
    }

    return 0;
}



int BmnnPreProcPriv::preproc_yolov5(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    // bm_handle_t handle;
    // bm_dev_request(&handle, 0);
    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];

    bmcv_padding_atrr_t padding_attr[batch_size];
    bmcv_rect rect[batch_size];
    // memset(&padding_attr, 0, batch_size * sizeof(padding_attr));
    // memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_img.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        bm_img[i] = in_img[i];
        int input_w = in_img[i].width;
        int input_h = in_img[i].height;

        float ratio_w = (float) net_w / input_w;
        float ratio_h = (float) net_h / input_h;
        
        padding_attr[i].dst_crop_sty = 0;
        padding_attr[i].dst_crop_stx = 0;
        padding_attr[i].padding_b = 114;
        padding_attr[i].padding_g = 114;
        padding_attr[i].padding_r = 114;
        padding_attr[i].if_memset = 1;

        if (ratio_h > ratio_w){
            padding_attr[i].dst_crop_w = net_w;
            padding_attr[i].dst_crop_h = input_h * ratio_w;
            padding_attr[i].dst_crop_sty = (net_h - padding_attr[i].dst_crop_h) / 2;
            padding_attr[i].dst_crop_stx = 0;
        }else{
            padding_attr[i].dst_crop_w = input_w * ratio_h;
            padding_attr[i].dst_crop_h = net_h;
            padding_attr[i].dst_crop_sty = 0;
            padding_attr[i].dst_crop_stx = (net_w - padding_attr[i].dst_crop_w) / 2;
        }
        rect[i].start_x = 0; 
        rect[i].start_y = 0;
        rect[i].crop_w = input_w;
        rect[i].crop_h = input_h;

        rect_num[i] = 1;
    }
    // for (int i = 0; i < 4; i++){
    //     bm_image resize_img = bm_img[i];
    //     std::string pic_name = "/gddeploy/preds/pre"+std::to_string(i)+".jpg";
    //     save_rgb_pic((char*)pic_name.c_str(), bm_handle_, resize_img, resize_img.width, resize_img.height);
    // }
    for (int i = 0; i < in_img.size(); i++){
        if (false == bm_image_is_attached(in_img[i])){
            GDDEPLOY_ERROR("[register] [bmnn preproc]2 bm_image_is_attached error !!!");
        }
    }

    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, bm_img, resize_bmcv_.get(), rect_num, rect, padding_attr)){
    // for (int i = 0; i < img_num; i++){
        // int ret = bmcv_image_vpp_convert_padding(bm_handle_, 1, bm_img[i], (bm_image*)resize_bmcv_.get()+i, &padding_attr[i], &rect[i]);
        // if (BM_SUCCESS != ret){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
            return -1; 
        // }
    }


    // for (int i = 0; i < 4; i++){
    //     bm_image resize_img = *((bm_image*)resize_bmcv_.get()+i);
    //     std::string pic_name = "/gddeploy/preds/resize"+std::to_string(i)+".jpg";
    //     save_rgb_pic((char*)pic_name.c_str(), bm_handle_, resize_img, resize_img.width, resize_img.height);
    // }

    float input_scale_255 = 1.0 / 255;
    float input_scale = input_scale_255 * net_info_->input_scales[0];
    // float input_scale = input_scale_255;
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = input_scale;
    convert_to_attr.beta_0 = 0;
    convert_to_attr.alpha_1 = input_scale;
    convert_to_attr.beta_1 = 0;
    convert_to_attr.alpha_2 = input_scale;
    convert_to_attr.beta_2 = 0;

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, (bm_image*)resize_bmcv_.get()+i, (bm_image*)out_img.data()+i)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv scale error !!!");
            return -1;
        }
    }
    // bm_dev_free(handle);

    return 0;
}


int BmnnPreProcPriv::preproc_yolox(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];

    bmcv_padding_atrr_t padding_attr[batch_size];
    bmcv_rect rect[batch_size];
    // memset(&padding_attr, 0, batch_size * sizeof(padding_attr));
    // memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_img.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        bm_img[i] = in_img[i];
        int input_w = in_img[i].width;
        int input_h = in_img[i].height;

        float ratio_w = (float) net_w / input_w;
        float ratio_h = (float) net_h / input_h;
        
        padding_attr[i].dst_crop_sty = 0;
        padding_attr[i].dst_crop_stx = 0;
        padding_attr[i].padding_b = 114;
        padding_attr[i].padding_g = 114;
        padding_attr[i].padding_r = 114;
        padding_attr[i].if_memset = 1;

        if (ratio_h > ratio_w){
            padding_attr[i].dst_crop_w = net_w;
            padding_attr[i].dst_crop_h = input_h * ratio_w;
            // padding_attr[i].dst_crop_sty = (net_h - padding_attr[i].dst_crop_h) / 2;
            padding_attr[i].dst_crop_stx = 0;
        }else{
            padding_attr[i].dst_crop_w = input_w * ratio_h;
            padding_attr[i].dst_crop_h = net_h;
            padding_attr[i].dst_crop_sty = 0;
            // padding_attr[i].dst_crop_stx = (net_w - padding_attr[i].dst_crop_w) / 2;
        }
        rect[i].start_x = 0; 
        rect[i].start_y = 0;
        rect[i].crop_w = input_w;
        rect[i].crop_h = input_h;

        rect_num[i] = 1;
    }


    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, bm_img, resize_bmcv_.get(), rect_num, rect, padding_attr)){
    // for (int i = 0; i < img_num; i++){
    //     int ret = bmcv_image_vpp_convert_padding(bm_handle, 1, bm_img[i], &resize_bmcv[i], &padding_attr[i], &rect[i]);
    //     if (BM_SUCCESS != ret){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
            return -1; 
        // }
    }

    float input_scale_255 = 1.0;
    float input_scale = input_scale_255 * net_info_->input_scales[0];
    // float input_scale = input_scale_255;
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = input_scale;
    convert_to_attr.beta_0 = 0;
    convert_to_attr.alpha_1 = input_scale;
    convert_to_attr.beta_1 = 0;
    convert_to_attr.alpha_2 = input_scale;
    convert_to_attr.beta_2 = 0;

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, (bm_image*)resize_bmcv_.get()+i, (bm_image*)out_img.data()+i)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv scale error !!!");
            return -1;
        }
    }

    return 0;
}

int BmnnPreProcPriv::preproc_classify(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];

    bmcv_rect rect[batch_size];
    // memset(&padding_attr, 0, batch_size * sizeof(padding_attr));
    // memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_img.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        bm_img[i] = in_img[i];
        int input_w = in_img[i].width;
        int input_h = in_img[i].height;

        rect[i].start_x = 0; 
        rect[i].start_y = 0;
        rect[i].crop_w = input_w;
        rect[i].crop_h = input_h;

        rect_num[i] = 1;
    }

    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, bm_img, resize_bmcv_.get(), rect_num, rect)){
    // for (int i = 0; i < img_num; i++){
    //     int ret = bmcv_image_vpp_convert_padding(bm_handle, 1, bm_img[i], &resize_bmcv[i], &padding_attr[i], &rect[i]);
    //     if (BM_SUCCESS != ret){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
            return -1; 
        // }
    }

    float input_scale = net_info_->input_scales[0];
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = 1 / 58.395 * input_scale;
    convert_to_attr.beta_0 = -1 / 58.395 * 123.68;
    convert_to_attr.alpha_1 = 1 / 57.12 * input_scale;
    convert_to_attr.beta_1 = -1 / 57.12 * 116.78;
    convert_to_attr.alpha_2 = 1 / 57.375 * input_scale;
    convert_to_attr.beta_2 = -1 / 57.375 * 103.94;

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, (bm_image*)resize_bmcv_.get()+i, (bm_image*)out_img.data()+i)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv scale error !!!");
            return -1;
        }
    }

    return 0;
}

int BmnnPreProcPriv::preproc_seg(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];

    bmcv_padding_atrr_t padding_attr[batch_size];
    bmcv_rect rect[batch_size];
    // memset(&padding_attr, 0, batch_size * sizeof(padding_attr));
    // memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_img.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        bm_img[i] = in_img[i];
        int input_w = in_img[i].width;
        int input_h = in_img[i].height;

        float ratio_w = (float) net_w / input_w;
        float ratio_h = (float) net_h / input_h;
        
        padding_attr[i].dst_crop_sty = 0;
        padding_attr[i].dst_crop_stx = 0;
        padding_attr[i].padding_b = 114;
        padding_attr[i].padding_g = 114;
        padding_attr[i].padding_r = 114;
        padding_attr[i].if_memset = 1;

        if (ratio_h > ratio_w){
            padding_attr[i].dst_crop_w = net_w;
            padding_attr[i].dst_crop_h = input_h * ratio_w;
            padding_attr[i].dst_crop_sty = (net_h - padding_attr[i].dst_crop_h) / 2;
            padding_attr[i].dst_crop_stx = 0;
        }else{
            padding_attr[i].dst_crop_w = input_w * ratio_h;
            padding_attr[i].dst_crop_h = net_h;
            padding_attr[i].dst_crop_sty = 0;
            padding_attr[i].dst_crop_stx = (net_w - padding_attr[i].dst_crop_w) / 2;
        }
        rect[i].start_x = 0; 
        rect[i].start_y = 0;
        rect[i].crop_w = input_w;
        rect[i].crop_h = input_h;

        rect_num[i] = 1;
    }


    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, bm_img, resize_bmcv_.get(), rect_num, rect, padding_attr)){
    // for (int i = 0; i < img_num; i++){
    //     int ret = bmcv_image_vpp_convert_padding(bm_handle, 1, bm_img[i], &resize_bmcv[i], &padding_attr[i], &rect[i]);
    //     if (BM_SUCCESS != ret){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
            return -1; 
        // }
    }

    float input_scale = net_info_->input_scales[0];
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = input_scale / 58.395;
    convert_to_attr.beta_0 = -123.675 / 58.395;
    convert_to_attr.alpha_1 = input_scale / 57.12;
    convert_to_attr.beta_1 = -116.28 / 57.12;
    convert_to_attr.alpha_2 = input_scale / 57.375;
    convert_to_attr.beta_2 = -103.53 / 57.375;

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, (bm_image*)resize_bmcv_.get()+i, (bm_image*)out_img.data()+i)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv scale error !!!");
            return -1;
        }
    }

    return 0;
}

int BmnnPreProcPriv::preproc_ocr_det(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];

    bmcv_rect rect[batch_size];
    memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_img.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        bm_img[i] = in_img[i];
        int input_w = in_img[i].width;
        int input_h = in_img[i].height;

        rect[i].start_x = 0; 
        rect[i].start_y = 0;
        rect[i].crop_w = input_w;
        rect[i].crop_h = input_h;

        rect_num[i] = 1;
    }

    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, bm_img, resize_bmcv_.get(), rect_num, rect)){
    // for (int i = 0; i < img_num; i++){
    //     int ret = bmcv_image_vpp_convert_padding(bm_handle, 1, bm_img[i], &resize_bmcv[i], &padding_attr[i], &rect[i]);
    //     if (BM_SUCCESS != ret){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
            return -1; 
        // }
    }

    float input_scale = net_info_->input_scales[0];
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = input_scale / 58.395;
    convert_to_attr.beta_0 = -123.675 / 58.395;
    convert_to_attr.alpha_1 = input_scale / 57.12;
    convert_to_attr.beta_1 = -116.28 / 57.12;
    convert_to_attr.alpha_2 = input_scale / 57.375;
    convert_to_attr.beta_2 = -103.53 / 57.375;

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, (bm_image*)resize_bmcv_.get()+i, (bm_image*)out_img.data()+i)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv scale error !!!");
            return -1;
        }
    }


    return 0;
}

int getDistance(PoseKeyPoint pointO, PoseKeyPoint point1)
{
    int distance;
    distance = powf((pointO.x - point1.x), 2) + powf((pointO.y - point1.y), 2);
    distance = sqrtf(distance);
    return distance;
}
int SortPointV2(std::vector<PoseKeyPoint> &sort_point)
{
    std::vector<PoseKeyPoint> new_sort_point;
    
    std::sort(sort_point.begin(), sort_point.end(), [](PoseKeyPoint a, PoseKeyPoint b){
        if (a.y < b.y){ return true; }
        if (a.y == b.y){ if (a.x < b.x) return true; }
        return false;
    });

    std::sort(sort_point.begin(), sort_point.begin()+2, [](PoseKeyPoint a, PoseKeyPoint b){
        if (a.x < b.x){ return true; }
        if (a.x == b.x){ if (a.y > b.y) return true; }
        return false;
    });

    new_sort_point.emplace_back(sort_point[0]);

    new_sort_point.emplace_back(sort_point[1]);
    
    std::sort(sort_point.begin()+2, sort_point.end(), [](PoseKeyPoint a, PoseKeyPoint b){
        if (a.x > b.x){ return true; }
        if (a.x == b.x){ if (a.y > b.y) return true; }
        return false;
    });

    new_sort_point.emplace_back(sort_point[2]);

    new_sort_point.emplace_back(sort_point[3]);

    sort_point = new_sort_point;

    return 0;
}

int BmnnPreProcPriv::preproc_ocr_rec(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    
    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    if (result.size() == 0){    // 不基于结果基础上裁剪
        int img_num = in_img.size();
        resize_bmcv_ = std::shared_ptr<bm_image>(new bm_image[img_num]);

        for (int i = 0; i < img_num; i++){
            bm_status_t bm_ret = bm_image_create(bm_handle_,
                                model_h_,
                                model_w_,
                                FORMAT_RGB_PLANAR,
                                DATA_TYPE_EXT_1N_BYTE,
                                resize_bmcv_.get()+i);
        }

        for (int i = 0; i < in_img.size(); i++){
            bm_image input = in_img[i];
            int input_w = in_img[i].width;
            int input_h = in_img[i].height;


            bmcv_padding_atrr_t padding_attr;
            memset(&padding_attr, 0, sizeof(padding_attr));
            padding_attr.dst_crop_sty = 0;
            padding_attr.dst_crop_stx = 0;
            padding_attr.padding_b = 0;
            padding_attr.padding_g = 0;
            padding_attr.padding_r = 0;
            padding_attr.if_memset = 1;
            float ratio_w = (float) net_w / input_w;
            float ratio_h = (float) net_h / input_h;
            if (ratio_h > ratio_w){
                padding_attr.dst_crop_w = net_w;
                padding_attr.dst_crop_h = input_h * ratio_w;
            }else{
                padding_attr.dst_crop_w = input_w * ratio_h;
                padding_attr.dst_crop_h = net_h;
            }

            bmcv_rect rect = {0, 0, input_w, input_h};
            bm_image *resize_bmcv = resize_bmcv_.get() + i;
            if (BM_SUCCESS != bmcv_image_vpp_convert_padding(bm_handle_, 1, input, resize_bmcv, &padding_attr, &rect)){
                std::cout << "2bmcv yuv2rgb and resize error !!!" << std::endl;
                return -1; 
            }

            // char pic_name[32] = {0};
            // sprintf(pic_name, "./preds/pre_ocr_rec_%d.jpg", i);
            // save_rgb_pic(pic_name, bm_handle, coord_imgs[i], coord_imgs[i].width, coord_imgs[i].height);

            //归一化
            float input_scale = net_info_->input_scales[0];
            bmcv_convert_to_attr convert_to_attr;
            convert_to_attr.alpha_0 = input_scale / 127.5;
            convert_to_attr.beta_0 = -127.5 / 127.5;
            convert_to_attr.alpha_1 = input_scale / 127.5;
            convert_to_attr.beta_1 = -127.5 / 127.5;
            convert_to_attr.alpha_2 = input_scale / 127.5;
            convert_to_attr.beta_2 = -127.5 / 127.5;
            if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, resize_bmcv, &out_img[i])){
                std::cout << "bmcv scale error !!!" << std::endl;
                return -1;
            }
        }
    } else {
        if (result[0].result_type[0] == GDD_RESULT_TYPE_OCR_DETECT){
            // 先统计有多少目标数量，预先分配内存
            int coord_num_all = 0;
            for (int i = 0; i < in_img.size(); i++){
                OcrDetectResult &ocr_detect_result = result[i].ocr_detect_result;
                OcrDetectImg &ocr_det_img = ocr_detect_result.ocr_detect_imgs[i];
                for (int j = 0; j < ocr_det_img.ocr_objs.size(); j++){
                    //坐标重排
                    coord_num_all += ocr_det_img.ocr_objs[j].point.size();
                }
            }


            resize_bmcv_ = std::shared_ptr<bm_image>(new bm_image[coord_num_all]);
            for (int i = 0; i < coord_num_all; i++){
                bm_status_t bm_ret = bm_image_create(bm_handle_,
                                    model_h_,
                                    model_w_,
                                    FORMAT_RGB_PLANAR,
                                    DATA_TYPE_EXT_1N_BYTE,
                                    resize_bmcv_.get()+i);
            }

            for (int i = 0; i < in_img.size(); i++){
                OcrDetectResult &ocr_detect_result = result[i].ocr_detect_result;
                OcrDetectImg &ocr_det_img = ocr_detect_result.ocr_detect_imgs[i];

                cv::Mat in_mat;
                cv::bmcv::toMAT(&in_img[i], in_mat);  

                int skip_num = 0;
                int coord_num_sum = 0;
                for (int j = 0; j < ocr_det_img.ocr_objs.size(); j++){
                    //坐标重排
                    std::vector<PoseKeyPoint> sort_point = ocr_det_img.ocr_objs[j].point;
                    SortPointV2(sort_point);

                    int distance_1 = getDistance(sort_point[0],sort_point[1]);
                    int distance_2 = getDistance(sort_point[0],sort_point[3]);
                    float img_crop_w = 0, img_crop_h = 0;
                    std::vector<cv::Point2f> src_points;
                    src_points.resize(4);

                    if (sort_point[0].x < sort_point[3].x){
                        src_points[0].x = sort_point[1].x;
                        src_points[0].y = sort_point[1].y;
                        src_points[1].x = sort_point[2].x;
                        src_points[1].y = sort_point[2].y;
                        src_points[2].x = sort_point[3].x;
                        src_points[2].y = sort_point[3].y;
                        src_points[3].x = sort_point[0].x;
                        src_points[3].y = sort_point[0].y;
                    }else{  //2310
                        src_points[0].x = sort_point[3].x;
                        src_points[0].y = sort_point[3].y;
                        src_points[1].x = sort_point[0].x;
                        src_points[1].y = sort_point[0].y;
                        src_points[2].x = sort_point[1].x;
                        src_points[2].y = sort_point[1].y;
                        src_points[3].x = sort_point[2].x;
                        src_points[3].y = sort_point[2].y;
                    }
                    std::vector<cv::Point2f> src_points_tmp = src_points;
                    src_points[0].x = src_points_tmp[1].x;
                    src_points[0].y = src_points_tmp[1].y;
                    src_points[1].x = src_points_tmp[0].x;
                    src_points[1].y = src_points_tmp[0].y;
                    src_points[2].x = src_points_tmp[3].x;
                    src_points[2].y = src_points_tmp[3].y;
                    src_points[3].x = src_points_tmp[2].x;
                    src_points[3].y = src_points_tmp[2].y;
                    int dst_h = std::min(distance_1, distance_2);
                    int dst_w = std::max(distance_1, distance_2);

                    // std::vector<cv::Point2f> dst_points = {{0.0f, 0.0f}, {0.0f, (float)dst_h}, 
                    //             {(float)dst_w, (float)dst_h}, {(float)dst_w, 0.0f}};
                    std::vector<cv::Point2f> dst_points = {{0.0f, 0.0f}, {0.0f, (float)dst_h}, 
                                {(float)dst_w, (float)dst_h}, {(float)dst_w, 0.0f}};

                    cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
                    cv::Mat coord_mat;
                    cv::warpPerspective(in_mat, coord_mat, M, cv::Size(dst_w, dst_h));
                    if (coord_mat.cols < 16 || coord_mat.rows < 16){
                        skip_num++;
                        continue;
                    }

                    bm_image coord_img;
                    auto status = cv::bmcv::toBMI(coord_mat, &coord_img, true);
                    if (BM_SUCCESS != status){
                        skip_num++;
                        continue;
                    }

                    int input_w = coord_img.width;
                    int input_h = coord_img.height;

                    bmcv_padding_atrr_t padding_attr;
                    memset(&padding_attr, 0, sizeof(padding_attr));
                    padding_attr.dst_crop_sty = 0;
                    padding_attr.dst_crop_stx = 0;
                    padding_attr.padding_b = 0;
                    padding_attr.padding_g = 0;
                    padding_attr.padding_r = 0;
                    padding_attr.if_memset = 1;
                    float ratio_w = (float) net_w / input_w;
                    float ratio_h = (float) net_h / input_h;
                    if (ratio_h > ratio_w){
                        padding_attr.dst_crop_w = net_w;
                        padding_attr.dst_crop_h = input_h * ratio_w;
                    }else{
                        padding_attr.dst_crop_w = input_w * ratio_h;
                        padding_attr.dst_crop_h = net_h;
                    }

                    bm_image *resize_bmcv = (bm_image*)resize_bmcv_.get() + coord_num_sum + j - skip_num;

                    bmcv_rect rect = {0, 0, input_w, input_h};
                    if (BM_SUCCESS != bmcv_image_vpp_convert_padding(bm_handle_, 1, coord_img, resize_bmcv, &padding_attr, &rect)){
                        std::cout << "2bmcv yuv2rgb and resize error !!!" << std::endl;
                        return -1; 
                    }

                    // char pic_name[32] = {0};
                    // sprintf(pic_name, "./preds/pre_ocr_rec_%d.jpg", i);
                    // save_rgb_pic(pic_name, bm_handle, coord_imgs[i], coord_imgs[i].width, coord_imgs[i].height);


                    //归一化
                    float input_scale = net_info_->input_scales[0];
                    bmcv_convert_to_attr convert_to_attr;
                    convert_to_attr.alpha_0 = input_scale / 127.5;
                    convert_to_attr.beta_0 = -127.5 / 127.5;
                    convert_to_attr.alpha_1 = input_scale / 127.5;
                    convert_to_attr.beta_1 = -127.5 / 127.5;
                    convert_to_attr.alpha_2 = input_scale / 127.5;
                    convert_to_attr.beta_2 = -127.5 / 127.5;
                    if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, resize_bmcv, &out_img[coord_num_sum+j-skip_num])){
                        std::cout << "bmcv scale error !!!" << std::endl;
                        return -1;
                    }
                }
            }
        } else if (result[0].result_type[0] == GDD_RESULT_TYPE_DETECT){
            // // 先统计有多少目标数量，预先分配内存
            // int coord_num_all = 0;
            // for (int i = 0; i < in_img.size(); i++){

            //     for (int i = 0; i < in_img.size(); i++){
            //         DetectResult &detect_result = result[i].detect_result;
            //         DetectImg &det_img = detect_result.detect_imgs[i];

            //         cv::Mat in_mat;
            //         cv::bmcv::toMAT(&in_img[i], in_mat);  

            //         int skip_num = 0;
            //         int coord_num_sum = 0;
            //         for (int j = 0; j < det_img.detect_objs.size(); j++){
            //             auto &obj = det_img.detect_objs[j];

            //             int input_w = det_img.width;
            //             int input_h = det_img.height;

            //             bmcv_padding_atrr_t padding_attr;
            //             memset(&padding_attr, 0, sizeof(padding_attr));
            //             padding_attr.dst_crop_sty = 0;
            //             padding_attr.dst_crop_stx = 0;
            //             padding_attr.padding_b = 0;
            //             padding_attr.padding_g = 0;
            //             padding_attr.padding_r = 0;
            //             padding_attr.if_memset = 1;
            //             float ratio_w = (float) net_w / input_w;
            //             float ratio_h = (float) net_h / input_h;
            //             if (ratio_h > ratio_w){
            //                 padding_attr.dst_crop_w = net_w;
            //                 padding_attr.dst_crop_h = input_h * ratio_w;
            //             }else{
            //                 padding_attr.dst_crop_w = input_w * ratio_h;
            //                 padding_attr.dst_crop_h = net_h;
            //             }

            //             bm_image *resize_bmcv = (bm_image*)resize_bmcv_.get() + coord_num_sum + j - skip_num;

            //             bmcv_rect rect = {0, 0, input_w, input_h};
            //             if (BM_SUCCESS != bmcv_image_vpp_convert_padding(bm_handle_, 1, coord_img, resize_bmcv, &padding_attr, &rect)){
            //                 std::cout << "2bmcv yuv2rgb and resize error !!!" << std::endl;
            //                 return -1; 
            //             }

            //             // char pic_name[32] = {0};
            //             // sprintf(pic_name, "./preds/pre_ocr_rec_%d.jpg", i);
            //             // save_rgb_pic(pic_name, bm_handle, coord_imgs[i], coord_imgs[i].width, coord_imgs[i].height);


            //             //归一化
            //             float input_scale = net_info_->input_scales[0];
            //             bmcv_convert_to_attr convert_to_attr;
            //             convert_to_attr.alpha_0 = input_scale / 127.5;
            //             convert_to_attr.beta_0 = -127.5 / 127.5;
            //             convert_to_attr.alpha_1 = input_scale / 127.5;
            //             convert_to_attr.beta_1 = -127.5 / 127.5;
            //             convert_to_attr.alpha_2 = input_scale / 127.5;
            //             convert_to_attr.beta_2 = -127.5 / 127.5;
            //             if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, resize_bmcv, &out_img[coord_num_sum+j-skip_num])){
            //                 std::cout << "bmcv scale error !!!" << std::endl;
            //                 return -1;
            //             }
            //         }
            //     }
            // }
        }
    }
    
    return 0;
}

int BmnnPreProcPriv::preproc_image_retrieval(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{

    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];

    bmcv_padding_atrr_t padding_attr[batch_size];
    bmcv_rect rect[batch_size];
    // memset(&padding_attr, 0, batch_size * sizeof(padding_attr));
    // memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_img.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        bm_img[i] = in_img[i];
        int input_w = in_img[i].width;
        int input_h = in_img[i].height;

        float ratio_w = (float) net_w / input_w;
        float ratio_h = (float) net_h / input_h;
        
        padding_attr[i].dst_crop_sty = 0;
        padding_attr[i].dst_crop_stx = 0;
        padding_attr[i].padding_b = 114;
        padding_attr[i].padding_g = 114;
        padding_attr[i].padding_r = 114;
        padding_attr[i].if_memset = 1;

        if (ratio_h > ratio_w){
            padding_attr[i].dst_crop_w = net_w;
            padding_attr[i].dst_crop_h = input_h * ratio_w;
            padding_attr[i].dst_crop_sty = (net_h - padding_attr[i].dst_crop_h) / 2;
            padding_attr[i].dst_crop_stx = 0;
        }else{
            padding_attr[i].dst_crop_w = input_w * ratio_h;
            padding_attr[i].dst_crop_h = net_h;
            padding_attr[i].dst_crop_sty = 0;
            padding_attr[i].dst_crop_stx = (net_w - padding_attr[i].dst_crop_w) / 2;
        }
        rect[i].start_x = 0; 
        rect[i].start_y = 0;
        rect[i].crop_w = net_w;
        rect[i].crop_h = net_h;

        rect_num[i] = 1;
    }
    
    for (int i = 0; i < in_img.size(); i++){
        if (false == bm_image_is_attached(in_img[i])){
            GDDEPLOY_ERROR("[register] [bmnn preproc]2 bm_image_is_attached error !!!");
            return -1;
        }
    }

    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, bm_img, resize_bmcv_.get(), rect_num, rect, padding_attr)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
            return -1; 
    }

    float input_scale = net_info_->input_scales[0];
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = input_scale / 58.395;
    convert_to_attr.beta_0 = -123.675 / 58.395;
    convert_to_attr.alpha_1 = input_scale / 57.12;
    convert_to_attr.beta_1 = -116.28 / 57.12;
    convert_to_attr.alpha_2 = input_scale / 57.375;
    convert_to_attr.beta_2 = -103.53 / 57.375;

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, (bm_image*)resize_bmcv_.get()+i, (bm_image*)out_img.data()+i)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv scale error !!!");
            return -1;
        }
    }

    return 0;
}

#include "../cpu/preproc/face_align.h"
int BmnnPreProcPriv::preproc_face_retrieval(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    auto input_shape = net_info_->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];
    int stride = ((net_w*3 % 64 == 0) ? net_w*3 : (net_w*3/64+1) * 64);

    for (int i = 0; i < batch_size; i++) {
        bm_image_create(bm_handle_, net_h, net_w, FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE_SIGNED, &bm_img[i], &stride); 
    }

    bmcv_rect rect[batch_size];
    memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_img.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        int input_w = in_img[i].width;
        int input_h = in_img[i].height;

        rect[i].start_x = 0; 
        rect[i].start_y = 0;
        rect[i].crop_w = net_w;
        rect[i].crop_h = net_h;

        rect_num[i] = 1;

        // 获取姿态结果
        DetectPoseObject detect_pose = result[0].detect_pose_result.detect_imgs[0].detect_objs[0];

        // 把人脸进行仿射变换
        cv::Mat in_img_mat;
        cv::bmcv::toMAT(&in_img[i], in_img_mat);

        float keypoint_data[10] = {0};
        for (int i = 0; i < detect_pose.point.size(); i++){
            keypoint_data[2*i] = detect_pose.point[i].x;
            keypoint_data[2*i+1] = detect_pose.point[i].y;
        }

        float arcface_srcf[10] = {38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041};
        cv::Mat arcface_src(5, 2, CV_32FC1, arcface_srcf);
        arcface_src = arcface_src.mul(net_w / 112.0f);
        cv::Mat kps(5, 2, CV_32FC1, keypoint_data);
        cv::Mat M = FacePreprocess::similarTransform(kps, arcface_src);
        
        M = M(Rect(0, 0, 3, 2));
        
        cv::Mat out(net_h, net_w, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::warpAffine(in_img_mat, out, M, cv::Size(net_w, net_h));

        // cv::imwrite("/gddeploy/preds/yuantu_out.jpg", out);
        cv::bmcv::toBMI(out, &bm_img[i], true);
    }

    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, bm_img, resize_bmcv_.get(), rect_num, rect)){
        GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
        return -1; 
    }

    // for (int i = 0; i < 1; i++){
    //     bm_image resize_img = *((bm_image*)resize_bmcv_.get()+i);
    //     std::string pic_name = "/gddeploy/preds/resize"+std::to_string(i)+".jpg";
    //     save_rgb_pic((char*)pic_name.c_str(), bm_handle_, resize_img, resize_img.width, resize_img.height);
    // }

    float input_scale = net_info_->input_scales[0];
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = 1 / 127.5 * input_scale;
    convert_to_attr.beta_0 = -1;
    convert_to_attr.alpha_1 = 1 / 127.5 * input_scale;
    convert_to_attr.beta_1 = -1;
    convert_to_attr.alpha_2 = 1 / 127.5 * input_scale;
    convert_to_attr.beta_2 = -1;

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle_, 1, convert_to_attr, (bm_image*)resize_bmcv_.get()+i, (bm_image*)out_img.data()+i)){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv scale error !!!");
            return -1;
        }
    }

    return 0;
}

int BmnnPreProcPriv::PreProc(std::vector<bm_image> &in_img, std::vector<bm_image> &out_img, std::vector<InferResult> result)
{
    ModelPropertiesPtr mp = model_->GetModelInfoPriv();
    std::string model_type = mp->GetModelType();
    std::string net_type = mp->GetNetType();

    int ret = 0;
    if (net_type == "classify"){
        preproc_classify(in_img, out_img, result);
    } else if (net_type == "yolo"){
        ret = preproc_yolov5(in_img, out_img, result);
    } else if (net_type == "yolox"){
        ret = preproc_yolox(in_img, out_img, result);
    } else if (net_type == "OCRNet"){
        preproc_seg(in_img, out_img, result);
    // }else if (net_type == "action"){
    //     preproc_yolov5(in_img, out_img, result);
    } else if (net_type == "dolg"){
        preproc_image_retrieval(in_img, out_img, result);
    } else if (net_type == "arcface"){
        preproc_face_retrieval(in_img, out_img, result);
    } else if (net_type == "ocr_det"){
        preproc_ocr_det(in_img, out_img, result);
    } else if (net_type == "ocr_rec" || net_type == "resnet31v2ctc"){
        preproc_ocr_rec(in_img, out_img, result);
    }
    
    return ret;
}
std::vector<bm_image> BmnnPreProcPriv::package2bmimage(PackagePtr pack)
{
    std::vector<bm_image> imgs;

    for (auto &data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        bm_image img;
        convertBufSurface2BmImage(img, surf);

        #if 0
        cv::Mat in_img;
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        convertBufSurface2Mat(in_img, surf, true);
        in_img.fromhardware = 1;

        auto src_param = surf->GetSurfaceParams();
        in_img = *((cv::Mat*)src_param->_reserved[0]);
        // cv::imwrite("/gddeploy/preds/test_mat.jpg", in_img);
        
        
        int img_h = src_param->height;
        int img_w = src_param->width;

        bm_image img;
        int stride = (((int)img_w % 64 == 0) ? img_w : (img_w/64+1) * 64);
        bm_image_create(bm_handle_, img_h, img_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &img, &stride);
        bm_image_alloc_dev_mem_heap_mask(img, 6);
        bm_image_from_mat(bm_handle_, in_img, img);
        // cv::bmcv::toBMI(in_img, &img, true);
        // void *buffer[1] = {surf->GetData(0, 0)};
        // bm_image_copy_host_to_device(img, buffer);

        // cv::Mat test;
        // cv::bmcv::toMAT(&img, test, true);
        // cv::imwrite("/data/gddeploy/preds/test_mat2.jpg", test);
        // save_rgb_pic("/gddeploy/preds/test_mat2.jpg", bm_handle_, img, img_w, img_h);

        // auto infer_data = pack->data[0];
        // ModelIO& in_mlu = infer_data->GetLref<ModelIO>();
        // Shape shape = in_mlu.shapes[0];
        // int img_h = shape[2];
        // int img_w = shape[3];
        // // memcpy((uint8_t *)img_mat.data(), in_mlu.buffers[0].Data(), in_mlu.buffers[0].MemorySize());
        // cv::Mat img_mat(img_h, img_w, CV_8UC3, in_mlu.buffers[0].MutableData());
        // bm_image img;
        // bm_image_create(bm_handle_, img_mat.cols, img_mat.rows, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &img);
        // bm_image_alloc_dev_mem_heap_mask(img, 6);
        // bm_image_from_mat(bm_handle_, img_mat, img);
        #endif

        imgs.emplace_back(img);
    }
    return imgs;
}


PackagePtr mat2package(bm_image)
{
    return nullptr;
}

Status BmnnPreProc::Init(std::string config) noexcept
{ 
    printf("Bmnn Init\n");

    //TODO: 这里要补充解析配置，得到网络类型等新型
    if (false == HaveParam("model_info")){
        return gddeploy::Status::INVALID_PARAM;
    }
    ModelPtr model = GetParam<ModelPtr>("model_info");

    priv_ = std::make_shared<BmnnPreProcPriv>(model);

    priv_->Init(config);

    return gddeploy::Status::SUCCESS; 
}

Status BmnnPreProc::Init(ModelPtr model, std::string config)
{
    priv_ = std::make_shared<BmnnPreProcPriv>(model);

    priv_->Init(config);

    model_ = model;

    return gddeploy::Status::SUCCESS; 
}

Status BmnnPreProc::Process(PackagePtr pack) noexcept
{
    // TODO: 判断pack数据是内存还是显存
    std::vector<bm_image> in_imgs;

    int batch_idx = 0;
    for (auto &data : pack->data){
        auto surf = data->GetLref<gddeploy::BufSurfWrapperPtr>();
        BufSurface *surface = surf->GetBufSurface();
        BufSurfaceParams *src_param = surf->GetSurfaceParams(0);
        int plane_num = getPlaneNumByFormat(src_param->color_format);

        bm_image img;
        int stride[3] = {0};
        getStride(stride, convertSurfFormat2BmFormat(src_param->color_format), DATA_TYPE_EXT_1N_BYTE, src_param->width);
        bm_image_create(priv_->GetBmHandle(), src_param->height, src_param->width, 
            convertSurfFormat2BmFormat(src_param->color_format), DATA_TYPE_EXT_1N_BYTE, &img, stride);

        // bm_image_alloc_dev_mem_heap_mask(img, 6);
        if (surface->mem_type == GDDEPLOY_BUF_MEM_BMNN){
            bm_device_mem_t dev_mem[plane_num]; 
            if (src_param->plane_params.data_ptr[0] != nullptr){
                for (int i = 0; i < plane_num; i++){
                    bm_set_device_mem(&dev_mem[i], src_param->plane_params.psize[i], (unsigned long long)src_param->plane_params.data_ptr[i]);
                }
            } else {
                bm_set_device_mem(&dev_mem[0], src_param->data_size, (unsigned long long)src_param->data_ptr);
            }

            if ( BM_SUCCESS != bm_image_attach(img, dev_mem)){
                GDDEPLOY_ERROR("[register] [bmnn preproc] bm_image_attach error !!!");
            }
        } else {
            void *buffer[3] = {surf->GetData(0, 0), surf->GetData(0, 0)+src_param->height*src_param->width, surf->GetData(0, 0)+src_param->height*src_param->width*2};
            
            bm_image_alloc_dev_mem_heap_mask(img, 6);
            bm_image_copy_host_to_device(img, buffer);
        }

        batch_idx++;
        in_imgs.emplace_back(img);
    }

    for (int i = 0; i < in_imgs.size(); i++){
        if (false == bm_image_is_attached(in_imgs[i])){
            GDDEPLOY_ERROR("[register] [bmnn preproc] bm_image_is_attached error !!!");
        }
    }
    // for (int i = 0; i < in_imgs.size(); i++){
    //     bm_image resize_img = in_imgs[i];
    //     std::string pic_name = "/root/gddeploy/preds/pre"+std::to_string(i)+".jpg";
    //     save_rgb_pic((char*)pic_name.c_str(), priv_->GetBmHandle(), resize_img, resize_img.width, resize_img.height);
    // }
    // auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<InferResult> result;
    for (auto &data : pack->data){
        if (data->HasMetaValue())
            result.emplace_back(data->GetMetaData<gddeploy::InferResult>());
    }

    BufSurfWrapperPtr buf = priv_->RequestBuffer();

    auto out_imgs = priv_->RequestBMImages(buf);
    // auto t0 = std::chrono::high_resolution_clock::now();
    int ret = priv_->PreProc(in_imgs, out_imgs, result);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    if (ret != 0){
        GDDEPLOY_ERROR("[register] [bmnn preproc] PreProc error !!!");
        for (auto in_img : in_imgs){
            bm_image_destroy(in_img);
        }
        return gddeploy::Status::ERROR_BACKEND;
    }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    std::shared_ptr<InferData> infer_data = std::make_shared<InferData>();
    infer_data->Set(std::move(buf));
    
    pack->predict_io =infer_data;

    for (auto in_img : in_imgs){
        bm_image_detach(in_img);
        bm_image_destroy(in_img);
    }



    return gddeploy::Status::SUCCESS; 
}