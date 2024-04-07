#include "bmnn_cv.h"

#include <string>
#include <vector>
#include <memory>

#include "core/cv.h"
#include "common/logger.h"

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"
// #include "bm_wrapper.hpp"
#include "bmcv_api_ext.h"
#include "bmnn_common.h"

namespace gddeploy
{

BmnnCV::BmnnCV() {
    bm_dev_request(&bm_handle_, 0);
}

BmnnCV::~BmnnCV() {
    bm_dev_free(bm_handle_);
}

int BmnnCV::Resize(BufSurfWrapperPtr src, BufSurfWrapperPtr dst) {
    


    return 0;
}

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

int BmnnCV::Crop(BufSurfWrapperPtr src, std::vector<BufSurfWrapperPtr> dst, std::vector<CropParam> crop_params) {
    // 将src转换为bm_image
    std::vector<bm_image> in_imgs;
    auto surf = src;
    BufSurface *surface = surf->GetBufSurface();
    BufSurfaceParams *src_param = surf->GetSurfaceParams(0);
    int plane_num = getPlaneNumByFormat(src_param->color_format);

    bm_image img;
    int stride[3] = {0};
    getStride(stride, convertSurfFormat2BmFormat(src_param->color_format), DATA_TYPE_EXT_1N_BYTE, src_param->width);
    bm_image_create(bm_handle_, src_param->height, src_param->width, 
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

    in_imgs.emplace_back(img);

    // 将dst转换为bm_image
    std::vector<bm_image> out_imgs;
    for (auto dst_surf_ptr : dst) {
        BufSurface *surf = dst_surf_ptr->GetBufSurface();
        BufSurfaceParams *dst_param = dst_surf_ptr->GetSurfaceParams(0);
        int plane_num = getPlaneNumByFormat(dst_param->color_format);

        bm_image img;
        int stride[3] = {0};
        getStride(stride, convertSurfFormat2BmFormat(dst_param->color_format), DATA_TYPE_EXT_1N_BYTE, dst_param->width);
        bm_image_create(bm_handle_, dst_param->height, dst_param->width, 
            convertSurfFormat2BmFormat(dst_param->color_format), DATA_TYPE_EXT_1N_BYTE, &img, stride);

        // bm_image_alloc_dev_mem_heap_mask(img, 6);
        if (surface->mem_type == GDDEPLOY_BUF_MEM_BMNN){
            bm_device_mem_t dev_mem[plane_num]; 
            if (dst_param->plane_params.data_ptr[0] != nullptr){
                for (int i = 0; i < plane_num; i++){
                    bm_set_device_mem(&dev_mem[i], dst_param->plane_params.psize[i], (unsigned long long)dst_param->plane_params.data_ptr[i]);
                }
            } else {
                bm_set_device_mem(&dev_mem[0], dst_param->data_size, (unsigned long long)dst_param->data_ptr);
            }

            if ( BM_SUCCESS != bm_image_attach(img, dev_mem)){
                GDDEPLOY_ERROR("[register] [bmnn preproc] bm_image_attach error !!!");
            }
        } else {
            void *buffer[3] = {dst_surf_ptr->GetData(0, 0), dst_surf_ptr->GetData(0, 0)+dst_param->height*dst_param->width, dst_surf_ptr->GetData(0, 0)+dst_param->height*dst_param->width*2};
            
            bm_image_alloc_dev_mem_heap_mask(img, 6);
            bm_image_copy_host_to_device(img, buffer);
        }

        out_imgs.emplace_back(img);
    }

    // resize
    int batch_size = 1;
    bm_image in_bm_img[batch_size];

    int out_num = out_imgs.size();
    bmcv_padding_atrr_t padding_attr[out_num];
    bmcv_rect rect[out_num];
    // memset(&padding_attr, 0, batch_size * sizeof(padding_attr));
    // memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = in_imgs.size();
    in_bm_img[0] = in_imgs[0];
    int rect_num[out_num] = {0};
    for (int i = 0; i < out_num; i++){
        rect[i].start_x = crop_params[i].start_x;
        rect[i].start_y = crop_params[i].start_y;
        rect[i].crop_w = crop_params[i].crop_w;
        rect[i].crop_h = crop_params[i].crop_h;

        int input_w = rect[i].crop_w;
        int input_h = rect[i].crop_h;
        int net_w = out_imgs[i].width;
        int net_h = out_imgs[i].height;

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
    }
    rect_num[0] = out_num;

    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle_, img_num, in_bm_img, out_imgs.data(), rect_num, rect, padding_attr)){
        GDDEPLOY_ERROR("[register] [bmnn preproc] bmcv yuv2rgb and resize error !!!");
        return -1; 
    }

    // for (int i = 0; i < out_num; i++){
    //     bm_image resize_img = *((bm_image*)out_imgs.data()+i);
    //     std::string pic_name = "/root/gddeploy/preds/resize"+std::to_string(i)+".jpg";
    //     save_rgb_pic((char*)pic_name.c_str(), bm_handle_, resize_img, resize_img.width, resize_img.height);
    // }

    // 释放bm_image
    for (auto img : in_imgs){
        bm_image_destroy(img);
    }
    for (auto img : out_imgs){
        bm_image_destroy(img);
    }

    return 0;
}

}