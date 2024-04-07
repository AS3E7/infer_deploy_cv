#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <vector>

#include "yolov5_common.h"

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"
#include "bm_wrapper.hpp"

int bmnn_yolov5_preproc(bm_handle_t bm_handle, const bm_net_info_t *net_info, std::vector<cv::Mat> img_mats, bm_image *out_imgs)
{
    auto input_shape = net_info->stages[0].input_shapes[0];
    int batch_size = input_shape.dims[0];
    int net_h = input_shape.dims[2];
    int net_w = input_shape.dims[3];

    bm_image bm_img[batch_size];

    bmcv_padding_atrr_t padding_attr[batch_size];
    bmcv_rect rect[batch_size];
    // memset(&padding_attr, 0, batch_size * sizeof(padding_attr));
    // memset(&rect, 0, batch_size * sizeof(bmcv_rect));

    int img_num = img_mats.size();
    int rect_num[batch_size] = {0};
    for (int i = 0; i < img_num; i++){
        cv::Mat img_mat = img_mats[i];
        int input_w = img_mat.cols;
        int input_h = img_mat.rows;

        
        int stride = ((input_w % 64 == 0) ? input_w : (input_w/64+1) * 64);

        // bm_image_create(bm_handle, input_h, input_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &bm_img[i], &stride);
        bm_image_from_mat(bm_handle, img_mat, bm_img[i]);

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

    // 分配输出的img的空间
    bm_image resize_bmcv[batch_size];
    bm_status_t bm_ret = bm_image_create_batch(bm_handle,
                            net_h,
                            net_w,
                            FORMAT_RGB_PLANAR,
                            DATA_TYPE_EXT_1N_BYTE,
                            resize_bmcv,
                            batch_size);
    if (BM_SUCCESS != bmcv_image_vpp_basic(bm_handle, img_num, bm_img, resize_bmcv, rect_num, rect, padding_attr)){
    // for (int i = 0; i < img_num; i++){
    //     int ret = bmcv_image_vpp_convert_padding(bm_handle, 1, bm_img[i], &resize_bmcv[i], &padding_attr[i], &rect[i]);
    //     if (BM_SUCCESS != ret){
            std::cout << "bmcv yuv2rgb and resize error !!!" << std::endl;
            return -1; 
        // }
    }


    //归一化
    bm_image_data_format_ext data_type = (BM_FLOAT32 == net_info->input_dtypes[0]) ? DATA_TYPE_EXT_FLOAT32 : DATA_TYPE_EXT_1N_BYTE_SIGNED;
    bm_ret = bm_image_create_batch (bm_handle,
                                net_h,
                                net_w,
                                FORMAT_RGB_PLANAR,
                                data_type,
                                out_imgs,
                                batch_size);

    float input_scale_255 = 1.0 / 255;
    float input_scale = input_scale_255 * net_info->input_scales[0];
    // float input_scale = input_scale_255;
    bmcv_convert_to_attr convert_to_attr;
    convert_to_attr.alpha_0 = input_scale;
    convert_to_attr.beta_0 = 0;
    convert_to_attr.alpha_1 = input_scale;
    convert_to_attr.beta_1 = 0;
    convert_to_attr.alpha_2 = input_scale;
    convert_to_attr.beta_2 = 0;
    auto action_preproc_t0 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < 1000; k++){

    for (int i = 0; i < img_num; i++){
        if (BM_SUCCESS != bmcv_image_convert_to(bm_handle, 1, convert_to_attr, &resize_bmcv[i], &out_imgs[i])){
            std::cout << "bmcv scale error !!!" << std::endl;
            return -1;
        }
    }
    }
            auto action_preproc_t1 = std::chrono::high_resolution_clock::now();
        printf("Action preproc time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(action_preproc_t1 - action_preproc_t0).count()/1000);

    bm_image_destroy_batch(resize_bmcv, batch_size);

    return 0;
}

int main()
{
    const char *bmodel_path = "/volume1/gddi-data/lgy/models/yolo/helmet/gddi_model_jit.int8.bmodel";
    std::string pic_path = "./data/pic/helmet3.jpg";

    // 1. 读取模型和相关属性
    bm_handle_t bm_handle;
    bm_dev_request (&bm_handle, 0);
    void *p_bmrt = bmrt_create(bm_handle);
    bmrt_load_bmodel(p_bmrt, bmodel_path);

    const char **net_names;
    bmrt_get_network_names(p_bmrt, &net_names);
    std::string net_name = net_names[0];
    free(net_names);

    const bm_net_info_t *net_info = bmrt_get_network_info(p_bmrt, net_name.c_str());
    if (NULL == net_info) {
        std::cout << "ERROR: get net-info failed!" << std::endl;
        return -1;
    }

    int output_num = net_info->output_num;
    std::vector<void *> output_ptrs;
    for(int i = 0; i < output_num; i++){
        bm_shape_t output_shape = net_info->stages[0].output_shapes[i];
        int output_count = bmrt_shape_count(&output_shape);
        auto output_ptr = (void*)malloc(sizeof(float) * output_count);
        memset(output_ptr, 0, sizeof(float) * output_count);
        output_ptrs.emplace_back(output_ptr);
    }

    bm_tensor_t *input_tensors = new bm_tensor_t[net_info->input_num];
    bm_tensor_t *output_tensors = new bm_tensor_t[net_info->output_num];
    for(int i = 0; i < net_info->input_num; ++i) {
        input_tensors[i].dtype = net_info->input_dtypes[i];
        input_tensors[i].shape = net_info->stages[0].input_shapes[i];
        input_tensors[i].st_mode = BM_STORE_1N;
        // bm_img device mem should be provided outside, such as from image's contiguous mem
        input_tensors[i].device_mem = bm_mem_null();
    }

    for(int i = 0; i < net_info->output_num; ++i) {
        output_tensors[i].dtype = net_info->output_dtypes[i];
        output_tensors[i].shape = net_info->stages[0].output_shapes[i];
        output_tensors[i].st_mode = BM_STORE_1N;
        
        // alloc as max size to reuse device mem, avoid to alloc and free everytime
        size_t max_size=0;
        for(int s=0; s<net_info->stage_num; s++){
            size_t out_size = bmrt_shape_count(&net_info->stages[s].output_shapes[i]);
            if(max_size<out_size){
                max_size = out_size;
            }
        }
        if(BM_FLOAT32 == net_info->output_dtypes[i]) max_size *= 4;
        auto ret =  bm_malloc_device_byte(bm_handle, &output_tensors[i].device_mem, max_size);
            assert(BM_SUCCESS == ret);
    }

    // 2. 读取数据
    cv::Mat img_mat = cv::imread(pic_path);
    std::vector<cv::Mat> img_mats;
    img_mats.emplace_back(img_mat);
    img_mats.emplace_back(img_mat);
    img_mats.emplace_back(img_mat);
    img_mats.emplace_back(img_mat);
    bm_image pre_img[4];

    if (bmnn_yolov5_preproc(bm_handle, net_info, img_mats, pre_img)){
        return -1;
    }

    int image_n = img_mats.size();

    // 3. 推理

    //分配device memory
    bm_device_mem_t input_dev_mem;
    bm_image_get_contiguous_device_mem(image_n, pre_img, &input_dev_mem);
    input_tensors[0].device_mem = input_dev_mem;

    // run inference
    if (BM_SUCCESS != bmrt_launch_tensor_ex(p_bmrt, net_name.c_str(), input_tensors, net_info->input_num,
                output_tensors, net_info->output_num, true, false)){
        
    }

    for(int i = 0; i < output_num; i++){
        int output_count = bmrt_shape_count(&output_tensors[i].shape) * sizeof(float);
        bm_memcpy_d2s_partial(bm_handle, output_ptrs[i], output_tensors[i].device_mem, output_count);
    }

    // 4. 后处理
    std::vector<ObjDetectInfos> objInfoss;
    int imgWidth = 500;
    int imgHeight = 350;
    Yolov5DetectionOutput(output_ptrs, objInfoss, 0, imgWidth, imgHeight);

    DrawRect(pic_path, "./data/pic/preds/helmet3.jpg", objInfoss[0].objInfos);

    bm_free_device(bm_handle, input_tensors[0].device_mem);

    for(int i = 0; i < output_num; i++){
        bm_free_device(bm_handle, output_tensors[i].device_mem);
        free(output_ptrs[i]);
    }

    delete input_tensors;
    delete output_tensors;

    bmrt_destroy(p_bmrt);
    bm_dev_free(bm_handle);

    return 0;
}