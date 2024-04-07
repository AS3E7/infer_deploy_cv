#include <string>
#include <memory.h>
#include <vector>

#include "bmnn_predictor.h"
#include "core/mem/buf_surface.h"
#include "core/processor.h"
#include "core/mem/buf_surface_util.h"
#include "common/logger.h"

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"
#include "bmcv_api_ext.h"
// #include "bm_wrapper.hpp"

using namespace gddeploy;
// 通过ort_model的priv_获取到session
// 拷贝输入PackPtr里面的数据到session  Bmnn::Value input_tensor = Bmnn::Value::CreateTensor
// session推理，得到的结果拷贝到输出的output PackPtr

namespace gddeploy
{
class BmnnPredictorPrivate{
public:
    BmnnPredictorPrivate() = default;
    BmnnPredictorPrivate(ModelPtr model):model_(model){
        p_bmrt_ = gddeploy::any_cast<std::shared_ptr<void>>(model->GetModel());

        const char **net_names;
        bmrt_get_network_names(p_bmrt_.get(), &net_names);
        std::string net_name(net_names[0]);
        free(net_names);

        net_info_ = bmrt_get_network_info(p_bmrt_.get(), net_name.c_str());
        if (NULL == net_info_) {
            std::cout << "ERROR: get net-info failed!" << std::endl;
        }

        bm_handle_ = (bm_handle_t)bmrt_get_bm_handle(p_bmrt_.get());
        // bm_dev_request(&bm_handle_, 0);
    }
    ~BmnnPredictorPrivate(){
        delete [] input_tensors_;
        delete [] output_tensors_;
        // bm_dev_free(bm_handle_);
        for (auto pool : pools_){
            // pool->DestroyPool();
            delete pool;
        }
        pools_.clear();
    }

    BufSurfWrapperPtr RequestBuffer(int idx){
        BufSurfWrapperPtr buf = pools_[idx]->GetBufSurfaceWrapper();

        return buf;
    }

    int Init(std::string config);

    bm_handle_t GetBmHandle() { return bm_handle_; }

    size_t num_inputs_;
    size_t num_outputs_;
    std::vector<const char *> input_node_names_;
    std::vector<const char *> output_node_names_;

    ModelPtr model_;

    const bm_net_info_t *net_info_;

    bm_tensor_t *input_tensors_;
    bm_tensor_t *output_tensors_;
    bm_handle_t bm_handle_;
    std::shared_ptr<void> p_bmrt_;

private:
    std::vector<BufPool*> pools_;
};
}

int BmnnPredictorPrivate::Init(std::string config)
{
    // int output_num = net_info_->output_num;
    // std::vector<void *> output_ptrs;
    // for(int i = 0; i < output_num; i++){
    //     bm_shape_t output_shape = net_info_->stages[0].output_shapes[i];
    //     int output_count = bmrt_shape_count(&output_shape);
    //     auto output_ptr = (void*)malloc(sizeof(float) * output_count);
    //     memset(output_ptr, 0, sizeof(float) * output_count);
    //     output_ptrs.emplace_back(output_ptr);
    // }

    input_tensors_ = new bm_tensor_t[net_info_->input_num];
    output_tensors_ = new bm_tensor_t[net_info_->output_num];
    for(int i = 0; i < net_info_->input_num; ++i) {
        (input_tensors_+i)->dtype = net_info_->input_dtypes[i];
        (input_tensors_+i)->shape = net_info_->stages[0].input_shapes[i];
        (input_tensors_+i)->st_mode = BM_STORE_1N;
        // bm_img device mem should be provided outside, such as from image's contiguous mem
        size_t in_size = bmrt_shape_count(&net_info_->stages[0].input_shapes[i]);
        if(BM_FLOAT32 == net_info_->input_dtypes[i]) in_size *= 4;
        if (BM_SUCCESS != bm_malloc_device_byte(bm_handle_, &(input_tensors_+i)->device_mem, in_size)){
            GDDEPLOY_ERROR("[register] [bmnn predictor] input bm_malloc_device_byte error !!!");
        }
    }

    for(int i = 0; i < net_info_->output_num; ++i) {
        (output_tensors_+i)->dtype = net_info_->output_dtypes[i];
        (output_tensors_+i)->shape = net_info_->stages[0].output_shapes[i];
        (output_tensors_+i)->st_mode = BM_STORE_1N;
        
        // alloc as max size to reuse device mem, avoid to alloc and free everytime
        size_t max_size=0;
        for(int s=0; s<net_info_->stage_num; s++){
            size_t out_size = bmrt_shape_count(&net_info_->stages[s].output_shapes[i]);
            if(max_size<out_size){
                max_size = out_size;
            }
        }
        if(BM_FLOAT32 == net_info_->output_dtypes[i]) max_size *= 4;
        if (BM_SUCCESS != bm_malloc_device_byte(bm_handle_, &(output_tensors_+i)->device_mem, max_size)){
            GDDEPLOY_ERROR("[register] [bmnn predictor] output bm_malloc_device_byte error !!!");
        }
    }


    size_t o_num = model_->OutputNum();
    for (size_t i_idx = 0; i_idx < o_num; ++i_idx) {
        const DataLayout input_layout =  model_->OutputLayout(i_idx);
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

        int model_h = 1, model_w = 1, model_c = 1, model_b = 1;
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
        } else if (shape.Size() == 2){
            model_b = shape[0];
            model_h = 1;
            model_w = 1;
            model_c = shape[1];
        } else if (shape.Size() == 3){
            model_b = shape[0];
            model_h = 1;
            model_w = shape[2];
            model_c = shape[1];
        }

        BufSurfaceCreateParams create_params;
        memset(&create_params, 0, sizeof(create_params));
        create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
        create_params.force_align_1 = 1;  // to meet mm's requirement
        create_params.device_id = 0;
        create_params.batch_size = model_b;
        create_params.size = model_h * model_w * model_c;
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

Status BmnnPredictor::Init(PredictorConfigPtr config, ModelPtr model) noexcept 
{
    priv_ = std::make_shared<BmnnPredictorPrivate>(model);
    priv_->Init("");
    model_ = model;

    return gddeploy::Status::SUCCESS; 
}


Status BmnnPredictor::Process(PackagePtr pack) noexcept
{
    BufSurfWrapperPtr in_buf = pack->predict_io->GetLref<BufSurfWrapperPtr>();
    BufSurface *surf = in_buf->GetBufSurface();
    int batch_size = surf->batch_size;  

    bm_image imgs[batch_size];
#if 0
    auto input_shape = *priv_->net_info_->stages[0].input_shapes;

    if (surf->mem_type == GDDEPLOY_BUF_MEM_BMNN){
        for (int i = 0; i < batch_size; i++){
            imgs[i] = *((bm_image *)surf->surface_list[i].data_ptr);
        }
    } else {    // 一般是CPU前处理，需要临时分配dev mem存放前处理后的数据
        auto src_param = in_buf->GetSurfaceParams(0);
        uint32_t data_size = src_param->data_size;

        bm_image_alloc_contiguous_mem(surf->batch_size, imgs);
        
        bm_device_mem_t pmem = bm_mem_null();
        bm_image_get_contiguous_device_mem(batch_size, imgs, &pmem);
        // if (bm_malloc_device_byte(priv_->GetBmHandle(), &pmem, batch_size*data_size)){
        //     std::cout << "Malloc device memory fail" << std::endl;
        // }

        auto ret = bm_memcpy_s2d_partial(priv_->GetBmHandle(), pmem, (void *)in_buf->GetData(0, 0), data_size);
        if (ret == -1){
            printf("copy s2d error\n");
        }

        input_shape.dims[0] = batch_size;
    } 

    std::vector<void *> outputs;
    std::vector<BufSurfWrapperPtr> out_bufs;
    // 1. 后处理，画图
    for (unsigned int i = 0; i < priv_->net_info_->output_num; ++i){
        // 请求申请一块CPU内存
        BufSurfWrapperPtr buf = priv_->RequestBuffer(i);
    
        outputs.emplace_back(buf->GetData(0, 0));

        out_bufs.emplace_back(buf);
    }

    auto ret = bm_inference(priv_->p_bmrt_.get(), imgs, outputs, input_shape, priv_->net_info_->name);
#else
    auto src_param = in_buf->GetSurfaceParams();

    int ret = -1;
    // bm_memset_device(priv_->GetBmHandle(), 0, priv_->input_tensors_->device_mem);
    // for(int i = 0; i < priv_->net_info_->output_num; ++i) {
    //     bm_memset_device(priv_->GetBmHandle(), 0, priv_->output_tensors_[i].device_mem);
    // }
    // 1. 拷贝数据
    if (surf->mem_type == GDDEPLOY_BUF_MEM_BMNN){
        int img_size = src_param->data_size;

        for (int i = 0; i < batch_size; i++){
            imgs[i] = *((bm_image *)surf->surface_list[i].data_ptr);
            bm_device_mem_t pmem[3];
            if(BM_SUCCESS != bm_image_get_device_mem(imgs[i], pmem)){
                GDDEPLOY_ERROR("[MemAllocatorBmnn] bm_image_get_device_mem fail");
            }

            int plane_size = img_size/3;
            // for (int j = 0; j < 3; j++){
                // ret = bm_memcpy_d2d(priv_->GetBmHandle(), priv_->input_tensors_->device_mem, i*img_size+j*plane_size, pmem[0], j*plane_size, plane_size);
                ret = bm_memcpy_d2d_byte(priv_->GetBmHandle(), priv_->input_tensors_->device_mem, i*img_size, pmem[0], 0, img_size);
                // ret = bm_memcpy_c2c(priv_->GetBmHandle(), priv_->GetBmHandle(), pmem[0], priv_->input_tensors_->device_mem, false);
                if (ret == -1){
                    GDDEPLOY_ERROR("[register] [bmnn predictor] copy d2d error !!!");
                }
            // }
        }

    } else {
        uint32_t img_size = src_param->data_size;
        // if (priv_->input_tensors_->device_mem) {
            
            bm_device_mem_t pmem = bm_mem_null();
            if (bm_malloc_device_byte(priv_->GetBmHandle(), &pmem, 4*img_size)){
                std::cout << "Malloc device memory fail" << std::endl;
            }
            priv_->input_tensors_->device_mem = pmem;
        // }
        ret = bm_memcpy_s2d_partial(priv_->GetBmHandle(), priv_->input_tensors_->device_mem, (void *)in_buf->GetData(0, 0), img_size);
        if (ret == -1){
            GDDEPLOY_ERROR("[register] [bmnn predictor] copy s2d error !!!");
        }
        
    }
    
    // 2. 推理
    // auto t0 = std::chrono::high_resolution_clock::now();
    if (BM_SUCCESS != (ret = bmrt_launch_tensor_ex(priv_->p_bmrt_.get(), priv_->net_info_->name, 
                priv_->input_tensors_, priv_->net_info_->input_num,
                priv_->output_tensors_, priv_->net_info_->output_num, true, true))){
        // printf("bmrt_launch_tensor_ex error\n");
    }
    if (false == ret){
        GDDEPLOY_ERROR("[register] [bmnn predictor] bm_inference error !!!");
        return gddeploy::Status::ERROR_BACKEND;
    }
    // sync, wait for finishing inference
    bm_thread_sync(priv_->bm_handle_);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("inference time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    std::vector<BufSurfWrapperPtr> out_bufs;
    // 1. 后处理，画图
    for (unsigned int i = 0; i < priv_->net_info_->output_num; ++i){
        // 请求申请一块CPU内存
        BufSurfWrapperPtr buf = priv_->RequestBuffer(i);
    
        auto output_shape = priv_->model_->OutputShape(i);
        auto data_count  = output_shape.BatchDataCount();

        int data_size = (priv_->output_tensors_[i].dtype == BM_FLOAT32) ? sizeof(float) : sizeof(int8_t);
        memset(buf->GetData(0, 0), 1, data_count * data_size);
        // 拷贝到CPU内存
        if (BM_SUCCESS != bm_memcpy_d2s_partial(priv_->bm_handle_, (void *)buf->GetData(0, 0), priv_->output_tensors_[i].device_mem, data_count * data_size)){
        // if (BM_SUCCESS != bm_memcpy_d2s(priv_->bm_handle_, (void *)buf->GetData(0, 0), priv_->output_tensors_[i].device_mem)){
            GDDEPLOY_ERROR("[register] [bmnn predictor] copy d2s error !!!");
        }

        out_bufs.emplace_back(buf);
    }

#endif
    

    #if 0
    int ret = -1;
    // 1. 拷贝数据
    if (surf->mem_type == GDDEPLOY_BUF_MEM_BMNN){
        // bm_device_mem_t *device_mem = (bm_device_mem_t *)in_buf->GetData(0, 0);

        bm_device_mem_t device_mem;
        // bm_set_device_mem(&device_mem, surf->surface_list[0].data_size * surf->batch_size, (unsigned long long)in_buf->GetData(0, 0));
        std::vector<bm_image> imgs;
        imgs.resize(batch_size);
        for (int i = 0; i < batch_size; i++){
            imgs[i] = *((bm_image *)surf->surface_list[i].data_ptr);
        }
        bm_image_get_contiguous_device_mem(batch_size, imgs.data(), &device_mem);

        priv_->input_tensors_->device_mem = device_mem;
    } else {    // 一般是CPU前处理，需要临时分配dev mem存放前处理后的数据
        auto src_param = in_buf->GetSurfaceParams(0);
        uint32_t data_size = src_param->data_size;
            
        bm_device_mem_t pmem = bm_mem_null();
        if (bm_malloc_device_byte(priv_->GetBmHandle(), &pmem, batch_size*data_size)){
            std::cout << "Malloc device memory fail" << std::endl;
        }
        priv_->input_tensors_->device_mem = pmem;

        ret = bm_memcpy_s2d_partial(priv_->GetBmHandle(), priv_->input_tensors_->device_mem, (void *)in_buf->GetData(0, 0), data_size);
        if (ret == -1){
            printf("copy s2d error\n");
        }
        
    }
    
    // 2. 推理
    if (BM_SUCCESS != (ret = bmrt_launch_tensor_ex(priv_->p_bmrt_.get(), priv_->net_info_->name, 
                priv_->input_tensors_, priv_->net_info_->input_num,
                priv_->output_tensors_, priv_->net_info_->output_num, true, false))){
        // printf("bmrt_launch_tensor_ex error\n");
    }
    #endif

    pack->predict_io->Set(std::move(out_bufs));

    // if (surf->mem_type != GDDEPLOY_BUF_MEM_BMNN){
        // bm_free_device(priv_->GetBmHandle(), priv_->input_tensors_->device_mem);
        // priv_->input_tensors_->device_mem = bm_mem_null();
    // }
    // auto t1 = std::chrono::high_resolution_clock::now();
    // printf("inference time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());


    return gddeploy::Status::SUCCESS; 
}


// REGISTER_PREDICTOR_CREATOR("ort", "cpu", BmnnPredictorCreator)