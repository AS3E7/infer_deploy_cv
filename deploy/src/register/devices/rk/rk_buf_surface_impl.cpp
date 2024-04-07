#include "rk_buf_surface_impl.h"

#include <cstdint>
#include <cstdlib>  // for malloc/free
#include <cstring>  // for memset
#include <string>
#include <iostream>

#include "common/logger.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"



using namespace gddeploy;

MemAllocatorRk::MemAllocatorRk()
{
}

MemAllocatorRk::~MemAllocatorRk()
{
}

int MemAllocatorRk::Create(BufSurfaceCreateParams *params) {
    create_params_ = *params;
    uint32_t alignment = 4;
    if (create_params_.batch_size == 0) {
        create_params_.batch_size = 1;
    }
    if (params->force_align_1) {
        alignment = 1;
    }

    memset(&plane_params_, 0, sizeof(BufSurfacePlaneParams));
    block_size_ = params->size;

    GetColorFormatInfo(params->color_format, params->width, params->height, alignment, alignment, &plane_params_);
    if (block_size_) {
        if (create_params_.color_format == GDDEPLOY_BUF_COLOR_FORMAT_INVALID) {
            create_params_.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;
        }
        block_size_ = (block_size_ + alignment - 1) / alignment * alignment;
    } else {
        for (uint32_t i = 0; i < plane_params_.num_planes; i++) {
            block_size_ += plane_params_.psize[i];
        }
    }


    created_ = true;
    return 0;
}

int MemAllocatorRk::Destroy() {
    created_ = false;
    
    return 0;
}


int MemAllocatorRk::Alloc(BufSurface *surf) {
    memset(surf, 0, sizeof(BufSurface));
    surf->mem_type = create_params_.mem_type;
    surf->opaque = nullptr;  // will be filled by MemPool
    surf->batch_size = create_params_.batch_size;
    surf->device_id = create_params_.device_id;
    surf->is_contiguous = 1;
    surf->surface_list = reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * surf->batch_size));
    memset(surf->surface_list, 0, sizeof(BufSurfaceParams) * surf->batch_size);
    
    rknn_tensor_mem* input_mems = new rknn_tensor_mem[surf->batch_size];
    char *buf = (char *)malloc(block_size_ * surf->batch_size);

    for (int i = 0; i < surf->batch_size; i++) {        
        surf->surface_list[i].color_format = create_params_.color_format;
        surf->surface_list[i].data_ptr = (void *)(buf + i * block_size_);
        surf->surface_list[i].width = create_params_.width;
        surf->surface_list[i].height = create_params_.height;
        surf->surface_list[i].pitch = plane_params_.pitch[0];
        surf->surface_list[i].data_size = block_size_;
        // surf->surface_list[i].plane_params = plane_params;
    }
    return 0;
}

int MemAllocatorRk::Free(BufSurface *surf) {

    free(surf->surface_list[0].data_ptr);
    surf->surface_list[0].data_ptr = nullptr;

    if (surf->surface_list[0].mapped_data_ptr){
        ::free(surf->surface_list[0].mapped_data_ptr);
    }
    ::free(reinterpret_cast<void *>(surf->surface_list));
    return 0;
}

int MemAllocatorRk::Copy(BufSurface *src_surf, BufSurface *dst_surf)
{
    if (src_surf->batch_size != dst_surf->batch_size){
        GDDEPLOY_ERROR("[MemAllocatorRk] src batch size is not equal dst.");
        return -1;
    }

    // if (src_surf->mem_type == GDDEPLOY_BUF_MEM_RK_RGA && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
    //     if (src_surf->is_contiguous && dst_surf->is_contiguous){
    //         bm_device_mem_t src;
    //         bm_set_device_mem(&src, src_surf->surface_list[0].data_size * src_surf->batch_size, (unsigned long long)src_surf->surface_list[0].data_ptr);
            
    //         void *dst = (void *)dst_surf->surface_list[0].data_ptr;
    //         auto ret = bm_memcpy_d2s_partial(handle_, dst, src, src_surf->batch_size * src_surf->surface_list[0].data_size);
    //         if ( 0 != ret){
    //             GDDEPLOY_ERROR("[register] [rk] bm_memcpy_d2s_partial error , ret:{}!!!", ret);
    //         }
    //     }else{
    //         for (uint32_t i = 0; i < src_surf->batch_size; i++) {
    //             BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;

    //             if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
    //                 for (uint32_t j = 0; j < plane_param.num_planes; j++){
    //                     bm_device_mem_t src;
    //                     bm_set_device_mem(&src, src_surf->surface_list[i].plane_params.psize[j], (unsigned long long)src_surf->surface_list[i].plane_params.data_ptr[j]);

    //                     void *dst = (void *)dst_surf->surface_list[i].data_ptr + src_surf->surface_list[i].plane_params.offset[j];  //主存一般是连续空间分配
    //                     auto ret = bm_memcpy_d2s_partial(handle_, (void *)dst,
    //                                 src, src_surf->surface_list[i].plane_params.psize[j]);
    //                     if ( 0 != ret){
    //                         GDDEPLOY_ERROR("[register] [rk] bm_memcpy_d2s_partial error , ret:{}!!!", ret);
    //                     }
    //                 }

    //             } else {
    //                 bm_device_mem_t src;
    //                 bm_set_device_mem(&src, src_surf->surface_list[i].data_size, (unsigned long long)src_surf->surface_list[i].data_ptr);

    //                 void *dst = (void *)dst_surf->surface_list[i].data_ptr;

    //                 auto ret = bm_memcpy_d2s_partial(handle_, dst,
    //                                             src, src_surf->surface_list[i].data_size);
    //                 if ( 0 != ret){
    //                     GDDEPLOY_ERROR("[register] [rk] bm_memcpy_d2s_partial error , ret:{}!!!", ret);
    //                 }
    //             }
    //         }
    //     }
    // }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_RK_RGA){
    //     if (src_surf->is_contiguous && dst_surf->is_contiguous){
    //         bm_device_mem_t src;
    //         bm_set_device_mem(&src, src_surf->surface_list[0].data_size * src_surf->batch_size, (unsigned long long)src_surf->surface_list[0].data_ptr);

    //         void *dst = (void *)dst_surf->surface_list[0].data_ptr;

    //         auto ret = bm_memcpy_s2d(handle_, src, dst);
    //         if ( 0 != ret){
    //             GDDEPLOY_ERROR("[register] [rk] bm_memcpy_s2d error , ret:{}!!!", ret);
    //         }
    //     }else{
    //         for (uint32_t i = 0; i < src_surf->batch_size; i++) {
    //             BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;

    //             if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
    //                 for (uint32_t j = 0; j < plane_param.num_planes; j++){
    //                     void *src = (void *)src_surf->surface_list[i].data_ptr + dst_surf->surface_list[i].plane_params.psize[j];  //主存一般是连续空间分配

    //                     bm_device_mem_t dst;
    //                     bm_set_device_mem(&dst, dst_surf->surface_list[i].plane_params.psize[j], (unsigned long long)dst_surf->surface_list[i].plane_params.data_ptr[j]);

    //                     auto ret = bm_memcpy_s2d_partial(handle_, dst,
    //                                 src, dst_surf->surface_list[i].plane_params.psize[j]);
    //                     if ( 0 != ret){
    //                         GDDEPLOY_ERROR("[register] [rk] bm_memcpy_s2d_partial error , ret:{}!!!", ret);
    //                     }
    //                 }
    //             } else {
    //                 void *src = (void *)src_surf->surface_list[i].data_ptr;  //主存一般是连续空间分配

    //                 bm_device_mem_t dst;
    //                 bm_set_device_mem(&dst, dst_surf->surface_list[i].data_size, (unsigned long long)dst_surf->surface_list[i].data_ptr);

    //                 auto ret = bm_memcpy_s2d_partial(handle_, dst,
    //                                         src, dst_surf->surface_list[i].data_size);
    //                 if ( 0 != ret){
    //                     GDDEPLOY_ERROR("[register] [rk] bm_memcpy_s2d_partial error , ret:{}!!!", ret);
    //                 }
    //             }
    //         }
    //     }
    // }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_RK_RGA && dst_surf->mem_type == GDDEPLOY_BUF_MEM_RK_RGA){
    //     if (src_surf->is_contiguous && dst_surf->is_contiguous){
    //         bm_device_mem_t src;
    //         bm_set_device_mem(&src, src_surf->surface_list[0].data_size * src_surf->batch_size, (unsigned long long)src_surf->surface_list[0].data_ptr);

    //         bm_device_mem_t dst;
    //         bm_set_device_mem(&dst, dst_surf->surface_list[0].data_size * dst_surf->batch_size, (unsigned long long)dst_surf->surface_list[0].data_ptr);

    //         size_t size = src_surf->surface_list[0].data_size * src_surf->batch_size;
    //         // auto ret = bm_memcpy_d2d_byte(handle_, dst, 0, src, 0, size);
    //         auto ret = bm_memcpy_c2c(handle_, handle_, src, dst, true);
    //         if ( 0 != ret){
    //             GDDEPLOY_ERROR("[register] [rk] bm_memcpy_d2d_byte error , ret:{}!!!", ret);
    //         }
    //     }else{
    //         for (uint32_t i = 0; i < src_surf->batch_size; i++) {
    //             BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;

    //             if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
    //                 for (uint32_t j = 0; j < plane_param.num_planes; j++){

    //                     bm_device_mem_t src = bm_mem_from_device((unsigned long long)src_surf->surface_list[i].plane_params.data_ptr[j], src_surf->surface_list[i].plane_params.psize[j]);
    //                     bm_device_mem_t dst = bm_mem_from_device((unsigned long long)dst_surf->surface_list[i].plane_params.data_ptr[j], dst_surf->surface_list[i].plane_params.psize[j]);
    //                     // bm_set_device_mem(&dst, dst_surf->surface_list[i].plane_params.psize[j], (unsigned long long)dst_surf->surface_list[i].plane_params.data_ptr[j]);

    //                     // auto ret = bm_memcpy_d2d_byte(handle_, *dst, 0, src, 0, src_surf->surface_list[i].plane_params.psize[j]);
    //                     // auto ret = bm_memcpy_d2d_byte(handle_, dst, 0, src, 0, src_surf->surface_list[i].plane_params.psize[j]);
    //                     auto ret = bm_memcpy_c2c(handle_, handle_, src, dst, true);
    //                     if ( 0 != ret){
    //                         GDDEPLOY_ERROR("[register] [rk] bm_memcpy_d2d_byte error , ret:{}!!!", ret);
    //                     }
    //                 }
    //             } else {
    //                 bm_device_mem_t src;
    //                 bm_set_device_mem(&src, src_surf->surface_list[i].data_size, (unsigned long long)src_surf->surface_list[i].data_ptr);

    //                 bm_device_mem_t dst;
    //                 bm_set_device_mem(&dst, dst_surf->surface_list[i].data_size, (unsigned long long)dst_surf->surface_list[i].data_ptr);

    //                 auto ret = bm_memcpy_d2d_byte(handle_, dst, 0, src, 0, src_surf->surface_list[i].data_size);
    //                 if ( 0 != ret){
    //                     GDDEPLOY_ERROR("[register] [rk] bm_memcpy_d2d_byte error , ret:{}!!!", ret);
    //                 }
    //             }
    //         }
    //     }
    // }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
    //     for (uint32_t i = 0; i < src_surf->batch_size; i++) {
    //         memcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size);
    //     }
    // }
    return 0;
}

int MemAllocatorRk::Memset(BufSurface *surf, int value)
{
    for (uint32_t i = 0; i < surf->batch_size; i++) {
        // bm_device_mem_t *mem = (bm_device_mem_t *)surf->surface_list[i].data_ptr;
        // bm_memset_device(handle_, value, *mem);
    }
    return 0;
}