#include "cambricon_buf_surface_impl.h"
#if 1
#include <cstdint>
#include <cstdlib>  // for malloc/free
#include <cstring>  // for memset
#include <string>
#include <iostream>

#include "common/logger.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"

#include "cnrt.h"

using namespace gddeploy;

int MemAllocatorCambricon::Create(BufSurfaceCreateParams *params) {
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

    if (block_size_) {
        if (create_params_.color_format == GDDEPLOY_BUF_COLOR_FORMAT_INVALID) {
            create_params_.color_format = GDDEPLOY_BUF_COLOR_FORMAT_GRAY8;
        }
        block_size_ = (block_size_ + alignment - 1) / alignment * alignment;
        memset(&plane_params_, 0, sizeof(plane_params_));
    } else {
        GetColorFormatInfo(params->color_format, params->width, params->height, alignment, alignment, &plane_params_);
        for (uint32_t i = 0; i < plane_params_.num_planes; i++) {
            block_size_ += plane_params_.psize[i];
        }
    }

    created_ = true;
    return 0;
}

int MemAllocatorCambricon::Destroy() {
    created_ = false;
    
    return 0;
}

int MemAllocatorCambricon::Alloc(BufSurface *surf) {
    void *addr = nullptr;
    cnrtRet_t error_code = cnrtMalloc(&addr, block_size_ * create_params_.batch_size);
    if (!addr) {
        GDDEPLOY_ERROR("[MemAllocatorCambricon] Alloc(): malloc failed");
        return -1;
    }
    memset(surf, 0, sizeof(BufSurface));
    surf->mem_type = create_params_.mem_type;
    surf->opaque = nullptr;  // will be filled by MemPool
    surf->batch_size = create_params_.batch_size;
    surf->device_id = create_params_.device_id;
    surf->surface_list =
        reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * surf->batch_size));
    memset(surf->surface_list, 0, sizeof(BufSurfaceParams) * surf->batch_size);
    uint8_t *addr8 = reinterpret_cast<uint8_t *>(addr);

    for (uint32_t i = 0; i < surf->batch_size; i++) {
        surf->surface_list[i].color_format = create_params_.color_format;
        surf->surface_list[i].data_ptr = addr8;
        addr8 += block_size_;
        surf->surface_list[i].width = create_params_.width;
        surf->surface_list[i].height = create_params_.height;
        surf->surface_list[i].pitch = plane_params_.pitch[0];
        surf->surface_list[i].data_size = block_size_;
        surf->surface_list[i].plane_params = plane_params_;
    }
    return 0;
}

int MemAllocatorCambricon::Free(BufSurface *surf) {
    cnrtFree(surf->surface_list[0].data_ptr);

    if (surf->surface_list[0].mapped_data_ptr){
        ::free(surf->surface_list[0].mapped_data_ptr);
    }
    ::free(reinterpret_cast<void *>(surf->surface_list));
    return 0;
}

int MemAllocatorCambricon::Copy(BufSurface *src_surf, BufSurface *dst_surf)
{
    if (src_surf->batch_size != dst_surf->batch_size){
        GDDEPLOY_ERROR("[MemAllocatorCambricon] src batch size is not equal dst.");
        return -1;
    }

    if (src_surf->mem_type == GDDEPLOY_BUF_MEM_CAMBRICON && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            void *src = (void *)src_surf->surface_list[0].data_ptr;
            void *dst = (void *)dst_surf->surface_list[0].data_ptr;

            cnrtMemcpy(dst, src, dst_surf->surface_list[0].data_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                void *src = (void *)src_surf->surface_list[i].data_ptr;
                void *dst = (void *)dst_surf->surface_list[i].data_ptr;

                if (src_surf->surface_list[0].data_size !=  dst_surf->surface_list[0].data_size){
                    auto src_param = src_surf->surface_list[0];
                    uint32_t size = src_param.data_size;

                    cnrtMemcpy(dst_surf->surface_list[0].data_ptr, src_surf->surface_list[0].data_ptr, size, CNRT_MEM_TRANS_DIR_DEV2HOST);
                }else {
                    cnrtMemcpy(dst, src, dst_surf->surface_list[0].data_size, CNRT_MEM_TRANS_DIR_DEV2HOST);
                }
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_CAMBRICON){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            void *src = (void *)src_surf->surface_list[0].data_ptr;
            void *dst = (void *)dst_surf->surface_list[0].data_ptr;

            cnrtMemcpy(dst, src, dst_surf->surface_list[0].data_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                void *src = (void *)src_surf->surface_list[i].data_ptr;
                void *dst = (void *)dst_surf->surface_list[i].data_ptr;

                if (src_surf->surface_list[0].data_size != dst_surf->surface_list[0].data_size){
                    auto src_param = src_surf->surface_list[0];
                    uint32_t size = src_param.data_size;

                    cnrtMemcpy(dst, src, size, CNRT_MEM_TRANS_DIR_HOST2DEV);
                }else {
                    cnrtMemcpy(dst, src, dst_surf->surface_list[0].data_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
                }
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_CAMBRICON && dst_surf->mem_type == GDDEPLOY_BUF_MEM_CAMBRICON){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            void *src = (void *)src_surf->surface_list[0].data_ptr;
            void *dst = (void *)dst_surf->surface_list[0].data_ptr;

            cnrtMemcpy(dst, src, dst_surf->surface_list[0].data_size, CNRT_MEM_TRANS_DIR_HOST2DEV);
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                void *src = (void *)src_surf->surface_list[i].data_ptr;
                void *dst = (void *)dst_surf->surface_list[i].data_ptr;

                cnrtMemcpy(dst, src, dst_surf->surface_list[0].data_size, CNRT_MEM_TRANS_DIR_DEV2DEV);
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
        for (uint32_t i = 0; i < src_surf->batch_size; i++) {
            memcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size);
        }
    }
    return 0;
}

int MemAllocatorCambricon::Memset(BufSurface *surf, int value)
{
    for (uint32_t i = 0; i < surf->batch_size; i++) {
        cnrtMemset(surf->surface_list[i].data_ptr, value, surf->surface_list[i].data_size);
    }
    return 0;
}
#endif