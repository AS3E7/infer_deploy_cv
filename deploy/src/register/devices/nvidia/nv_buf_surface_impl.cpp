#include "nv_buf_surface_impl.h"

#include <cstdint>
#include <cstdlib>  // for malloc/free
#include <cstring>  // for memset
#include <string>
#include <iostream>

#include "common/logger.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"

#include "cuda_runtime.h"

using namespace gddeploy;

int MemAllocatorNv::Create(BufSurfaceCreateParams *params) {
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

int MemAllocatorNv::Destroy() {
    created_ = false;
    return 0;
}

int MemAllocatorNv::Alloc(BufSurface *surf) {
    memset(surf, 0, sizeof(BufSurface));
    surf->mem_type = create_params_.mem_type;
    surf->opaque = nullptr;  // will be filled by MemPool
    surf->batch_size = create_params_.batch_size;
    surf->device_id = create_params_.device_id;
    surf->is_contiguous = 1;
    surf->surface_list = reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * surf->batch_size));
    memset(surf->surface_list, 0, sizeof(BufSurfaceParams) * surf->batch_size);

    void *data_ptr = nullptr;
    if (cudaMalloc(&data_ptr, block_size_ * create_params_.batch_size)){
        GDDEPLOY_ERROR("cudaMalloc fail, size:{}", block_size_ * create_params_.batch_size);
        return -1;
    }

    for (uint32_t i = 0; i < surf->batch_size; i++) {
        BufSurfacePlaneParams plane_params = plane_params_;
        
        for (uint32_t j = 0; j < plane_params.num_planes; j++) {
            plane_params.data_ptr[j] = (void *)((char *)data_ptr + block_size_*i + plane_params_.offset[j]);
        }
        surf->surface_list[i].color_format = create_params_.color_format;
        surf->surface_list[i].data_ptr = (void *)((char *)data_ptr + block_size_*i);
        surf->surface_list[i].width = create_params_.width;
        surf->surface_list[i].height = create_params_.height;
        surf->surface_list[i].pitch = plane_params_.pitch[0];
        surf->surface_list[i].data_size = block_size_;
        surf->surface_list[i].plane_params = plane_params;
    }
    return 0;
}

int MemAllocatorNv::Free(BufSurface *surf) {
    if (surf->is_contiguous){
        cudaFree(surf->surface_list[0].data_ptr);
        surf->surface_list[0].data_ptr = nullptr;
    } else {
        for (uint32_t i = 0; i < surf->batch_size; i++) {
            cudaFree(surf->surface_list[i].data_ptr);
            surf->surface_list[i].data_ptr = nullptr;
        }
    }

    if (surf->surface_list[0].mapped_data_ptr){
        ::free(surf->surface_list[0].mapped_data_ptr);
    }
    ::free(reinterpret_cast<void *>(surf->surface_list));
    return 0;
}

int MemAllocatorNv::Copy(BufSurface *src_surf, BufSurface *dst_surf)
{
    if (src_surf->batch_size != dst_surf->batch_size){
        GDDEPLOY_ERROR("[MemAllocatorNv] src batch size is not equal dst.");
        return -1;
    }

    if (src_surf->mem_type == GDDEPLOY_BUF_MEM_NVIDIA && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            void *src = src_surf->surface_list[0].data_ptr;
            void *dst = (void *)dst_surf->surface_list[0].data_ptr;
            int size = src_surf->batch_size * src_surf->surface_list[0].data_size;

            cudaMemcpy(src, dst, size, cudaMemcpyDeviceToHost);
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;

                if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
                    for (uint32_t j = 0; j < plane_param.num_planes; j++){
                        void *src = src_surf->surface_list[i].plane_params.data_ptr[j];
                        void *dst = (void *)((char *)dst_surf->surface_list[i].data_ptr + src_surf->surface_list[i].plane_params.offset[j]);
                        int size = src_surf->surface_list[i].plane_params.psize[j];

                        cudaMemcpy(src, dst, size, cudaMemcpyDeviceToHost);
                    }

                } else {
                    void *src = src_surf->surface_list[i].data_ptr;
                    void *dst = dst_surf->surface_list[i].data_ptr;
                    int size = src_surf->surface_list[i].data_size;

                    cudaMemcpy(src, dst, size, cudaMemcpyDeviceToHost);
                }
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_NVIDIA){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            void *src = src_surf->surface_list[0].data_ptr;
            void *dst = dst_surf->surface_list[0].data_ptr;
            int size = src_surf->batch_size * src_surf->surface_list[0].data_size;

            cudaMemcpy(src, dst, size, cudaMemcpyHostToDevice);
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;

                if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
                    for (uint32_t j = 0; j < plane_param.num_planes; j++){
                        void *src = src_surf->surface_list[i].data_ptr + dst_surf->surface_list[i].plane_params.psize[j];
                        void *dst = dst_surf->surface_list[i].plane_params.data_ptr[j];
                        int size = src_surf->surface_list[i].plane_params.psize[j];

                        cudaMemcpy(src, dst, size, cudaMemcpyHostToDevice);
                    }
                } else {
                    void *src = src_surf->surface_list[i].data_ptr;
                    void *dst = dst_surf->surface_list[i].data_ptr;
                    int size = src_surf->surface_list[i].data_size;

                    cudaMemcpy(src, dst, size, cudaMemcpyHostToDevice);
                }
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_NVIDIA && dst_surf->mem_type == GDDEPLOY_BUF_MEM_NVIDIA){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            void *src = src_surf->surface_list[0].data_ptr;
            void *dst = dst_surf->surface_list[0].data_ptr;
            int size = src_surf->batch_size * src_surf->surface_list[0].data_size;

            cudaMemcpy(src, dst, size, cudaMemcpyDeviceToDevice);
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;

                if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
                    for (uint32_t j = 0; j < plane_param.num_planes; j++){
                        void *src = src_surf->surface_list[i].plane_params.data_ptr[j];
                        void *dst = dst_surf->surface_list[i].plane_params.data_ptr[j];
                        int size = src_surf->surface_list[i].plane_params.psize[j];

                        cudaMemcpy(src, dst, size, cudaMemcpyDeviceToDevice);
                    }
                } else {
                    void *src = src_surf->surface_list[i].data_ptr;
                    void *dst = dst_surf->surface_list[i].data_ptr;
                    int size = src_surf->surface_list[i].data_size;

                    cudaMemcpy(src, dst, size, cudaMemcpyDeviceToDevice);
                }
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
        for (uint32_t i = 0; i < src_surf->batch_size; i++) {
            memcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size);
        }
    }
    return 0;
}

int MemAllocatorNv::Memset(BufSurface *surf, int value)
{
    for (uint32_t i = 0; i < surf->batch_size; i++) {
        cudaMemset(surf->surface_list[i].data_ptr, value, surf->surface_list[i].data_size);
    }
    return 0;
}