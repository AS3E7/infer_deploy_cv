#include "ascend_buf_surface_impl.h"

#include <cstdint>
#include <cstdlib>  // for malloc/free
#include <cstring>  // for memset
#include <string>
#include <iostream>

#include "common/logger.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"

#include "acl/acl.h"
#define ENABLE_DVPP_INTERFACE 1
#include "acl/ops/acl_dvpp.h"

using namespace gddeploy;

MemAllocatorAscend::MemAllocatorAscend()
{
}

MemAllocatorAscend::~MemAllocatorAscend()
{
}

int MemAllocatorAscend::Create(BufSurfaceCreateParams *params) {
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

int MemAllocatorAscend::Destroy() {
    created_ = false;
    
    return 0;
}


int MemAllocatorAscend::Alloc(BufSurface *surf) {
    memset(surf, 0, sizeof(BufSurface));
    surf->mem_type = create_params_.mem_type;
    surf->opaque = nullptr;  // will be filled by MemPool
    surf->batch_size = create_params_.batch_size;
    surf->device_id = create_params_.device_id;
    surf->is_contiguous = 1;
    surf->surface_list = reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * surf->batch_size));
    memset(surf->surface_list, 0, sizeof(BufSurfaceParams) * surf->batch_size);
    
    // 根据设备内存类型，分配内存，比如dvpp和rt设备内存
    if (create_params_.mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP) {
        for (int i = 0; i < surf->batch_size; i++) {
            surf->surface_list[i].data_ptr = malloc(block_size_);

            auto ret = acldvppMalloc((void **)(&(surf->surface_list[i].data_ptr)), block_size_);
            if (ret != 0) {
                GDDEPLOY_ERROR("[MemAllocatorAscend] malloc failed, size:{}", block_size_);
                return -1;
            }
            surf->surface_list[i].data_size = block_size_;
        }
    } else if (create_params_.mem_type == GDDEPLOY_BUF_MEM_ASCEND_RT) {
        for (int i = 0; i < surf->batch_size; i++) {
            auto ret = aclrtMalloc(&(surf->surface_list[i].data_ptr), block_size_, ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != 0) {
                GDDEPLOY_ERROR("[MemAllocatorAscend] malloc failed, size:{}", block_size_);
                return -1;
            }
            surf->surface_list[i].data_size = block_size_;
        }
    } else {
        GDDEPLOY_ERROR("[MemAllocatorAscend] invalid mem type:{}", create_params_.mem_type);
        return -1;
    }

    BufSurfacePlaneParams plane_params = plane_params_;

    for (int i = 0; i < surf->batch_size; i++) {       
        surf->surface_list[i].color_format = create_params_.color_format;
        surf->surface_list[i].data_ptr = (void *)((char *)surf->surface_list[i].data_ptr + i * block_size_);
        surf->surface_list[i].width = create_params_.width;
        surf->surface_list[i].height = create_params_.height;
        surf->surface_list[i].pitch = plane_params_.pitch[0];
        surf->surface_list[i].data_size = block_size_;
        surf->surface_list[i].plane_params = plane_params;
    }
    return 0;
}

int MemAllocatorAscend::Free(BufSurface *surf) {
    if (surf->surface_list[0].data_ptr == nullptr) {
        aclrtFree(surf->surface_list[0].data_ptr);
        surf->surface_list[0].data_ptr = nullptr;
        return 0;
    }

    if (surf->surface_list[0].mapped_data_ptr){
        ::free(surf->surface_list[0].mapped_data_ptr);
    }
    ::free(reinterpret_cast<void *>(surf->surface_list));
    return 0;
}

int MemAllocatorAscend::Copy(BufSurface *src_surf, BufSurface *dst_surf)
{
    if (src_surf->batch_size != dst_surf->batch_size){
        GDDEPLOY_ERROR("[MemAllocatorAscend] src batch size is not equal dst.");
        return -1;
    }
    int ret = 0;

    if (src_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM
        || src_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_RT && dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            aclrtMemcpy(dst_surf->surface_list[0].data_ptr, dst_surf->surface_list[0].data_size, src_surf->surface_list[0].data_ptr, src_surf->surface_list[0].data_size, ACL_MEMCPY_DEVICE_TO_HOST);
            if ( 0 != ret){
                GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
            }
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;

                if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
                    for (uint32_t j = 0; j < plane_param.num_planes; j++){
                        aclrtMemcpy(dst_surf->surface_list[i].data_ptr, dst_surf->surface_list[i].data_size, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size, ACL_MEMCPY_DEVICE_TO_HOST);
                        if ( 0 != ret){
                            GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
                        }
                    }

                } else {
                    aclrtMemcpy(dst_surf->surface_list[i].data_ptr, dst_surf->surface_list[i].data_size, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size, ACL_MEMCPY_DEVICE_TO_HOST);
                    if ( 0 != ret){
                        GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
                    }
                }
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP 
        ||src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM && dst_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_RT){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            aclrtMemcpy(dst_surf->surface_list[0].data_ptr, dst_surf->surface_list[0].data_size, src_surf->surface_list[0].data_ptr, src_surf->surface_list[0].data_size, ACL_MEMCPY_HOST_TO_DEVICE);
            if ( 0 != ret){
                GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
            }
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;
                aclrtMemcpy(dst_surf->surface_list[i].data_ptr, dst_surf->surface_list[i].data_size, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size, ACL_MEMCPY_HOST_TO_DEVICE);
                if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
                    for (uint32_t j = 0; j < plane_param.num_planes; j++){
                        aclrtMemcpy(dst_surf->surface_list[i].data_ptr, dst_surf->surface_list[i].data_size, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size, ACL_MEMCPY_HOST_TO_DEVICE);
                        if ( 0 != ret){
                            GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
                        }
                    }
                } else {
                    aclrtMemcpy(dst_surf->surface_list[i].data_ptr, dst_surf->surface_list[i].data_size, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size, ACL_MEMCPY_HOST_TO_DEVICE);
                    if ( 0 != ret){
                        GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
                    }
                }
            }
        }
    }else if (src_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP && dst_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP
            || src_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_DVPP && dst_surf->mem_type == GDDEPLOY_BUF_MEM_ASCEND_RT){
        if (src_surf->is_contiguous && dst_surf->is_contiguous){
            aclrtMemcpy(dst_surf->surface_list[0].data_ptr, dst_surf->surface_list[0].data_size, src_surf->surface_list[0].data_ptr, src_surf->surface_list[0].data_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
            if ( 0 != ret){
                GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
            }
        }else{
            for (uint32_t i = 0; i < src_surf->batch_size; i++) {
                BufSurfacePlaneParams plane_param = src_surf->surface_list[i].plane_params;
                
                if (plane_param.data_ptr[0] != nullptr){    // 说明连plane都不连续
                    for (uint32_t j = 0; j < plane_param.num_planes; j++){
                        aclrtMemcpy(dst_surf->surface_list[i].data_ptr, dst_surf->surface_list[i].data_size, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
                        if ( 0 != ret){
                            GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
                        }
                    }
                } else {
                    aclrtMemcpy(dst_surf->surface_list[i].data_ptr, dst_surf->surface_list[i].data_size, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size, ACL_MEMCPY_DEVICE_TO_DEVICE);
                    if ( 0 != ret){
                        GDDEPLOY_ERROR("[register] [ascend] aclrtMemcpy error , ret:{}!!!", ret);
                    }
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

int MemAllocatorAscend::Memset(BufSurface *surf, int value)
{
    for (uint32_t i = 0; i < surf->batch_size; i++) {
    }
    return 0;
}