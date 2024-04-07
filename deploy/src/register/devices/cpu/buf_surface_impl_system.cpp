#include "buf_surface_impl_system.h"

#include <cstdlib> // for malloc/free
#include <cstring> // for memset
#include <string>
#include <iostream>

#include "core/mem/buf_surface_util.h"
using namespace gddeploy;

int MemAllocatorSystem::Create(BufSurfaceCreateParams *params)
{
    create_params_ = *params;
    uint32_t alignment = 4;
    if (create_params_.batch_size == 0 || create_params_.batch_size == -1)
    {
        create_params_.batch_size = 1;
    }
    if (params->force_align_1)
    {
        alignment = 1;
    }

    memset(&plane_params_, 0, sizeof(BufSurfacePlaneParams));
    block_size_ = params->size;

    if (block_size_)
    {
        if (create_params_.color_format == GDDEPLOY_BUF_COLOR_FORMAT_INVALID)
        {
            create_params_.color_format = GDDEPLOY_BUF_COLOR_FORMAT_GRAY8;
        }
        block_size_ = (block_size_ + alignment - 1) / alignment * alignment;
        memset(&plane_params_, 0, sizeof(plane_params_));
    }
    else
    {
        GetColorFormatInfo(params->color_format, params->width, params->height, alignment, alignment, &plane_params_);
        for (uint32_t i = 0; i < plane_params_.num_planes; i++)
        {
            block_size_ += plane_params_.psize[i];
        }
    }
    created_ = true;
    return 0;
}

int MemAllocatorSystem::Destroy()
{
    created_ = false;
    return 0;
}

int MemAllocatorSystem::Alloc(BufSurface *surf)
{
    void *addr = reinterpret_cast<void *>(malloc(block_size_ * create_params_.batch_size));
    if (!addr)
    {
        std::cout << "[gddeploy] [MemAllocatorSystem] Alloc(): malloc failed" << std::endl;
        return -1;
    }
    memset(addr, 0, block_size_ * create_params_.batch_size);
    memset(surf, 0, sizeof(BufSurface));
    surf->mem_type = create_params_.mem_type;
    surf->opaque = nullptr; // will be filled by MemPool
    surf->batch_size = create_params_.batch_size;
    surf->device_id = create_params_.device_id;
    surf->surface_list =
        reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * surf->batch_size));
    memset(surf->surface_list, 0, sizeof(BufSurfaceParams) * surf->batch_size);
    uint8_t *addr8 = reinterpret_cast<uint8_t *>(addr);

    for (uint32_t i = 0; i < surf->batch_size; i++)
    {
        surf->surface_list[i].color_format = create_params_.color_format;
        surf->surface_list[i].data_ptr = addr8 + i * block_size_;
        surf->surface_list[i].width = create_params_.width;
        surf->surface_list[i].height = create_params_.height;
        surf->surface_list[i].pitch = plane_params_.pitch[0];
        surf->surface_list[i].data_size = block_size_;
        surf->surface_list[i].plane_params = plane_params_;
    }
    return 0;
}

int MemAllocatorSystem::Free(BufSurface *surf)
{
    void *addr = surf->surface_list[0].data_ptr;
    ::free(addr);
    ::free(reinterpret_cast<void *>(surf->surface_list));
    return 0;
}

int MemAllocatorSystem::Copy(BufSurface *src_surf, BufSurface *dst_surf)
{
    for (uint32_t i = 0; i < src_surf->batch_size; i++)
    {
        if (src_surf->surface_list[i].plane_params.data_ptr[0] != nullptr){
            for (uint32_t j = 0; j < src_surf->surface_list[i].plane_params.num_planes; j++)
            {
                memcpy(dst_surf->surface_list[i].data_ptr + j*src_surf->surface_list[i].plane_params.offset[j], src_surf->surface_list[i].plane_params.data_ptr[j], src_surf->surface_list[i].plane_params.psize[j]);
            }
        } else {
            memcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size);
        }
    }
    return 0;
}

int MemAllocatorSystem::Memset(BufSurface *surf, int value)
{
    for (uint32_t i = 0; i < surf->batch_size; i++)
    {
        memset(surf->surface_list[i].data_ptr, value, surf->surface_list[i].data_size);
    }
    return 0;
}