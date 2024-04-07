#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface_impl.h"
#include "common/logger.h"
#include <cstddef>
#include <unistd.h>
#include <algorithm>
#include <memory>
#include <thread>
#include <iostream>

namespace gddeploy {
//
// BufSurfaceWrapper
//
BufSurface *BufSurfaceWrapper::GetBufSurface() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return surf_;
}

BufSurface *BufSurfaceWrapper::BufSurfaceChown() {
    std::unique_lock<std::mutex> lk(mutex_);
    BufSurface *surf = surf_;
    surf_ = nullptr;
    return surf;
}

BufSurfaceParams *BufSurfaceWrapper::GetSurfaceParams(uint32_t batch_idx) const {
    std::unique_lock<std::mutex> lk(mutex_);
    return &surf_->surface_list[batch_idx];
}

uint32_t BufSurfaceWrapper::GetNumFilled() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return surf_->num_filled;
}

BufSurfaceColorFormat BufSurfaceWrapper::GetColorFormat() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return GetSurfaceParamsPriv(0)->color_format;
}

uint32_t BufSurfaceWrapper::GetBatch() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return surf_->batch_size;
}

uint32_t BufSurfaceWrapper::GetWidth() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return GetSurfaceParamsPriv(0)->width;
}

uint32_t BufSurfaceWrapper::GetHeight() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return GetSurfaceParamsPriv(0)->height;
}

uint32_t BufSurfaceWrapper::GetStride(uint32_t i) const {
    std::unique_lock<std::mutex> lk(mutex_);
    BufSurfacePlaneParams *params = &(GetSurfaceParamsPriv(0)->plane_params);
    if (i < 0 || i >= params->num_planes) {
        GDDEPLOY_ERROR("[BufSurfaceWrapper] GetStride(): plane index is invalid.");
        return 0;
    }
    return params->pitch[i];
}

uint32_t BufSurfaceWrapper::GetPlaneNum() const {
    std::unique_lock<std::mutex> lk(mutex_);
    BufSurfacePlaneParams *params = &(GetSurfaceParamsPriv(0)->plane_params);
    return params->num_planes;
}

uint32_t BufSurfaceWrapper::GetPlaneBytes(uint32_t i) const {
    std::unique_lock<std::mutex> lk(mutex_);
    BufSurfacePlaneParams *params = &(GetSurfaceParamsPriv(0)->plane_params);
    if (i < 0 || i >= params->num_planes) {
        GDDEPLOY_ERROR("[BufSurfaceWrapper] GetPlaneBytes(): plane index is invalid.");
        return 0;
    }
    return params->psize[i];
}

int BufSurfaceWrapper::GetDeviceId() const {
    std::unique_lock<std::mutex> lk(mutex_);
    return surf_->device_id;
}

void *BufSurfaceWrapper::GetData(uint32_t plane_idx, uint32_t batch_idx) {
    std::unique_lock<std::mutex> lk(mutex_);
    BufSurfaceParams *params = GetSurfaceParamsPriv(batch_idx);
    unsigned char *addr = static_cast<unsigned char *>(params->data_ptr);
    return static_cast<void *>(addr + params->plane_params.offset[plane_idx]);
}

void *BufSurfaceWrapper::GetMappedData(uint32_t plane_idx, uint32_t batch_idx) {
    std::unique_lock<std::mutex> lk(mutex_);
    BufSurfaceParams *params = GetSurfaceParamsPriv(batch_idx);
    unsigned char *addr = static_cast<unsigned char *>(params->mapped_data_ptr);
    return static_cast<void *>(addr + params->plane_params.offset[plane_idx]);
}

uint64_t BufSurfaceWrapper::GetPts() const {
    std::unique_lock<std::mutex> lk(mutex_);
    if (surf_) {
        return surf_->pts;
    } else {
        return pts_;
    }
}

void BufSurfaceWrapper::SetPts(uint64_t pts) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (surf_) {
        surf_->pts = pts;
    } else {
        pts_ = pts;
    }
}

void *BufSurfaceWrapper::GetDeviceData(uint32_t plane_idx, uint32_t batch_idx, BufSurfaceMemType mem_type)
{
    std::unique_lock<std::mutex> lk(mutex_);
    
    unsigned char *addr = nullptr;
    BufSurfaceParams *params = GetSurfaceParamsPriv(batch_idx);

    if (surf_->mem_type == mem_type) {
        params->mapped_data_ptr = params->data_ptr;
    } else {
        IMemAllcator *allocator = CreateMemAllocator(mem_type);
        if (allocator == nullptr){
            GDDEPLOY_ERROR("CreateMemAllocator: Unsupported memory type: {}", mem_type);
            return nullptr;
        }

        auto surf = surf_;
        BufSurface device_surf = *surf;
        device_surf.mem_type = mem_type;
        device_surf.surface_list[0].data_ptr = nullptr;
        IMemAllcator *device_allocator = CreateMemAllocator(GDDEPLOY_BUF_MEM_SYSTEM);
        if (device_allocator == nullptr){
            GDDEPLOY_ERROR("CreateMemAllocator: Unsupported memory type: GDDEPLOY_BUF_MEM_SYSTEM");
            return nullptr;
        }
        device_allocator->Alloc(&device_surf);

        allocator->Copy(surf, &device_surf);

        params->mapped_data_ptr = device_surf.surface_list[0].data_ptr;
    }
 
    addr = static_cast<unsigned char *>(params->mapped_data_ptr);
    if (addr) {
        return static_cast<void *>(addr + params->plane_params.offset[plane_idx]);
    }

    GDDEPLOY_ERROR("[BufSurfaceWrapper] GetHostData(): Unsupported memory type");
    return nullptr;
}

void *BufSurfaceWrapper::GetHostData(uint32_t plane_idx, uint32_t batch_idx) {
    std::unique_lock<std::mutex> lk(mutex_);
    
    unsigned char *addr = nullptr;
    BufSurfaceParams *params = GetSurfaceParamsPriv(batch_idx);
    if (surf_->mem_type != GDDEPLOY_BUF_MEM_PINNED && surf_->mem_type != GDDEPLOY_BUF_MEM_SYSTEM) {
        IMemAllcator *allocator = CreateMemAllocator(surf_->mem_type);
        if (allocator == nullptr){
            GDDEPLOY_ERROR("CreateMemAllocator: Unsupported memory type: {}", surf_->mem_type);
            return nullptr;
        }

        IMemAllcator *host_allocator = CreateMemAllocator(GDDEPLOY_BUF_MEM_SYSTEM);
        if (host_allocator == nullptr){
            GDDEPLOY_ERROR("CreateMemAllocator: Unsupported memory type: GDDEPLOY_BUF_MEM_SYSTEM");
            return nullptr;
        }
        BufSurface host_surf = *surf_;
        host_surf.surface_list = reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * host_surf.batch_size)); 
        memcpy(host_surf.surface_list, surf_->surface_list, sizeof(BufSurfaceParams) * host_surf.batch_size);
        host_surf.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;

        if (params->mapped_data_ptr != nullptr){
            free(params->mapped_data_ptr);
            params->mapped_data_ptr = nullptr;
        }
            // auto surf = surf_;
        host_surf.surface_list[0].data_ptr = nullptr;
        
        // BufSurfaceCreateParams create_params;
        // memset(&create_params, 0, sizeof(create_params));
        // create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
        // create_params.force_align_1 = 1;  // to meet mm's requirement
        // create_params.device_id = 0;
        // create_params.batch_size = 1;
        // // create_params.size = host_surf.surface_list[0].width * host_surf.surface_list[0].height * 24 * sizeof(float);
        // create_params.size = host_surf.surface_list[0].data_size;
        // create_params.width = host_surf.surface_list[0].width;
        // create_params.height = host_surf.surface_list[0].height;
        // create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;
        // host_allocator->Create(&create_params);

        host_allocator->Alloc(&host_surf);

        allocator->Copy(surf_, &host_surf);

        params->mapped_data_ptr = host_surf.surface_list[0].data_ptr;
        
    } else {
        params->mapped_data_ptr = params->data_ptr;
    }    

    addr = static_cast<unsigned char *>(params->mapped_data_ptr);
    if (addr) {
        // BufSurfaceSyncForCpu(surf_, batch_idx, plane_idx);
        return static_cast<void *>(addr + params->plane_params.offset[plane_idx]);
        // return static_cast<void *>(addr + surf_->surface_list[batch_idx].data_size);
    }

    // // workaround, FIXME,  copy data from device to host here
    // if (surf_->mem_type == GDDEPLOY_BUF_MEM_DEVICE) {
    //     if (surf_->is_contiguous) {
    //         size_t total_size = surf_->batch_size * params->data_size;
    //         host_data_[0].reset(new unsigned char[(total_size + 63) / 64 * 64]);
    //         CALL_CNRT_FUNC(cnrtMemcpy(host_data_[0].get(), surf_->surface_list[0].data_ptr, total_size,
    //                                 cnrtMemcpyDevToHost),
    //                         "[BufSurfaceWrapper] GetHostData(): data is contiguous, copy data D2H failed");
    //         for (size_t i = 0; i < surf_->batch_size; i++) {
    //             GetSurfaceParamsPriv(i)->mapped_data_ptr = host_data_[0].get() + i * params->data_size;
    //         }
    //         addr = static_cast<unsigned char *>(params->mapped_data_ptr);
    //         return static_cast<void *>(addr + params->plane_params.offset[plane_idx]);
    //     } else {
    //         if (batch_idx >= 128) {
    //             GDDEPLOY_ERROR("[BufSurfaceWrapper] GetHostData(): batch index should not be greater than 128");
    //             return nullptr;
    //         }
    //         host_data_[batch_idx].reset(new unsigned char[(GetSurfaceParamsPriv(batch_idx)->data_size + 63) / 64 * 64]);
    //         CALL_CNRT_FUNC(cnrtMemcpy(host_data_[batch_idx].get(), GetSurfaceParamsPriv(batch_idx)->data_ptr,
    //                                 GetSurfaceParamsPriv(batch_idx)->data_size, cnrtMemcpyDevToHost),
    //                         "[BufSurfaceWrapper] GetHostData(): copy data D2H failed, batch_idx = " +
    //                         std::to_string(batch_idx));
    //         GetSurfaceParamsPriv(batch_idx)->mapped_data_ptr = host_data_[batch_idx].get();
    //         addr = static_cast<unsigned char *>(params->mapped_data_ptr);
    //         return static_cast<void *>(addr + params->plane_params.offset[plane_idx]);
    //     }
    // }
    GDDEPLOY_ERROR("[BufSurfaceWrapper] GetHostData(): Unsupported memory type");
    return nullptr;
}

void BufSurfaceWrapper::SyncHostToDevice(uint32_t plane_idx, uint32_t batch_idx) {
    // if (surf_->mem_type == GDDEPLOY_BUF_MEM_DEVICE) {
    //     if (batch_idx >= 0 && batch_idx < 128 && host_data_[batch_idx]) {
    //         CALL_CNRT_FUNC(cnrtMemcpy(surf_->surface_list[batch_idx].data_ptr, host_data_[batch_idx].get(),
    //                                 surf_->surface_list[batch_idx].data_size, cnrtMemcpyHostToDev),
    //                         "[BufSurfaceWrapper] SyncHostToDevice(): copy data H2D failed, batch_idx = " +
    //                         std::to_string(batch_idx));
    //         return;
    //     }

    //     if (batch_idx == (uint32_t)(-1)) {
    //         if (surf_->is_contiguous) {
    //         if (!host_data_[0]) {
    //             GDDEPLOY_ERROR("[BufSurfaceWrapper] SyncHostToDevice(): Host data is null");
    //             return;
    //         }
    //         size_t total_size = surf_->batch_size * GetSurfaceParamsPriv(0)->data_size;
    //         CALL_CNRT_FUNC(cnrtMemcpy(surf_->surface_list[0].data_ptr, host_data_[0].get(),
    //                                     total_size, cnrtMemcpyHostToDev),
    //                         "[BufSurfaceWrapper] SyncHostToDevice(): data is contiguous, copy data H2D failed");
    //         } else {
    //         if (surf_->batch_size >= 128) {
    //             GDDEPLOY_ERROR("[BufSurfaceWrapper] SyncHostToDevice: batch size should not be greater than 128,"
    //                        , " which is: ", surf_->batch_size);
    //             return;
    //         }
    //         for (uint32_t i = 0; i < surf_->batch_size; i++) {
    //             CALL_CNRT_FUNC(cnrtMemcpy(surf_->surface_list[i].data_ptr, host_data_[i].get(),
    //                                     surf_->surface_list[i].data_size, cnrtMemcpyHostToDev),
    //                             "[BufSurfaceWrapper] SyncHostToDevice(): copy data H2D failed, batch_idx = " +
    //                             std::to_string(batch_idx));
    //         }
    //         }
    //     }
    //     return;
    // }
    BufSurfaceSyncForDevice(surf_, batch_idx, plane_idx);
}
//
// BufPool
//
int BufPool::CreatePool(BufSurfaceCreateParams *params, uint32_t block_count) {
    std::unique_lock<std::mutex> lk(mutex_);

    int ret = BufPoolCreate(&pool_, params, block_count);
    if (ret != 0) {
        GDDEPLOY_ERROR("[BufPool] CreatePool(): Create BufSurface pool failed");
        return -1;
    }

    stopped_ = false;
    return 0;
}

void BufPool::DestroyPool(int timeout_ms) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (stopped_) {
        GDDEPLOY_ERROR("[BufPool] DestroyPool(): Pool has been stoped");
        return;
    }
    stopped_ = true;
    int count = timeout_ms + 1;
    int retry_cnt = 1;
    while (1) {

        if (pool_) {
            int ret;
            ret = BufPoolDestroy(pool_);
            if (ret == 0) { return; }

            count -= retry_cnt;
            GDDEPLOY_ERROR("[BufPool] DestroyPool(): retry, remaining times: {}", count);
            if (count <= 0) {
                GDDEPLOY_ERROR("[BufPool] DestroyPool(): Maximum number of attempts reached: {}",timeout_ms);
                return;
            }

            lk.unlock();
            usleep(1000 * retry_cnt);
            retry_cnt = std::min(retry_cnt * 2, 10);
            lk.lock();
        }
        return;
    }
}

BufSurfWrapperPtr BufPool::GetBufSurfaceWrapper(int timeout_ms) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (!pool_) {
        GDDEPLOY_ERROR("[BufPool] GetBufSurfaceWrapper(): Pool is not created");
        return nullptr;
    }

    BufSurface *surf = nullptr;
    int count = timeout_ms + 1;
    int retry_cnt = 1;
    while (1) {
        if (stopped_) {
            // Destroy called, disable alloc-new-block
            GDDEPLOY_ERROR("[BufPool] GetBufSurfaceWrapper(): Pool is stopped");
            return nullptr;
        }

        int ret = BufSurfaceCreateFromPool(&surf, pool_);
        if (ret == 0) {
            return std::make_shared<BufSurfaceWrapper>(surf);
        }
        count -= retry_cnt;
        GDDEPLOY_ERROR("[BufPool] GetBufSurfaceWrapper(): retry, remaining times: {}", count);
        if (count <= 0) {
            GDDEPLOY_ERROR("[BufPool] GetBufSurfaceWrapper(): Maximum number of attempts reached: {}", timeout_ms);
            return nullptr;
        }

        lk.unlock();
        usleep(1000 * retry_cnt);
        retry_cnt = std::min(retry_cnt * 2, 10);
        lk.lock();
    }
    return nullptr;
}


int GetColorFormatInfo(BufSurfaceColorFormat fmt, uint32_t width, uint32_t height, uint32_t align_size_w,
                       uint32_t align_size_h, BufSurfacePlaneParams *params) {
    memset(params, 0, sizeof(BufSurfacePlaneParams));
    switch (fmt) {
        case GDDEPLOY_BUF_COLOR_FORMAT_YUV420:
            params->num_planes = 3;
            for (uint32_t i = 0; i < params->num_planes; i++) {
                params->width[i] = i == 0 ? width : width / 2;
                params->height[i] = i == 0 ? height : height / 2;
                params->bytes_per_pix[i] = 1;
                params->pitch[i] =
                    (params->width[i] * params->bytes_per_pix[i] + align_size_w - 1) / align_size_w * align_size_w;
                params->psize[i] = params->pitch[i] * ((params->height[i] + align_size_h - 1) / align_size_h * align_size_h);
            }
            params->offset[0] = 0;
            params->offset[1] = params->psize[0];
            params->offset[2] = params->psize[0] + params->psize[1];
            return 0;
        case GDDEPLOY_BUF_COLOR_FORMAT_NV12:
        case GDDEPLOY_BUF_COLOR_FORMAT_NV21:
            params->num_planes = 2;
            for (uint32_t i = 0; i < params->num_planes; i++) {
                params->width[i] = width;
                params->height[i] = (i == 0) ? height : height / 2;
                params->bytes_per_pix[i] = 1;
                params->pitch[i] =
                    (params->width[i] * params->bytes_per_pix[i] + align_size_w - 1) / align_size_w * align_size_w;
                params->psize[i] = params->pitch[i] * ((params->height[i] + align_size_h - 1) / align_size_h * align_size_h);
            }
            params->offset[0] = 0;
            params->offset[1] = params->psize[0];
            return 0;  
        case GDDEPLOY_BUF_COLOR_FORMAT_RGB:
        case GDDEPLOY_BUF_COLOR_FORMAT_BGR:
            params->num_planes = 1;
            params->width[0] = width;
            params->height[0] = height;
            params->bytes_per_pix[0] = 3;
            params->pitch[0] = (width * params->bytes_per_pix[0] + align_size_w - 1) / align_size_w * align_size_w;
            params->offset[0] = 0;
            params->psize[0] = params->pitch[0] * ((params->height[0] + align_size_h - 1) / align_size_h * align_size_h);
            return 0;
        case GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER:
        case GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER:
            params->num_planes = 3;
            for (uint32_t i = 0; i < params->num_planes; i++) {
                params->width[i] = width;
                params->height[i] = height;
                params->bytes_per_pix[i] = 1;
                params->pitch[i] =
                    (params->width[i] * params->bytes_per_pix[i] + align_size_w - 1) / align_size_w * align_size_w;
                params->psize[i] = params->pitch[i] * ((params->height[i] + align_size_h - 1) / align_size_h * align_size_h);
            }
            params->offset[0] = 0;
            params->offset[1] = params->psize[0];
            params->offset[2] = params->psize[0] + params->psize[1];
            return 0;
        case GDDEPLOY_BUF_COLOR_FORMAT_INVALID:
        default: {
            GDDEPLOY_ERROR("GetColorFormatInfo(): Unsupported color format: {}", fmt);
            return -1;
        }
    }

    return 0;
}

int CheckParams(BufSurfaceCreateParams *params) {
    return 0;
}

}  // namespace cnedk
