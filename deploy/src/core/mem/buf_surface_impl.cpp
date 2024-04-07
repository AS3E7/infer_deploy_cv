#include "core/mem/buf_surface_impl.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"
#include "common/logger.h"
#include <string>
#include <thread>
#include <iostream>

namespace gddeploy{

IMemAllcatorManager *IMemAllcatorManager::pInstance_ = nullptr;

int MemPool::Create(BufSurfaceCreateParams *params, uint32_t block_num) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (created_) {
        GDDEPLOY_ERROR("[MemPool] Create(): Pool has been created");
        return -1;
    }

    if (CheckParams(params) < 0) {
        GDDEPLOY_ERROR("[MemPool] Create(): Parameters are invalid");
        return -1;
    }
    device_id_ = params->device_id;

    // params->mem_type = GDDEPLOY_BUF_MEM_DEVICE;

    allocator_ = CreateMemAllocator(params->mem_type);
    if (!allocator_) {
        GDDEPLOY_ERROR("[MemPool] Create(): Create memory allocator pointer failed");
        return -1;
    }

    if (allocator_->Create(params) < 0) {
        GDDEPLOY_ERROR("[MemPool] Create(): Memory allocator initialize resources failed");
        return -1;
    }

    is_fake_mapped_ = (params->mem_type == GDDEPLOY_BUF_MEM_DEVICE);
    // cache the blocks
    for (uint32_t i = 0; i < block_num; i++) {
        BufSurface surf;
        if (allocator_->Alloc(&surf) < 0) {
            GDDEPLOY_ERROR("[MemPool] Create(): Memory allocator alloc BufSurface failed");
            return -1;
        }
        surf.opaque = reinterpret_cast<void *>(this);
        cache_.push(surf);
    }

    alloc_count_ = 0;
    created_ = true;
    return 0;
}

int MemPool::Destroy() {
    std::unique_lock<std::mutex> lk(mutex_);
    if (!created_) {
        GDDEPLOY_ERROR("[MemPool] Destroy(): Memory pool is not created");
        return -1;
    }

    while (alloc_count_) {
        lk.unlock();
        std::this_thread::yield();
        lk.lock();
    }
    while (!cache_.empty()) {
        auto surf = cache_.front();
        cache_.pop();
        allocator_->Free(&surf);
    }

    // FIXME
    if (allocator_->Destroy() < 0) {
        GDDEPLOY_ERROR("[MemPool] Destroy(): Destroy memory allocator failed");
        return -1;
    }
    // delete allocator_, allocator_ = nullptr;

    alloc_count_ = 0;
    created_ = false;
    return 0;
}


int MemPool::Alloc(BufSurface *surf) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (!created_) {
        GDDEPLOY_ERROR("[MemPool] Alloc(): Memory pool is not created");
        return -1;
    }

    if (cache_.empty()) {
        GDDEPLOY_ERROR("[MemPool] Alloc(): Memory cache is empty");
        return -1;
    }

    *surf = cache_.front();
    cache_.pop();

    ++alloc_count_;
    return 0;
}

int MemPool::Free(BufSurface *surf) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (!created_) {
        GDDEPLOY_ERROR("[MemPool] Free(): Memory pool is not created");
        return -1;
    }

    if (is_fake_mapped_) {
        // reset mapped_data_ptr to zero
        for (size_t i = 0; i < surf->batch_size; i++) 
            surf->surface_list[i].mapped_data_ptr = nullptr;
    }
    cache_.push(*surf);
    --alloc_count_;
    return 0;
}

std::string ConvertMemType2String(BufSurfaceMemType mem_type)
{
    std::string device_name = ""; 
    switch (mem_type) {
        case GDDEPLOY_BUF_MEM_BMNN:
            device_name = "SOPHGO";
            break;
        case GDDEPLOY_BUF_MEM_SYSTEM:
            device_name = "cpu";
            break;
        case GDDEPLOY_BUF_MEM_CAMBRICON:
            device_name = "Cambricon";
            break;
        case GDDEPLOY_BUF_MEM_NVIDIA:
            device_name = "Nvidia";
            break;
        case GDDEPLOY_BUF_MEM_TS:
            device_name = "ts";
            break;
        default:
            device_name = "";
    }

    return device_name;
}

IMemAllcator *CreateMemAllocator(BufSurfaceMemType mem_type) {
    std::string device_name = ConvertMemType2String(mem_type);
    if (device_name == ""){
        GDDEPLOY_ERROR("CreateMemAllocator(): Unsupported memory type: {}", mem_type);
        return nullptr;
    }

    auto mgr = IMemAllcatorManager::Instance();
    IMemAllcator* allocator = mgr->GetMemAllcator(device_name);
    if (allocator != nullptr)
        return allocator;

    return nullptr;
}

// for non-pool case
int CreateSurface(BufSurfaceCreateParams *params, BufSurface *surf) {
    if (CheckParams(params) < 0) {
        GDDEPLOY_ERROR("CreateSurface(): Parameters are invalid");
        return -1;
    }

    // params->mem_type = GDDEPLOY_BUF_MEM_DEVICE;

    IMemAllcator *allocator = CreateMemAllocator(params->mem_type);
    if (allocator == nullptr){
        GDDEPLOY_ERROR("CreateMemAllocator(): Unsupported memory type: {}", params->mem_type);
        return -1;
    }
   
    if (allocator->Create(params) < 0) {
        GDDEPLOY_ERROR("CreateSurface(): Memory allocator initialize resources failed. mem_type = {}"
                   , params->mem_type);
        return -1;
    }
    if (allocator->Alloc(surf) < 0) {
        GDDEPLOY_ERROR("CreateSurface(): Memory allocator create BufSurface failed. mem_type = {}"
                   , params->mem_type);
        return -1;
    }
    return 0;
}


int DestroySurface(BufSurface *surf) {

    IMemAllcator *allocator = CreateMemAllocator(surf->mem_type);
    if (allocator == nullptr){
         GDDEPLOY_ERROR("CreateMemAllocator(): Unsupported memory type: {}", surf->mem_type);
        return -1;
    }

    if (allocator->Free(surf) < 0) {
      GDDEPLOY_ERROR("DestroySurface(): Memory allocator destroy BufSurface failed. mem_type = {}"
                , surf->mem_type);
      return -1;
    }

    return 0;
}

}

#include "cpu/buf_surface_impl_system.h"
#ifdef WITH_BM1684
#include "register/devices/bmnn/bmnn_buf_surface_impl.h"
#endif

#ifdef WITH_MLU220
#include "register/devices/cambricon/cambricon_buf_surface_impl.h"
#endif

#ifdef WITH_MLU270
#include "register/devices/cambricon/cambricon_buf_surface_impl.h"
#endif

#ifdef WITH_NVIDIA
#include "register/devices/nvidia/nv_buf_surface_impl.h"
#endif

#ifdef WITH_TS
#include "register/devices/ts/ts_buf_surface_impl.h"
#endif

#ifdef WITH_ASCEND
#include "register/devices/ascend/ascend_buf_surface_impl.h"
#endif

namespace gddeploy{
int register_mem_module()
{
    IMemAllcatorManager* mem_mgr = IMemAllcatorManager::Instance();
    GDDEPLOY_INFO("[Register] register mem module");

    MemAllocatorSystem *mem_system = new MemAllocatorSystem();
    mem_mgr->RegisterMem("cpu", mem_system);
#ifdef WITH_BM1684
    MemAllocatorBmnn *mem_bmnn = new MemAllocatorBmnn();
    mem_mgr->RegisterMem("SOPHGO", mem_bmnn);
#endif

#ifdef WITH_MLU220
    MemAllocatorCambricon *mem_cambricon = new MemAllocatorCambricon();
    mem_mgr->RegisterMem("Cambricon", mem_cambricon);
#endif

#ifdef WITH_MLU270
    MemAllocatorCambricon *mem_cambricon = new MemAllocatorCambricon();
    mem_mgr->RegisterMem("Cambricon", mem_cambricon);
#endif

#ifdef WITH_NVIDIA
    MemAllocatorNv *mem_nv = new MemAllocatorNv();
    mem_mgr->RegisterMem("Nvidia", mem_nv);
#endif

#ifdef WITH_TS
    MemAllocatorTs *mem_ts = new MemAllocatorTs();
    mem_mgr->RegisterMem("ts", mem_ts);
#endif

#ifdef WITH_ASCEND
    MemAllocatorAscend *mem_ascend = new MemAllocatorAscend();
    mem_mgr->RegisterMem("Ascend", mem_ascend);
#endif

    return 0;
}
}