#pragma once

#include <string>
#include <mutex>
#include <queue>
#include <unordered_map>

#include "buf_surface.h"

namespace gddeploy {

// 各个设备需要继承和实现
class IMemAllcator {
public:
    virtual ~IMemAllcator() {}
    virtual int Create(BufSurfaceCreateParams *params) = 0;
    virtual int Destroy() = 0;
    virtual int Alloc(BufSurface *surf) = 0;
    virtual int Free(BufSurface *surf) = 0;
    virtual int Copy(BufSurface *src_surf, BufSurface *dst_surf) = 0;
    virtual int Memset(BufSurface *surf, int value) = 0;
};

IMemAllcator *CreateMemAllocator(BufSurfaceMemType mem_type);

class MemPool {
public:
    MemPool() = default;
    ~MemPool() {
        // if (allocator_) delete allocator_, allocator_ = nullptr;
    }
    int Create(BufSurfaceCreateParams *params, uint32_t block_num);
    int Destroy();
    int Alloc(BufSurface *surf);
    int Free(BufSurface *surf);

 private:
    std::mutex mutex_;
    std::queue<BufSurface> cache_;

    bool created_ = false;
    int device_id_ = 0;
    uint32_t alloc_count_ = 0;
    IMemAllcator *allocator_ = nullptr;
    bool is_fake_mapped_ = false;
};

class IMemAllcatorManager {
public:
    static IMemAllcatorManager* Instance() noexcept{
        if (pInstance_ == nullptr){
            pInstance_ = new IMemAllcatorManager();
        }
        return pInstance_;
    }    
    IMemAllcator* GetMemAllcator(std::string manufacturer){
        if (mem_.find(manufacturer) != mem_.end())
            return mem_[manufacturer];
        return nullptr;
    }

    int RegisterMem(std::string manufacturer, IMemAllcator *mem) noexcept{
        mem_[manufacturer] = mem;
        return 0;
    }

private:
    static IMemAllcatorManager *pInstance_;
    std::unordered_map<std::string, IMemAllcator*> mem_;
};

int register_mem_module();

//  for non-pool case
int CreateSurface(BufSurfaceCreateParams *params, BufSurface *surf);
int DestroySurface(BufSurface *surf);

}
