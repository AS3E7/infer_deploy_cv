#pragma once

#include <string>
#include <mutex>

#include "core/mem/buf_surface_impl.h"

#include "bmlib_runtime.h"

namespace gddeploy {

class MemAllocatorBmnn : public IMemAllcator {
public:
    MemAllocatorBmnn();
    ~MemAllocatorBmnn();
    int Create(BufSurfaceCreateParams *params) override;
    int Destroy() override;
    int Alloc(BufSurface *surf) override;
    int Free(BufSurface *surf) override;
    int Copy(BufSurface *src_surf, BufSurface *dst_surf) override;
    int Memset(BufSurface *surf, int value) override;

private:
    bool created_ = false;
    BufSurfaceCreateParams create_params_;
    BufSurfacePlaneParams plane_params_;
    size_t block_size_;

    bm_handle_t handle_;
    int device_id_;
};

}  // namespace gddploy