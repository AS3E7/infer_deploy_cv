#pragma once

#include <string>
#include <mutex>

#include "core/mem/buf_surface_impl.h"

#include "rknn_api.h"

namespace gddeploy {

class MemAllocatorRk : public IMemAllcator {
public:
    MemAllocatorRk();
    ~MemAllocatorRk();
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

    // rknn_context ctx_;

    int device_id_;
};

}  // namespace gddploy