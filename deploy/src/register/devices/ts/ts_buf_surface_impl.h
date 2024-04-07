#pragma once

#include <string>
#include <mutex>

#include "core/mem/buf_surface_impl.h"
#include "ts_type.h"
#include "ts_comm_video.h"

namespace gddeploy {

class MemAllocatorTs : public IMemAllcator {
public:
    MemAllocatorTs() = default;
    ~MemAllocatorTs() = default;
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
};

}  // namespace gddploy

TS_S32 SAMPLE_VGS_GetFrameVb(VIDEO_FRAME_INFO_S *pstFrameInfo);
TS_S32 SAMPLE_VGS_ReleaseFrameVb(VIDEO_FRAME_INFO_S *pstFrameInfo);
PIXEL_FORMAT_E convertFormat(BufSurfaceColorFormat format);