#pragma once

#include <string>
#include "core/cv.h"
#include "core/mem/buf_surface_util.h"

#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"

namespace gddeploy
{

// class CVPrivate;
class BmnnCV : public CV
{
public:
    BmnnCV();
    ~BmnnCV();

    int Resize(BufSurfWrapperPtr src, BufSurfWrapperPtr dst) override;

    int Crop(BufSurfWrapperPtr src, std::vector<BufSurfWrapperPtr> dst, std::vector<CropParam> crop_params) override;

    // int Normalize(BufSurface &src, BufSurface &dst) override;

    // int Csc(BufSurface &src, BufSurface &dst) override;
private:
    bm_handle_t bm_handle_;
};

} // namespace gddeploy