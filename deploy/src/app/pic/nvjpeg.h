#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include "core/mem/buf_surface.h"

class NvjpegImageProcessor {
public:
    NvjpegImageProcessor();

    ~NvjpegImageProcessor();

    int Decode(std::vector<std::string> &filenames, BufSurface *surf);

    int Encode(std::vector<std::string> &filenames, BufSurface *surf, int quality = 90);

private:
    nvjpegHandle_t nvjpeg_handle_ = nullptr;
    cudaStream_t stream_;
};
