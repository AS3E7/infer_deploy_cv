#ifndef __NV_JPEG_DECODER_H__
#define __NV_JPEG_DECODER_H__

#if defined(WITH_NVIDIA)

#include "blockingconcurrentqueue.h"
#include <cstdint>
#include <functional>
#include <nvjpeg.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <thread>
#include <vector>

namespace gddi {
namespace wrapper {

struct JpegImage {
    int64_t index;
    std::vector<uchar> data;
    std::function<void(const int64_t, const cv::cuda::GpuMat &image)> callback;
};

class NvJpegDecoderPrivate;

class NvJpegDecoder {
public:
    NvJpegDecoder(const size_t instances, const size_t batch_size = 4,
                  const nvjpegOutputFormat_t format = NVJPEG_OUTPUT_BGRI);
    ~NvJpegDecoder();

    void process_image(const JpegImage &image);

private:
    bool active{true};
    moodycamel::BlockingConcurrentQueue<JpegImage> cache_jpeg_data_;
    std::vector<std::thread> vec_hanlde_;
};

}// namespace wrapper
}// namespace gddi

#endif

#endif