#include "../src/modules/wrapper/nv_jpeg_decoder.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <iterator>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <thread>
#include <vector>

int main(int argc, char *argv[]) {
#if defined(WITH_NVIDIA)

    gddi::wrapper::JpegImage jpeg_image;

    std::ifstream file(argv[1], std::ios::binary);
    if (!file.is_open()) { return 1; }
    file.seekg(0, std::ios::end);
    jpeg_image.data = std::vector<unsigned char>(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read((char *)jpeg_image.data.data(), jpeg_image.data.size());
    file.close();

    std::string buffer((std::istream_iterator<unsigned char>(file)),
                       std::istream_iterator<unsigned char>());
    auto jpeg_decoder_ = std::make_unique<gddi::wrapper::NvJpegDecoder>(16, 1);

    int64_t index{0};
    auto start = std::chrono::steady_clock::now();
    jpeg_image.callback = [&start](const int64_t idx, const cv::cuda::GpuMat &) {
        if (idx % 1000 == 0) {
            printf("================= %ld frame, Total time: %ldms\n", idx,
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count());
            start = std::chrono::steady_clock::now();
        }
    };
    while (true) {
        jpeg_image.index = index++;

        jpeg_decoder_->process_image(jpeg_image);

        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
#endif

    return 0;
}