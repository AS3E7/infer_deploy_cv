#include <chrono>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <sys/types.h>
#include <thread>
#include <vector>

#if defined(WITH_RV1126)
#include "wrapper/mpp_jpeg_codec.h"
#include <opencv2/imgcodecs.hpp>

void running(int index) {
    std::ifstream in_file("jiwei.yuv");
    in_file.seekg(0, std::ios::end);
    auto image = std::vector<char>(in_file.tellg());
    in_file.seekg(0, std::ios::beg);
    in_file.read(image.data(), image.size());
    in_file.close();

    while (true) {
        auto jpeg_decoder = std::make_unique<gddi::codec::MppJpegCodec>(gddi::codec::CodecType::kEncoder);
        if (!jpeg_decoder->init_codecer(1920, 1080, 50)) { exit(1); }
        // jpeg_decoder->codec_image((uint8_t *)image.data(), [](const uint8_t *data, const size_t dlen) {
        //     std::ofstream file("output.jpg", std::ios::binary);
        //     file.write((char *)data, dlen);
        //     file.close();
        // });
        jpeg_decoder->save_image((uint8_t *)image.data(), ("output" + std::to_string(index) + ".jpg").c_str());
    }
}
#endif

int main() {
#if defined(WITH_RV1126)
    std::thread handle[3];
    for (int i = 0; i < 3; i++) { handle[i] = std::thread(running, i); }

    for (int i = 0; i < 3; i++) {
        if (handle[i].joinable()) { handle[i].join(); }
    }
#endif
    return 0;
}