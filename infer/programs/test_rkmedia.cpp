#include <unistd.h>

#if defined(WITH_RV1126)
#include "wrapper/rkmedia_jpeg_codec.h"
#endif

int main() {
#if defined(WITH_RV1126)
    int width = 1920;
    int height = 1080;
    size_t yuv_data_size = 1920 * 1080 * 3 / 2;

    auto jpeg_decoder_ = std::make_unique<gddi::codec::RkMediaJpegCodec>();
    if (!jpeg_decoder_->init_codecer(width, height, 85)) {
        jpeg_decoder_.reset();
        return -1;
    }

    auto yuv_data = std::vector<uint8_t>(yuv_data_size);
    FILE *in_file = fopen("jiwei.yuv", "rb");
    if (!in_file) { return -1; }
    int len = fread(yuv_data.data(), sizeof(uint8_t), yuv_data_size, in_file);
    fclose(in_file);

    int index = 0;
    while (++index <= 10) {
        printf("=================================\n");
        std::vector<uint8_t> jpeg_data;
        jpeg_decoder_->codec_image(yuv_data.data(), jpeg_data);

        char jpeg_path[128];
        sprintf(jpeg_path, "test_jpeg%d.jpeg", index);
        FILE *out_file = fopen(jpeg_path, "w");
        if (out_file) {
            fwrite(jpeg_data.data(), 1, jpeg_data.size(), out_file);
            fclose(out_file);
        }
    }

#endif
    return 0;
}