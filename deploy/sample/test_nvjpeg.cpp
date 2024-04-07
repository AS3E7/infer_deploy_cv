#include "../src/app/pic/nvjpeg.h"
#include <iostream>

int main(int argc, char **argv)
{
    NvjpegImageProcessor jpg_processor;

    std::string pic_name = "/data/data/pic/helmet2.jpg";
    BufSurface surf;

    std::vector<std::string> pic_files = {"/data/data/pic/helmet2.jpg", "/data/data/pic/helmet3.jpg"};

    jpg_processor.Decode(pic_files, &surf);

    std::string pic_save_name = "/data/preds/helmet2.jpg";
    std::vector<std::string> pic_save_files = {"/data/preds/helmet2.jpg", "/data/preds/helmet3.jpg"};

    jpg_processor.Encode(pic_save_files, &surf);

    return 0;
}