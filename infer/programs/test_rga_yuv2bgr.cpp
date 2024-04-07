#include <bits/types/FILE.h>
#include <cstdlib>

#ifdef WITH_RV1126

#include <opencv2/highgui.hpp>
#include <rga/RgaUtils.h>
#include <rga/im2d.h>
#include <rga/rga.h>

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))
#define SRC_FORMAT RK_FORMAT_YCbCr_420_P
#define DST_FORMAT RK_FORMAT_BGR_888

#endif

int main(int argc, char *argv[]) {
    char *path = argv[1];
    int SRC_WIDTH = atoi(argv[2]);
    int SRC_HEIGHT = atoi(argv[3]);
    int DST_WIDTH = atoi(argv[2]);
    int DST_HEIGHT = atoi(argv[3]);

#ifdef WITH_RV1126
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) { return -1; }
    fseek(fp, 0, SEEK_END);
    auto buffer = std::vector<u_char>(ftell(fp));
    fseek(fp, 0, SEEK_SET);
    fread(buffer.data(), sizeof(u_char), buffer.size(), fp);
    fclose(fp);

    auto src_buf = (char *)malloc(SRC_WIDTH * SRC_HEIGHT * get_bpp_from_format(SRC_FORMAT));
    auto dst_buf = (char *)malloc(DST_WIDTH * DST_HEIGHT * get_bpp_from_format(DST_FORMAT));

    memcpy(src_buf, buffer.data(), SRC_WIDTH * SRC_HEIGHT * 3 / 2);

    auto src = wrapbuffer_virtualaddr(src_buf, SRC_WIDTH, SRC_HEIGHT, SRC_FORMAT);
    auto dst = wrapbuffer_virtualaddr(dst_buf, DST_WIDTH, DST_HEIGHT, DST_FORMAT);
    if (src.width == 0 || dst.width == 0) { printf("%s, %s\n", __FUNCTION__, imStrError()); }

    int STATUS = imcvtcolor(src, dst, src.format, dst.format);

    cv::Mat mat_iamge(DST_HEIGHT, DST_WIDTH, CV_8UC3);
    memcpy(mat_iamge.data, dst_buf, DST_WIDTH * DST_HEIGHT * 3);
    cv::imwrite("test.jpg", mat_iamge);

    free(src_buf);
    free(dst_buf);

#endif
}