#include "weighted_blend.h"
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudaimgproc.hpp>

__global__ void weighted_blend_kernel(const uchar *img1, const uchar *img2, uchar *output, int width, int height,
                                      float alpha) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) { return; }

    int idx = 4 * (y * width + x);

    double a1 = img1[idx + 3] * alpha;
    double a2 = img2[idx + 3] * (1 - alpha);
    double a_blended = a1 + a2;

    double b1 = img1[idx] * a1;
    double g1 = img1[idx + 1] * a1;
    double r1 = img1[idx + 2] * a1;
    double b2 = img2[idx] * a2;
    double g2 = img2[idx + 1] * a2;
    double r2 = img2[idx + 2] * a2;

    float img1_alpha = img1[idx + 3];
    float img2_alpha = img2[idx + 3];

    int b = a_blended != 0 ? static_cast<int>((b1 + b2) / a_blended) : 0;
    int g = a_blended != 0 ? static_cast<int>((g1 + g2) / a_blended) : 0;
    int r = a_blended != 0 ? static_cast<int>((r1 + r2) / a_blended) : 0;

    output[idx] = b;
    output[idx + 1] = g;
    output[idx + 2] = r;
    output[idx + 3] = a_blended;
}

cv::cuda::GpuMat weighted_blend_cuda(const cv::cuda::GpuMat &img1_bgra, const cv::cuda::GpuMat &img2_bgra,
                                     double alpha) {
    cv::cuda::GpuMat blended_image(img1_bgra.rows, img1_bgra.cols, CV_8UC4);

    dim3 block_dim(16, 16);
    dim3 grid_dim((img1_bgra.cols + block_dim.x - 1) / block_dim.x, (img1_bgra.rows + block_dim.y - 1) / block_dim.y);

    weighted_blend_kernel<<<grid_dim, block_dim>>>(img1_bgra.ptr(), img2_bgra.ptr(), blended_image.ptr(),
                                                   img1_bgra.cols, img1_bgra.rows, alpha);

    cudaDeviceSynchronize();

    return blended_image;
}