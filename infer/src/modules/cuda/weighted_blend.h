#include <opencv2/core/core.hpp>

extern "C" {
cv::cuda::GpuMat weighted_blend_cuda(const cv::cuda::GpuMat &img1, const cv::cuda::GpuMat &img2, double alpha);
}