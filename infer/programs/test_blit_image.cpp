#include "modules/cvrelate/graphics.h"
#include "utils.hpp"
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef WITH_NVIDIA
#include "modules/cuda/weighted_blend.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#ifdef WITH_MLU220
#include "modules/wrapper/mlu220_wrapper.hpp"
#endif

inline bool read_file(const std::string &file_name, std::vector<uint8_t> &data) noexcept {
    try {
        auto p = boost::filesystem::path(file_name);
        if (boost::filesystem::exists(p)) {
            auto size = boost::filesystem::file_size(p);
            data.resize(size);

            auto f = std::fopen(file_name.c_str(), "rb");
            auto rd_size = std::fread(data.data(), sizeof(uint8_t), size / sizeof(uint8_t), f);
            std::fclose(f);
            if (rd_size == size) { return true; }
        }
        return false;
    } catch (std::exception &e) { return false; }
}

inline bool write_file(const std::string &file_name, const std::vector<uint8_t> &data) noexcept {
    auto f = std::fopen(file_name.c_str(), "wb");
    auto rd_size = std::fwrite(data.data(), sizeof(uint8_t), data.size() / sizeof(uint8_t), f);
    std::fclose(f);
    return true;
}

cv::Mat weighted_blend(const cv::Mat &img1, const cv::Mat &img2_bgra, double alpha) {
    // 将输入图像转换为带有透明度通道的 BGRA 格式
    cv::Mat img1_bgra;
    cv::cvtColor(img1, img1_bgra, cv::COLOR_BGR2BGRA);
    // cv::cvtColor(img2, img2_bgra, cv::COLOR_BGR2BGRA);

    // 创建一个新的空白 BGRA 图像，用于存储融合后的图像
    cv::Mat blended_image(img1.rows, img1.cols, CV_8UC4);

    // 遍历输入图像的每个像素
    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img1.cols; ++x) {
            cv::Vec4b px1 = img1_bgra.at<cv::Vec4b>(y, x);
            cv::Vec4b px2 = img2_bgra.at<cv::Vec4b>(y, x);

            // Step 1: Pre-blend alpha channels
            double a1 = px1[3] * alpha;
            double a2 = px2[3] * (1 - alpha);
            double a_blended = a1 + a2;

            // Step 2: Calculate alpha-weighted color values
            double b1 = px1[0] * a1;
            double g1 = px1[1] * a1;
            double r1 = px1[2] * a1;
            double b2 = px2[0] * a2;
            double g2 = px2[1] * a2;
            double r2 = px2[2] * a2;

            // Step 3: Blend color values using pre-blended alpha channel
            int b = a_blended != 0 ? static_cast<int>((b1 + b2) / a_blended) : 0;
            int g = a_blended != 0 ? static_cast<int>((g1 + g2) / a_blended) : 0;
            int r = a_blended != 0 ? static_cast<int>((r1 + r2) / a_blended) : 0;

            blended_image.at<cv::Vec4b>(y, x) = cv::Vec4b(b, g, r, static_cast<int>(a_blended));
        }
    }

    return blended_image;
}

class BlendParallel : public cv::ParallelLoopBody {
public:
    BlendParallel(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &blended_image, double alpha)
        : img1_(img1), img2_(img2), blended_image_(blended_image), alpha_(alpha) {}

    void operator()(const cv::Range &range) const {
        for (int y = range.start; y < range.end; ++y) {
            for (int x = 0; x < img1_.cols; ++x) {
                cv::Vec4b px1 = img1_.at<cv::Vec4b>(y, x);
                cv::Vec4b px2 = img2_.at<cv::Vec4b>(y, x);

                // Step 1: Pre-blend alpha channels
                double a1 = px1[3] * alpha_;
                double a2 = px2[3] * (1 - alpha_);
                double a_blended = a1 + a2;

                // Step 2: Calculate alpha-weighted color values
                double b1 = px1[0] * a1;
                double g1 = px1[1] * a1;
                double r1 = px1[2] * a1;
                double b2 = px2[0] * a2;
                double g2 = px2[1] * a2;
                double r2 = px2[2] * a2;

                // Step 3: Blend color values using pre-blended alpha channel
                int b = a_blended != 0 ? static_cast<int>((b1 + b2) / a_blended) : 0;
                int g = a_blended != 0 ? static_cast<int>((g1 + g2) / a_blended) : 0;
                int r = a_blended != 0 ? static_cast<int>((r1 + r2) / a_blended) : 0;

                blended_image_.at<cv::Vec4b>(y, x) = cv::Vec4b(b, g, r, static_cast<int>(a_blended));
            }
        }
    }

private:
    const cv::Mat &img1_;
    const cv::Mat &img2_;
    cv::Mat &blended_image_;
    double alpha_;
};

cv::Mat weighted_blend2(const cv::Mat &img1, const cv::Mat &img2, double alpha) {
    cv::Mat img1_bgra, img2_bgra;
    cv::cvtColor(img1, img1_bgra, cv::COLOR_BGR2BGRA);
    cv::cvtColor(img2, img2_bgra, cv::COLOR_BGR2BGRA);

    cv::Mat blended_image(img1.rows, img1.cols, CV_8UC4);

    // 创建并行融合操作的实例
    BlendParallel blend_parallel(img1_bgra, img2_bgra, blended_image, alpha);

    // 使用 parallel_for_ 函数执行并行处理
    parallel_for_(cv::Range(0, img1.rows), blend_parallel);

    return blended_image;
}

int main(int argc, char *argv[]) {
    auto src_image = cv::imread("src_image.jpg");
    cv::cvtColor(src_image, src_image, cv::COLOR_BGR2BGRA);
    auto background_image = cv::imread("background.png", cv::IMREAD_UNCHANGED);

#if defined(WITH_NVIDIA)
    cv::cuda::GpuMat src_image_gpu, background_image_gpu;
    src_image_gpu.upload(src_image);
    background_image_gpu.upload(background_image);

    auto start = std::chrono::steady_clock::now();
    auto dst_image_gpu = weighted_blend_cuda(src_image_gpu, background_image_gpu, 0.2);
    printf("time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
    cv::Mat dst_image;
    dst_image_gpu.download(dst_image);
#else
    auto start = std::chrono::steady_clock::now();
    auto dst_image = weighted_blend2(src_image, background_image, 0.2);
    printf("time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
#endif
    cv::imwrite("blit_image.jpg", dst_image);

    return 0;
}