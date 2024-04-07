#include "nv_jpeg_decoder.h"
#include "basic_logs.hpp"
#include "common_basic/thread_dbg_utils.hpp"
#include <chrono>
#include <fstream>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
#include <utility>
#include <vector>

inline int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
inline int dev_free(void *p) { return (int)cudaFree(p); }
inline int host_malloc(void **p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }
inline int host_free(void *p) { return (int)cudaFreeHost(p); }

#define CHECK_CUDA(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t _e = (call);                                                                                       \
        if (_e != cudaSuccess) {                                                                                       \
            spdlog::error("CUDA Runtime failure: '# {}' at {} : {} ", _e, __FILE__, __LINE__);                         \
            return false;                                                                                              \
        }                                                                                                              \
    }

#define CHECK_NVJPEG(call)                                                                                             \
    {                                                                                                                  \
        nvjpegStatus_t _e = (call);                                                                                    \
        if (_e != NVJPEG_STATUS_SUCCESS) {                                                                             \
            spdlog::error("NVJPEG failure: '# {}' at {} : {} ", _e, __FILE__, __LINE__);                               \
            return false;                                                                                              \
        }                                                                                                              \
    }

namespace gddi {
namespace wrapper {

class NvJpegDecoderPrivate {
public:
    NvJpegDecoderPrivate(const size_t batch_size, const nvjpegOutputFormat_t format)
        : batch_size_(batch_size), format_(format) {}

    ~NvJpegDecoderPrivate() {
        nvjpegDecoderDestroy(nvjpeg_decoder_);
        nvjpegJpegStateDestroy(nvjpeg_decoupled_state_);
        nvjpegDecodeParamsDestroy(nvjpeg_decode_params_);
        nvjpegBufferDeviceDestroy(device_buffer_);
        nvjpegJpegStreamDestroy(jpeg_streams_[0]);
        nvjpegJpegStreamDestroy(jpeg_streams_[1]);
        nvjpegBufferPinnedDestroy(pinned_buffers_[0]);
        nvjpegBufferPinnedDestroy(pinned_buffers_[1]);

        nvjpegDestroy(nvjpeg_handle_);
        nvjpegJpegStateDestroy(nvjpeg_state_);

        cudaStreamDestroy(stream_);
    }

    bool init_decoder() {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
        nvjpegPinnedAllocator_t pinned_allocator = {&host_malloc, &host_free};

        nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, &dev_allocator, &pinned_allocator,
                                               NVJPEG_FLAGS_DEFAULT, &nvjpeg_handle_);
        if (status == NVJPEG_STATUS_ARCH_MISMATCH) {
            hw_decode_available_ = false;
            CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &pinned_allocator, NVJPEG_FLAGS_DEFAULT,
                                        &nvjpeg_handle_));
        } else {
            CHECK_NVJPEG(status);
        }

        CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle_, &nvjpeg_state_));

        CHECK_NVJPEG(nvjpegDecoderCreate(nvjpeg_handle_, NVJPEG_BACKEND_DEFAULT, &nvjpeg_decoder_));
        CHECK_NVJPEG(nvjpegDecoderStateCreate(nvjpeg_handle_, nvjpeg_decoder_, &nvjpeg_decoupled_state_));
        CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &pinned_buffers_[0]));
        CHECK_NVJPEG(nvjpegBufferPinnedCreate(nvjpeg_handle_, NULL, &pinned_buffers_[1]));
        CHECK_NVJPEG(nvjpegBufferDeviceCreate(nvjpeg_handle_, NULL, &device_buffer_));

        CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[0]));
        CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle_, &jpeg_streams_[1]));
        CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle_, &nvjpeg_decode_params_));

        return true;
    }
    bool init_nvjpeg_image(const std::vector<JpegImage> &jpeg_images) {
        vec_iout_.clear();
        img_width_.clear();
        img_height_.clear();

        for (const auto &item : jpeg_images) {

            nvjpegImage_t iout, isz;
            for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
                iout.channel[c] = nullptr;
                iout.pitch[c] = 0;
                isz.pitch[c] = 0;
            }

            int channels;
            int widths[NVJPEG_MAX_COMPONENT];
            int heights[NVJPEG_MAX_COMPONENT];
            nvjpegChromaSubsampling_t subsampling;
            CHECK_NVJPEG(nvjpegGetImageInfo(nvjpeg_handle_, item.data.data(), item.data.size(), &channels, &subsampling,
                                            widths, heights));

            // spdlog::info("Processing: " << current_names[i]);
            // spdlog::info("Image is " << channels << " channels.");
            // for (int c = 0; c < channels; c++) {
            //     spdlog::info("Channel #" << c << " size: " << widths[c] << " x " << heights[c]
            //              );
            // }
            // switch (subsampling) {
            //     case NVJPEG_CSS_444: spdlog::info("YUV 4:4:4 chroma subsampling"); break;
            //     case NVJPEG_CSS_440: spdlog::info("YUV 4:4:0 chroma subsampling"); break;
            //     case NVJPEG_CSS_422: spdlog::info("YUV 4:2:2 chroma subsampling"); break;
            //     case NVJPEG_CSS_420: spdlog::info("YUV 4:2:0 chroma subsampling"); break;
            //     case NVJPEG_CSS_411: spdlog::info("YUV 4:1:1 chroma subsampling"); break;
            //     case NVJPEG_CSS_410: spdlog::info("YUV 4:1:0 chroma subsampling"); break;
            //     case NVJPEG_CSS_GRAY: spdlog::info("Grayscale JPEG "); break;
            //     case NVJPEG_CSS_UNKNOWN: spdlog::info("Unknown chroma subsampling"); return false;
            // }

            img_width_.push_back(widths[0]);
            img_height_.push_back(heights[0]);

            if (subsampling == NVJPEG_CSS_UNKNOWN) {
                spdlog::info("Unknown chroma subsampling");
                return false;
            }

            int mul = 1;
            if (format_ == NVJPEG_OUTPUT_RGBI || format_ == NVJPEG_OUTPUT_BGRI) {
                channels = 1;
                mul = 3;
            } else if (format_ == NVJPEG_OUTPUT_RGB || format_ == NVJPEG_OUTPUT_BGR) {
                channels = 3;
                widths[1] = widths[2] = widths[0];
                heights[1] = heights[2] = heights[0];
            }

            for (int c = 0; c < channels; c++) {
                int aw = mul * widths[c];
                int ah = heights[c];
                int sz = aw * ah;
                iout.pitch[c] = aw;
                if (sz > isz.pitch[c]) {
                    if (iout.channel[c]) { CHECK_CUDA(cudaFree(iout.channel[c])); }
                    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&iout.channel[c]), sz));
                    isz.pitch[c] = sz;
                }
            }

            vec_iout_.emplace_back(std::move(iout));
        }

        return true;
    }

    bool process_nvjpeg_image(const std::vector<JpegImage> &jpeg_images, std::vector<cv::cuda::GpuMat> &vec_images) {
        std::vector<const unsigned char *> vec_decode_bitstreams;
        std::vector<size_t> vec_decode_bitstreams_size;
        std::vector<nvjpegImage_t> vec_decode_output;

        CHECK_CUDA(cudaStreamSynchronize(stream_));

        if (hw_decode_available_) {
            for (int i = 0; i < vec_iout_.size(); i++) {
                // extract bitstream meta data to figure out whether a bit-stream can be decoded
                nvjpegJpegStreamParseHeader(nvjpeg_handle_, jpeg_images[i].data.data(), jpeg_images[i].data.size(),
                                            jpeg_streams_[0]);
                int isSupported = -1;
                nvjpegDecodeBatchedSupported(nvjpeg_handle_, jpeg_streams_[0], &isSupported);

                if (isSupported == 0) {
                    vec_decode_bitstreams.push_back(jpeg_images[i].data.data());
                    vec_decode_bitstreams_size.push_back(jpeg_images[i].data.size());
                } else {
                    vec_decode_bitstreams.push_back(jpeg_images[i].data.data());
                    vec_decode_bitstreams_size.push_back(jpeg_images[i].data.size());
                }
            }
        } else {
            for (int i = 0; i < vec_iout_.size(); i++) {
                vec_decode_bitstreams.push_back(jpeg_images[i].data.data());
                vec_decode_bitstreams_size.push_back(jpeg_images[i].data.size());
            }
        }

        // cudaEvent_t startEvent = nullptr;
        // cudaEvent_t stopEvent = nullptr;
        // CHECK_CUDA(cudaStreamSynchronize(stream_));
        // CHECK_CUDA(cudaEventCreateWithFlags(&startEvent, cudaEventBlockingSync));
        // CHECK_CUDA(cudaEventCreateWithFlags(&stopEvent, cudaEventBlockingSync));
        // CHECK_CUDA(cudaEventRecord(startEvent, stream_));

        if (hw_decode_available_) {
            CHECK_NVJPEG(
                nvjpegDecodeBatchedInitialize(nvjpeg_handle_, nvjpeg_state_, vec_decode_bitstreams.size(), 1, format_));

            CHECK_NVJPEG(nvjpegDecodeBatched(nvjpeg_handle_, nvjpeg_state_, vec_decode_bitstreams.data(),
                                             vec_decode_bitstreams_size.data(), vec_iout_.data(), stream_));
        } else {
            CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state_, device_buffer_));
            int buffer_index = 0;
            CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params_, format_));
            for (int i = 0; i < vec_iout_.size(); i++) {
                CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle_, vec_decode_bitstreams[i],
                                                   vec_decode_bitstreams_size[i], 0, 0, jpeg_streams_[buffer_index]));

                CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state_, pinned_buffers_[buffer_index]));

                CHECK_NVJPEG(nvjpegDecodeJpegHost(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_,
                                                  nvjpeg_decode_params_, jpeg_streams_[buffer_index]));

                CHECK_CUDA(cudaStreamSynchronize(stream_));

                CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_,
                                                              jpeg_streams_[buffer_index], stream_));

                buffer_index = 1 - buffer_index;// switch pinned buffer in pipeline mode to avoid an extra sync

                CHECK_NVJPEG(nvjpegDecodeJpegDevice(nvjpeg_handle_, nvjpeg_decoder_, nvjpeg_decoupled_state_,
                                                    &vec_iout_[i], stream_));
            }
        }

        // CHECK_CUDA(cudaEventRecord(stopEvent, stream_));
        // CHECK_CUDA(cudaEventSynchronize(stopEvent));

        // float loopTime = 0;
        // CHECK_CUDA(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));

        // #ifdef DEBUG
        // spdlog::info("Total size: {}, decoding time: {} (s)", vec_iout_.size(),
        //              0.001 * static_cast<double>(loopTime));
        // #endif

        // for (size_t i = 0; i < vec_iout_.size(); i++) {
        //     std::vector<cv::Mat> vchan_bgr;
        // vchan_bgr.emplace_back(cv::Mat(img_height_[i], img_width_[i], CV_8UC1));
        // vchan_bgr.emplace_back(cv::Mat(img_height_[i], img_width_[i], CV_8UC1));
        // vchan_bgr.emplace_back(cv::Mat(img_height_[i], img_width_[i], CV_8UC1));

        // CHECK_CUDA(cudaMemcpy2D(vchan_bgr[2].data, (size_t)img_width_[i],
        //                         vec_iout_[i].channel[0], (size_t)vec_iout_[i].pitch[0],
        //                         img_width_[i], img_height_[i], cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy2D(vchan_bgr[1].data, (size_t)img_width_[i],
        //                         vec_iout_[i].channel[1], (size_t)vec_iout_[i].pitch[1],
        //                         img_width_[i], img_height_[i], cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy2D(vchan_bgr[0].data, (size_t)img_width_[i],
        //                         vec_iout_[i].channel[2], (size_t)vec_iout_[i].pitch[2],
        //                         img_width_[i], img_height_[i], cudaMemcpyDeviceToHost));

        // cv::Mat mat_image;
        // cv::merge(vchan_bgr, mat_image);
        // mat_images.emplace_back(std::move(mat_image));
        // }

        for (size_t i = 0; i < vec_iout_.size(); i++) {
            vec_images.emplace_back(cv::cuda::GpuMat(img_height_[i], img_width_[i], CV_8UC3, vec_iout_[i].channel[0]));
        }

        // for (int i = 0; i < vec_iout_.size(); i++) {
        //     for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++)
        //         if (vec_iout_[i].channel[c]) CHECK_CUDA(cudaFree(vec_iout_[i].channel[c]));
        // }

        CHECK_CUDA(cudaStreamSynchronize(stream_));

        return true;
    }

private:
    size_t batch_size_;
    nvjpegOutputFormat_t format_;
    bool hw_decode_available_{true};
    nvjpegHandle_t nvjpeg_handle_;
    nvjpegJpegState_t nvjpeg_state_;
    cudaStream_t stream_;

    nvjpegJpegDecoder_t nvjpeg_decoder_;
    nvjpegJpegState_t nvjpeg_decoupled_state_;
    nvjpegDecodeParams_t nvjpeg_decode_params_;
    nvjpegBufferDevice_t device_buffer_;
    nvjpegJpegStream_t jpeg_streams_[2];
    nvjpegBufferPinned_t pinned_buffers_[2];

    std::vector<nvjpegImage_t> vec_iout_;
    std::vector<int> img_width_;
    std::vector<int> img_height_;

    std::vector<const unsigned char *> vec_decode_bitstreams;
    std::vector<size_t> vec_decode_bitstreams_size;
    std::vector<nvjpegImage_t> vec_decode_output;
};

NvJpegDecoder::NvJpegDecoder(const size_t instances, const size_t batch_size, const nvjpegOutputFormat_t format) {
    auto instance_max = instances > 3 ? 3 : instances;
    for (size_t i = 0; i < instance_max; i++) {
        vec_hanlde_.emplace_back(std::thread([batch_size, format, this, i]() {
            auto decoder = std::make_unique<NvJpegDecoderPrivate>(batch_size, format);

            gddi::thread_utils::set_cur_thread_name(fmt::format("NvDecInit"));
            decoder->init_decoder();
            gddi::thread_utils::set_cur_thread_name(fmt::format("NvDec-{}", i));

            while (active) {
                std::vector<JpegImage> jpeg_images;

                JpegImage image;
                if (!cache_jpeg_data_.wait_dequeue_timed(image, std::chrono::milliseconds(10))) { continue; }
                jpeg_images.emplace_back(std::move(image));

                // 尝试凑 BATCH
                for (int i = 1; i < batch_size; i++) {
                    if (cache_jpeg_data_.try_dequeue(image)) { jpeg_images.emplace_back(std::move(image)); }
                }

                if (!decoder->init_nvjpeg_image(jpeg_images)) {
                    // TODO: Failed
                    continue;
                }

                std::vector<cv::cuda::GpuMat> vec_images;
                if (decoder->process_nvjpeg_image(jpeg_images, vec_images)) {
                    for (size_t i = 0; i < vec_images.size(); i++) {
                        if (jpeg_images[i].callback) { jpeg_images[i].callback(jpeg_images[i].index, vec_images[i]); }
                    }
                } else {
                    // for (auto &item : jpeg_images) {
                    //     auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    //                          std::chrono::system_clock::now().time_since_epoch())
                    //                          .count();
                    //     std::ofstream file("/home/data/error/" + std::to_string(timestamp) + ".jpeg");
                    //     file.write((char *)item.data.data(), item.data.size());
                    //     file.close();
                    // }

                    vec_images.clear();
                    for (size_t i = 0; i < jpeg_images.size(); i++) {
                        vec_images.emplace_back(cv::cuda::GpuMat(100, 100, CV_8UC3));
                        if (jpeg_images[i].callback) { jpeg_images[i].callback(jpeg_images[i].index, vec_images[i]); }
                    }
                }
            }
        }));
    }
}

NvJpegDecoder::~NvJpegDecoder() {
    active = false;
    for (auto &item : vec_hanlde_) {
        if (item.joinable()) { item.join(); }
    }
}

void NvJpegDecoder::process_image(const JpegImage &image) { cache_jpeg_data_.enqueue(image); }

}// namespace wrapper
}// namespace gddi