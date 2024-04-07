#include <nvjpeg.h>
#include <iostream>
#include <cstring>

class NvJpegDecoder {
public:
    NvJpegDecoder() {
        nvjpegCreateSimple(&nv_handle_);
    }

    ~NvJpegDecoder() {
        nvjpegDestroySimple(nv_handle_);
    }

    bool decode(const char* filename, unsigned char*& output_data, int& width, int& height) {
        // 1. 读取JPEG文件
        FILE* fp = fopen(filename, "rb");
        if (!fp) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        fseek(fp, 0, SEEK_END);
        size_t file_size = ftell(fp);
        rewind(fp);
        char* jpeg_data = new char[file_size];
        fread(jpeg_data, 1, file_size, fp);
        fclose(fp);

        // 2. 分配和初始化nvjpeg解码器的输入和输出对象
        nvjpegJpegState_t nv_jpeg_state;
        nvjpegJpegStream_t nv_jpeg_stream;
        nvjpegDecodeParams_t nv_decode_params;
        nvjpegBufferPinned_t nv_input_pinned_buf;
        nvjpegBuffer_t nv_output_buf;
        nvjpegJpegStateCreate(nv_handle_, &nv_jpeg_state, NULL);
        nvjpegJpegStreamCreate(nv_handle_, &nv_jpeg_stream, jpeg_data, file_size, NULL);
        nvjpegDecodeParamsCreate(nv_handle_, &nv_decode_params, NVJPEG_OUTPUT_RGB);
        nvjpegBufferPinnedCreate(nv_handle_, &nv_input_pinned_buf, file_size);
        nvjpegBufferCreate(nv_handle_, &nv_output_buf, NULL);

        // 3. 处理nvjpeg解码器的输入对象
        nvjpegBufferPinnedSetCPU(nv_handle_, nv_input_pinned_buf, jpeg_data, file_size);
        nvjpegStatus_t nvjpeg_status;
        nvjpeg_status = nvjpegDecodeBatchedInitialize(nv_handle_, nv_jpeg_state, nv_jpeg_stream, 1, &nv_decode_params, 1);
        if (nvjpeg_status != NVJPEG_STATUS_SUCCESS) {
            std::cerr << "Failed to initialize nvjpeg decoder" << std::endl; 
            return false; 
        } 
        nvjpeg_status = nvjpegDecodeBatched(nv_handle_, nv_jpeg_state, nv_jpeg_stream, 1, &nv_input_pinned_buf, &nv_output_buf, NULL); 
        if (nvjpeg_status != NVJPEG_STATUS_SUCCESS) { 
            std::cerr << "Failed to decode jpeg image" << std::endl; return false; 
            }

        // 4. 处理nvjpeg解码器的输出对象
        width = nv_output_buf->pitch / 3;
        height = nv_output_buf->height;
        size_t output_size = width * height * 3;
        output_data = new unsigned char[output_size];
        memcpy(output_data, nv_output_buf->mem, output_size);

        // 5. 释放内存
        delete[] jpeg_data;
        nvjpegJpegStateDestroy(nv_jpeg_state);
        nvjpegJpegStreamDestroy(nv_jpeg_stream);
        nvjpegDecodeParamsDestroy(nv_decode_params);
        nvjpegBufferPinnedDestroy(nv_input_pinned_buf);
        nvjpegBufferDestroy(nv_output_buf);

        return true;
    }
}

class NvJpegEncoder {
public:
    NvJpegEncoder() {
        nvjpegCreateSimple(&nv_handle_);
    }

    ~NvJpegEncoder() {
        nvjpegDestroySimple(nv_handle_);
    }

    bool encode(const unsigned char* input_data, int width, int height, int quality, const char* filename) {
        // 1. 分配和初始化nvjpeg编码器的输入和输出对象
        nvjpegEncoderState_t nv_encoder_state;
        nvjpegEncoderParams_t nv_encoder_params;
        nvjpegBufferPinned_t nv_input_pinned_buf;
        nvjpegBuffer_t nv_output_buf;
        nvjpegEncoderStateCreate(nv_handle_, &nv_encoder_state, NULL);
        nvjpegEncoderParamsCreate(nv_handle_, &nv_encoder_params, NVJPEG_OUTPUT_JPEG);
        nvjpegBufferPinnedCreate(nv_handle_, &nv_input_pinned_buf, width * height * 3);
        nvjpegBufferCreate(nv_handle_, &nv_output_buf, NULL);

        // 2. 处理nvjpeg编码器的输入对象
        nvjpegBufferPinnedSetCPU(nv_handle_, nv_input_pinned_buf, input_data, width * height * 3);
        nvjpegStatus_t nvjpeg_status;
        nvjpeg_status = nvjpegEncodeImage(nv_handle_, nv_encoder_state, nv_encoder_params, &nv_input_pinned_buf, NVJPEG_INPUT_RGB,
                                          width, height, quality, &nv_output_buf, NULL);
        if (nvjpeg_status != NVJPEG_STATUS_SUCCESS) {
            std::cerr << "Failed to encode image" << std::endl;
            return false;
        }

        // 3. 保存JPEG文件
        FILE* fp = fopen(filename, "wb");
        if (!fp) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }
        fwrite(nv_output_buf->mem, 1, nv_output_buf->pitch, fp);
        fclose(fp);

        // 4. 释放内存
        nvjpegEncoderStateDestroy(nv_encoder_state);
        nvjpegEncoderParamsDestroy(nv_encoder_params);
        nvjpegBufferPinnedDestroy(nv_input_pinned_buf);
        nvjpegBufferDestroy(nv_output_buf);

        return true;
    }

private:
     nvjpegHandle_t nv_handle_; 
};


bool resize(const unsigned char* input_data, int input_width, int input_height, int output_width, int output_height, unsigned char*& output_data) {
    NppiSize src_size = {input_width, input_height};
    NppiSize dst_size = {output_width, output_height};
    int src_pitch = input_width * 3;
    int dst_pitch = output_width * 3;
    NppiRect src_roi = {0, 0, input_width, input_height};
    NppiRect dst_roi = {0, 0, output_width, output_height};

    // 分配输入和输出图像的内存
    unsigned char* d_src;
    cudaMalloc((void**)&d_src, src_pitch * input_height);
    cudaMemcpy(d_src, input_data, src_pitch * input_height, cudaMemcpyHostToDevice);
    unsigned char* d_dst;
    cudaMalloc((void**)&d_dst, dst_pitch * output_height);
    cudaMemset(d_dst, 0, dst_pitch * output_height);

    // 调用npp库的resize函数实现图像的resize
    NppStatus npp_status = nppiResize_8u_C3R(d_src, src_pitch, src_size, src_roi, d_dst, dst_pitch, dst_size, dst_roi, NPPI_INTER_LINEAR);
    if (npp_status != NPP_SUCCESS) {
        std::cerr << "Failed to resize image" << std::endl;
        return false;
    }

    // 从设备端将输出图像数据拷贝回主机端
    output_data = new unsigned char[dst_pitch * output_height];
    cudaMemcpy(output_data, d_dst, dst_pitch * output_height, cudaMemcpyDeviceToHost);

    // 释放内存
    cudaFree(d_src);
    cudaFree(d_dst);

    return true;
}
};

int main() { 
    NvJpegDecoder decoder; 
    ImageResizer resizer; 
    unsigned char* input_data; 
    // RGB图像数据 
    int input_width, input_height;
     // 图像宽度和高度 // 读取RGB图像数据和图像宽度、高度 // ... unsigned char* output_data; int output_width = 640, output_height = 480; 
     // resize后的图像宽度和高度 
     if (decoder.decode("test.jpg", input_data, input_width, input_height)) { 
        if (resizer.resize(input_data, input_width, input_height, output_width, output_height, output_data)) { 
            // 处理resize后的RGB图像数据 // ... 
            delete[] output_data; 
            } 
            delete[] 
            input_data; } return 0; }

