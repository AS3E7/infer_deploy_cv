#if WITH_NVIDIA
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <memory.h>
#include <cuda_runtime_api.h>
#include "nvjpeg.h"


NvjpegImageProcessor::NvjpegImageProcessor() {
    // Initialize nvJPEG library
    nvjpegStatus_t status = nvjpegCreateSimple(&nvjpeg_handle_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to initialize nvJPEG");
    }

    cudaStreamCreate(&stream_);
}

NvjpegImageProcessor::~NvjpegImageProcessor() {
    if (nvjpeg_handle_ != nullptr) {
        cudaStreamDestroy(stream_);
        nvjpegDestroy(nvjpeg_handle_);
    }
}
  
int NvjpegImageProcessor::Decode(std::vector<std::string> &filenames, BufSurface *surf) {
    nvjpegImage_t image;
    nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_444;

    nvjpegJpegState_t jpeg_state;
    nvjpegJpegStateCreate(nvjpeg_handle_, &jpeg_state);

    uint32_t pic_num = filenames.size();
    surf->surface_list = new BufSurfaceParams[pic_num];
    memset(surf->surface_list, 0, sizeof(BufSurfaceParams) * pic_num);

    surf->mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
    surf->device_id = 0; 
    surf->batch_size = pic_num;
    surf->num_filled = 1;
    surf->is_contiguous = 1;

    int pic_idx = 0;
    for (auto &filename : filenames){
        memset(&image, 0, sizeof(nvjpegImage_t));

        // Read JPEG image from file
        std::ifstream input_file(filename, std::ios::binary);
        if (!input_file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
            return -1;
        }

        // Get file size
        input_file.seekg(0, std::ios::end);
        std::streampos file_size = input_file.tellg();
        input_file.seekg(0, std::ios::beg);

        // Allocate buffer for file data
        std::vector<char> file_data(file_size);
        if (!input_file.read(file_data.data(), file_size)) {
            throw std::runtime_error("Failed to read file data: " + filename);
            return -1;
        }

        // 创建surface
        BufSurfaceParams *surf_param = &surf->surface_list[pic_idx++];
        BufSurfacePlaneParams *surf_plane_param = &surf_param->plane_params;

        // Get image info,  这里要注意nvjpegGetImageInfo是按照yuv420存储，只需要取出0维度即可；
        int width[NVJPEG_MAX_COMPONENT];
        int height[NVJPEG_MAX_COMPONENT];
        int components;
        nvjpegStatus_t status = nvjpegGetImageInfo(nvjpeg_handle_, (const unsigned char *)file_data.data(), file_data.size(), 
                    &components, &subsampling, width, height);
        if (status != NVJPEG_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to get image info");
        }

        //填充必要数据
        surf_plane_param->num_planes = 3;
        for (int i = 0; i < GDDEPLOY_BUF_MAX_PLANES; i++) {
            surf_plane_param->width[i] = width[0];
            surf_plane_param->height[i] = height[0];
            surf_plane_param->bytes_per_pix[i] = 1;
            surf_plane_param->pitch[i] = surf_plane_param->width[0];
            surf_plane_param->psize[i] = surf_plane_param->width[0] * surf_plane_param->height[0];
        }
        for (int i = 1; i < GDDEPLOY_BUF_MAX_PLANES; i++) {
            surf_plane_param->offset[i] = surf_plane_param->offset[i-1] + surf_plane_param->psize[i-1];
        }

        surf_param->width = surf_plane_param->width[0];
        surf_param->height = surf_plane_param->height[0];
        surf_param->color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER;  //
        surf_param->data_ptr = nullptr;
        surf_param->data_size = surf_param->width * surf_param->height * 3;
        surf_param->pitch = surf_param->width;

        cudaError_t cudaStatus = cudaMalloc(&surf_param->data_ptr, surf_param->data_size);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory");
            return -1;
        }

        // Decode image
        for (int i = 0; i < GDDEPLOY_BUF_MAX_PLANES; i++) {
            image.channel[i] = (unsigned char *)(surf_param->data_ptr + surf_plane_param->offset[i]);
            image.pitch[i] = surf_plane_param->width[i];
        }

        nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_RGB;
        status = nvjpegDecode(nvjpeg_handle_, jpeg_state, (const unsigned char *)file_data.data(), file_data.size(), 
                        output_format, &image, stream_);
        if (status != NVJPEG_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to decode image");
        }
    }
    cudaStreamSynchronize(stream_);

    nvjpegJpegStateDestroy(jpeg_state);
    return 0;
}

int NvjpegImageProcessor::Encode(std::vector<std::string> &filename, BufSurface *surf, int quality) {
     //  encode test!!!!!!!!
    nvjpegEncoderState_t encoder_state;
    nvjpegEncoderParams_t encoder_params;

    nvjpegEncoderStateCreate(nvjpeg_handle_, &encoder_state, NULL);
    nvjpegEncoderParamsCreate(nvjpeg_handle_, &encoder_params, NULL);

    // set params
	nvjpegEncoderParamsSetEncoding(encoder_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, NULL);
	nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, 1, NULL);
	nvjpegEncoderParamsSetQuality(encoder_params, quality, NULL);
	nvjpegEncoderParamsSetSamplingFactors(encoder_params, nvjpegChromaSubsampling_t::NVJPEG_CSS_444, NULL);

    for (int i = 0; i < surf->batch_size; i++){
        BufSurfaceParams *surf_param = &surf->surface_list[i];
        BufSurfacePlaneParams *surf_plane_param = &surf_param->plane_params;

        nvjpegImage_t input;
        for (int i = 0; i < 3; i++){
            input.pitch[i] = surf_plane_param->width[i];
            input.channel[i] = (unsigned char *)(surf_param->data_ptr + surf_plane_param->offset[i]);
        }
        nvjpegStatus_t status = nvjpegEncodeImage(nvjpeg_handle_, encoder_state, encoder_params, &input, 
                NVJPEG_INPUT_RGB, surf_param->width,  surf_param->height, nullptr);
        if (status != NVJPEG_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to decode image");
            return -1;
        }

        // 取出压缩后的数据
        std::vector<unsigned char> obuffer;
        size_t length;
        nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, encoder_state, NULL, &length, NULL);
    
        obuffer.resize(length);
        nvjpegEncodeRetrieveBitstream(nvjpeg_handle_, encoder_state, obuffer.data(), &length, NULL);
    
        std::ofstream outputFile(filename[i], std::ios::out | std::ios::binary);
        outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
    }
    
 
    nvjpegEncoderParamsDestroy(encoder_params);
    nvjpegEncoderStateDestroy(encoder_state);

    return 0;
}

#endif