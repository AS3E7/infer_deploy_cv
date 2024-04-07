#include "common/type_convert.h"

#include "core/mem/buf_surface.h"

extern "C" {  
#include "libavutil/avutil.h"  
#include "libavcodec/avcodec.h"  
#include "libavformat/avformat.h"  
#include "libswscale/swscale.h"  
#include "libavutil/imgutils.h" 
} 
#include <opencv2/core.hpp>  

using namespace gddeploy;


//-------------------------opencv--------------------------------------
int convertMat2BufSurface(cv::Mat &img, gddeploy::BufSurfWrapperPtr &surf, bool is_copy)
{
    // BufSurfaceParams *src_param = new BufSurfaceParams();
    // memset(src_param, 0, sizeof(BufSurfaceParams));

    // src_param->data_size = img.cols * img.rows * 3;
    // src_param->color_format = GDDEPLOY_BUF_COLOR_FORMAT_BGR;

    // src_param->width = img.cols;
    // src_param->height = img.rows;
    // src_param->plane_params.width[0] = img.cols;
    // src_param->plane_params.height[0] = img.rows;
    // src_param->plane_params.num_planes = 3;
    // src_param->plane_params.offset[0] = 0;
    // src_param->plane_params.offset[1] = img.cols * img.rows;
    // src_param->plane_params.offset[2] = img.cols * img.rows * 2;
    // src_param->plane_params.pitch[0] = img.cols;
    // src_param->plane_params.pitch[1] = img.cols;

    // if (is_copy){
    //     int size = img.cols * img.rows * 3;
    //     src_param->data_ptr = malloc(size);
    //     memcpy(src_param->data_ptr, img.data, size);
    // }else{
    //     src_param->data_ptr = reinterpret_cast<void *>(img.data);
    // }

    // BufSurface *surface = new BufSurface();
    // surface->batch_size = 1;
    // surface->num_filled = 1;
    // surface->device_id = 0;
    // surface->mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
    // surface->surface_list = src_param;

    // surf = std::make_shared<gddeploy::BufSurfaceWrapper>(surface, true);


    BufSurface *surface;
    BufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(create_params));
    create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = 1;
    create_params.size = img.cols * img.rows * 3;
    create_params.width = img.cols;
    create_params.height = img.rows;
    create_params.color_format = GDDEPLOY_BUF_COLOR_FORMAT_BGR;

    BufSurfaceCreate(&surface, &create_params);

    surf = std::make_shared<gddeploy::BufSurfaceWrapper>(surface, true);

    // BufSurface *surface = new BufSurface();
    // surface->mem_type = GDDEPLOY_BUF_MEM_BMNN;
    // surface->batch_size = 1;
    // surface->num_filled = 1;

    // BufSurfacePlaneParams plane_param;
    // plane_param.num_planes = 1;
    // plane_param.width[0] = img.cols;
    // plane_param.height[0] = img.rows;

    // BufSurfaceParams *param = new BufSurfaceParams();
    // param->plane_params = plane_param;
    // param->color_format = GDDEPLOY_BUF_COLOR_FORMAT_BGR;
    // param->data_size = img.cols * img.rows * 3 ;
    // param->width = img.cols;
    // param->height = img.rows;
    // param->data_ptr = nullptr;
    
    // surface->surface_list = param;
    // surf = std::make_shared<gddeploy::BufSurfaceWrapper>(surface, true);


    auto src_param = surf->GetSurfaceParams();
    if (is_copy){
        int size = img.cols * img.rows * 3;
        memcpy(src_param->data_ptr, img.data, size);
    }else{
        src_param->data_ptr = reinterpret_cast<void *>(img.data);
    }
    src_param->_reserved[0] = &img;

    return 0;
}

int convertBufSurface2Mat(cv::Mat &img, gddeploy::BufSurfWrapperPtr &surf, bool is_copy)
{
    auto surface = surf->GetBufSurface();

    auto src_param = surf->GetSurfaceParams();
    int img_h = src_param->height;
    int img_w = src_param->width;
    // memcpy((uint8_t *)img_mat.data(), in_mlu.buffers[0].Data(), in_mlu.buffers[0].MemorySize());
    if (is_copy){
        if (src_param->color_format == GDDEPLOY_BUF_COLOR_FORMAT_RGB){
            img = cv::Mat(img_h, img_w, CV_8UC3, surf->GetHostData(0, 0));
        } else if (src_param->color_format == GDDEPLOY_BUF_COLOR_FORMAT_BGR){
            img = cv::Mat(img_h, img_w, CV_8UC3, surf->GetHostData(0, 0));
        } else if (src_param->color_format == GDDEPLOY_BUF_COLOR_FORMAT_YUV420 || src_param->color_format == GDDEPLOY_BUF_COLOR_FORMAT_NV12){
            // 讲surf->GetHostData(0, 0)中的yuv420数据转为rgb格式的Mat变量
            // cv::Mat yuv_mat = cv::Mat(img_h, img_w, CV_8UC3, (void *)surf->GetHostData(0, 0));
            // cv::cvtColor(yuv_mat, img, cv::COLOR_YUV2BGR_NV12);
            // cv::Mat cv_yuv(img_h + img_h/2, img_w, CV_8UC1, surf->GetHostData(0, 0));//pFrame为YUV数据地址,另外这里就是用 CV_8UC1非 CV_8UC3.
            // img = cv::Mat(img_h, img_w, CV_8UC3); 
            // cv::cvtColor(cv_yuv, img, cv::COLOR_YUV2BGR_I420);

            int width = src_param->width;
            int height = src_param->height;

            auto dst_frame = av_frame_alloc();
            dst_frame->format = AV_PIX_FMT_BGR24;
            dst_frame->width = width;
            dst_frame->height = height;
            int buf_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, width, height, 1);
            if (buf_size < 0) { throw std::runtime_error("fail to calc frame buffer size"); }
            av_frame_get_buffer(dst_frame, 1);

            int linesize[8] = { 0 };
            linesize[0] = width;
            linesize[1] = width;
            uint8_t *src_data[6] = { 0 };
            src_data[0] = (uint8_t *)surf->GetHostData(0, 0);
            src_data[1] = src_data[0] + width * height;

            SwsContext *conversion = sws_getContext(
                width, height, (AVPixelFormat)AV_PIX_FMT_NV12, width, height,
                AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
            sws_scale(conversion, (const uint8_t * const *)src_data, linesize, 0, height, dst_frame->data,
                    dst_frame->linesize);
            sws_freeContext(conversion);
            img = cv::Mat(height, width, CV_8UC3, dst_frame->data[0], dst_frame->linesize[0]).clone();
            av_frame_free(&dst_frame);
        }
    }else{
        if (src_param->color_format == GDDEPLOY_BUF_COLOR_FORMAT_RGB){
            img = cv::Mat(img_h, img_w, CV_8UC3, surf->GetData(0, 0));
        } else if (src_param->color_format == GDDEPLOY_BUF_COLOR_FORMAT_BGR){
            img = cv::Mat(img_h, img_w, CV_8UC3, surf->GetData(0, 0));
        } else if (src_param->color_format == GDDEPLOY_BUF_COLOR_FORMAT_YUV420){
            // img.
        }
    }

    return 0;
}

//-------------------------avframe--------------------------------------
int AVFrame_GetSize(AVFrame *in)
{
    int data_size = -1;
    if (in->format == AV_PIX_FMT_YUV420P)
        data_size = in->height * in->width * 3 / 2;
    if (in->format == AV_PIX_FMT_RGB24)
        data_size = in->height * in->width * 3;
    return data_size;
}

BufSurfaceColorFormat AVFrame_ChangeColor(AVFrame *in)
{
    if (in->format == AV_PIX_FMT_YUV420P)
        return GDDEPLOY_BUF_COLOR_FORMAT_YUV420;
    if (in->format == AV_PIX_FMT_RGB24)
        return GDDEPLOY_BUF_COLOR_FORMAT_RGB;
    
    return GDDEPLOY_BUF_COLOR_FORMAT_INVALID;
}

int convertAVFrame2BufSurface(AVFrame *in, gddeploy::BufSurfWrapperPtr surf, bool is_copy)
{
    BufSurface *surface;
    BufSurfaceCreateParams create_params;
    memset(&create_params, 0, sizeof(create_params));
    create_params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
    create_params.force_align_1 = 1;  // to meet mm's requirement
    create_params.device_id = 0;
    create_params.batch_size = 1;
    create_params.size = AVFrame_GetSize(in);
    create_params.width = in->width;
    create_params.height = in->height;
    create_params.color_format = AVFrame_ChangeColor(in);

    BufSurfaceCreate(&surface, &create_params);

    surf = std::make_shared<gddeploy::BufSurfaceWrapper>(surface, true);

    //TODO: 修改为根据设备类型调用不同的拷贝函数
    BufSurfaceParams *src_param = &surface->surface_list[0];
    if (is_copy){
        int size = in->width * in->height * 3 / 2;
        memcpy(src_param->data_ptr, in->data, size);
    }else{
        src_param->data_ptr = reinterpret_cast<void *>(in->data);
    }

    return 0;
}


//AVFrame 转 cv::mat  
cv::Mat avframeToCvmat(const AVFrame * frame)  
{  
    int width = frame->width;  
    int height = frame->height;  
    cv::Mat image(height, width, CV_8UC3);  
    int cvLinesizes[1];  
    cvLinesizes[0] = image.step1();  
    // SwsContext* conversion = sws_getContext(width, height, (AVPixelFormat) frame->format, width, height, AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);  
    // sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cvLinesizes);  
    // sws_freeContext(conversion);  
    return image;  
}  


//cv::Mat 转 AVFrame  
AVFrame* cvmatToAvframe(cv::Mat* image, AVFrame * frame)
{  
    // int width = image->cols;  
    // int height = image->rows;  
    // int cvLinesizes[1];  
    // cvLinesizes[0] = image->step1();  
    // if (frame == NULL){  
    //     frame = av_frame_alloc();  
    //     av_image_alloc(frame->data, frame->linesize, width, height, AVPixelFormat::AV_PIX_FMT_YUV420P, 1);  
    // }  
    // SwsContext* conversion = sws_getContext(width, height, AVPixelFormat::AV_PIX_FMT_BGR24, width, height, (AVPixelFormat) frame->format, SWS_FAST_BILINEAR, NULL, NULL, NULL);  
    // sws_scale(conversion, &image->data, cvLinesizes , 0, height, frame->data, frame->linesize);  
    // sws_freeContext(conversion);  
    return  frame;  
}

#if WITH_BM1684
#define USE_OPENCV 1
#define USE_FFMPEG 1
#include "bmruntime_interface.h"
// #include "bm_wrapper.hpp"
#include "bmcv_api_ext.h"

int convertBmImage2BufSurface(bm_image &img, gddeploy::BufSurfWrapperPtr &surf, bool is_copy)
{
    BufSurface *surface = new BufSurface();
    surface->mem_type = GDDEPLOY_BUF_MEM_BMNN;
    surface->batch_size = 1;
    surface->num_filled = 1;

    BufSurfacePlaneParams plane_param;
    memset(&plane_param, 0, sizeof(BufSurfacePlaneParams));
    plane_param.num_planes = bm_image_get_plane_num(img);
    plane_param.width[0] = img.width;
    plane_param.height[0] = img.height;

    BufSurfaceParams *param = new BufSurfaceParams();
    memset(param, 0, sizeof(BufSurfaceParams));
    param->plane_params = plane_param;
    param->color_format = GDDEPLOY_BUF_COLOR_FORMAT_BGR;
    param->data_size = img.width * img.height * 3 ;
    param->width = img.width;
    param->height = img.height;
    param->data_ptr = nullptr;
    
    surface->surface_list = param;
    surf = std::make_shared<gddeploy::BufSurfaceWrapper>(surface, false);

    auto src_param = surf->GetSurfaceParams();
    if (is_copy){
        int size = img.width * img.height * 3;
        // memcpy(src_param->data_ptr, input_mat.data, size);
    }else{
        src_param->data_ptr = reinterpret_cast<void *>(img.image_private);       
        for (uint32_t j = 0; j < plane_param.num_planes; j++) {
            bm_device_mem_t pmem;
            bm_image_get_device_mem(img, &pmem);
            unsigned long long dev_mem_base_ptr = bm_mem_get_device_addr(pmem);
            plane_param.data_ptr[j] = (void *)dev_mem_base_ptr;
            plane_param.psize[j] = pmem.size;
        }
        bm_device_mem_t pmem;
        bm_image_get_device_mem(img, &pmem);
        unsigned long long dev_mem_base_ptr = bm_mem_get_device_addr(pmem);
        param->data_ptr = (void *)(dev_mem_base_ptr);
        param->data_size = pmem.size;
        param->plane_params = plane_param;
    }
    src_param->_reserved[0] = img.image_private;

    return 0;
}

int convertBufSurface2BmImage(bm_image &img, gddeploy::BufSurfWrapperPtr &surf)
{
    auto src_param = surf->GetSurfaceParams();

    img.image_format = FORMAT_BGR_PACKED;
    img.image_private = (bm_image_private*)src_param->data_ptr;
    img.data_type = DATA_TYPE_EXT_1N_BYTE;
    img.height = src_param->height;
    img.width = src_param->width;

    return 0;
}

#endif
