#include "app/runner_video.h"
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "common/logger.h"
#include "core/mem/buf_surface_util.h"
#include "core/mem/buf_surface_impl.h"
#include "common/data_queue.h"
#include "api/infer_api.h"
#include "api/global_config.h"
#include "common/type_convert.h"
#include "app/result_handle.h"
#include "video/video_decode_ffmpeg.h"


namespace gddeploy {

class VideoRunnerPriv;
class VideoRunnerPriv{
public:
    VideoRunnerPriv()=default;
    ~VideoRunnerPriv(){
        is_exit_ = true;
        decoder_ = nullptr;
        condition_.notify_all();
        Join();
    }

    int Init(const std::string config, std::string model_path, std::string license_path);
    int Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths);

    int OpenStream(std::string video_path, std::string save_path="", bool is_draw=false);  
    int Start();
    int Join();

    int OpencvOpen(std::string video_path, std::string save_path="", bool is_draw=false);

private:
    InferAPI infer_api_;
    std::vector<InferAPI> infer_api_v_;

    std::thread thread_read_frame_;
    std::thread thread_handle_;
    bool is_exit_ = false;
    std::mutex mutex_;                 //互斥锁
    std::condition_variable condition_;//条件变

    std::string model_path_;
    std::string save_path_;
    std::string input_url_;
    std::shared_ptr<Decoder> decoder_;
    SafeQueue<std::shared_ptr<AVFrame>> input_queue_;
    SafeQueue<BufSurfWrapperPtr> pool_surf_;

    BufPool *pool_;
    // std::vector<gddeploy::BufSurfWrapperPtr> pool_surf_;
};

} // namespace gddeploy

using namespace gddeploy;

void asyncCallback(gddeploy::Status status, gddeploy::PackagePtr data, gddeploy::any user_data){
    if (data->data.size() < 1)
        return;
    if (data->data[0] == nullptr)
        return ;
    gddeploy::InferResult result = data->data[0]->GetMetaData<gddeploy::InferResult>(); 
    PrintResult(result);
}


int VideoRunnerPriv::Init(const std::string config, std::string model_path, std::string license_path)
{
    gddeploy_init("");
    
    infer_api_.Init(config, model_path, license_path, ENUM_API_SESSION_API);

    return 0;
}

int VideoRunnerPriv::Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths)
{
    gddeploy_init("");
    for (int i = 0; i < model_paths.size(); i++){
        std::string model_path = model_paths[i];
        std::string license_path = license_paths[i];

        InferAPI infer_api;
        infer_api.Init(config, model_path, license_path, ENUM_API_SESSION_API);

        infer_api_v_.push_back(infer_api);
    }
    return 0;
}

int VideoRunnerPriv::Join()
{
    if (thread_read_frame_.joinable()) { thread_read_frame_.join(); }
    if (thread_handle_.joinable()) { thread_handle_.join(); }
    return 0;
}

int VideoRunnerPriv::Start()
{
    
        
    return 0;
}

static BufSurfaceColorFormat convertFormat(int format)
{
    if (format == AV_PIX_FMT_YUV420P)
        return GDDEPLOY_BUF_COLOR_FORMAT_YUV420;
    if( format == AV_PIX_FMT_YUVJ420P)
        return GDDEPLOY_BUF_COLOR_FORMAT_YUVJ420P;
    if (format == AV_PIX_FMT_NV12)
        return GDDEPLOY_BUF_COLOR_FORMAT_NV12;
    if (format == AV_PIX_FMT_NV21) 
        return GDDEPLOY_BUF_COLOR_FORMAT_NV21;
    if (format == AV_PIX_FMT_RGB24)
        return GDDEPLOY_BUF_COLOR_FORMAT_RGB;
}

static int AVFrame_GetSize(AVFrame *in)
{
    int data_size = -1;
    if (in->format == AV_PIX_FMT_YUV420P || 
        in->format == AV_PIX_FMT_NV12 || 
        in->format == AV_PIX_FMT_NV21 || 
        in->format == AV_PIX_FMT_YUVJ420P)
        data_size = in->height * in->width * 3 / 2;

    if (in->format == AV_PIX_FMT_RGB24 || in->format == AV_PIX_FMT_BGR24)
        data_size = in->height * in->width * 3;

    return data_size;
}

static BufSurfaceColorFormat AVFrame_ChangeColor(AVFrame *in)
{
    if (in->format == AV_PIX_FMT_YUV420P)
        return GDDEPLOY_BUF_COLOR_FORMAT_YUV420;
    if (in->format == AV_PIX_FMT_RGB24)
        return GDDEPLOY_BUF_COLOR_FORMAT_RGB;
    
    return GDDEPLOY_BUF_COLOR_FORMAT_INVALID;
}
#if WITH_TS
#include "ts_comm_vdec.h"
#endif
static int convertAVFrame2BufSurface1(AVFrame *in, gddeploy::BufSurfWrapperPtr &surf, bool is_copy)
{
    BufSurface *dst_surf = surf->GetBufSurface();

    //TODO: 修改为根据设备类型调用不同的拷贝函数
    if (is_copy){
        BufSurface src_surf;
#if WITH_BM1684
        src_surf.mem_type = GDDEPLOY_BUF_MEM_BMNN;
#elif WITH_NVIDIA
        src_surf.mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
#elif WITH_TS
        src_surf.mem_type = GDDEPLOY_BUF_MEM_TS;
#else
        src_surf.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
#endif
        src_surf.batch_size = 1;
        src_surf.num_filled = 1;
        src_surf.is_contiguous = 0;    // AVFrame的两个plane地址不一定连续

        BufSurfaceParams param;

        BufSurfacePlaneParams plane_param;
        plane_param.width[0] = in->width;
        plane_param.height[0] = in->height;
        plane_param.bytes_per_pix[0] = 1;

        if (in->format == AV_PIX_FMT_YUV420P || in->format == AV_PIX_FMT_YUVJ420P){
            plane_param.num_planes = 3; 
#if WITH_BM1684
            plane_param.psize[0] = in->linesize[4] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[4];
            plane_param.psize[1] = in->linesize[5] * in->height / 2;
            plane_param.data_ptr[1] = (void *)in->data[5];
            plane_param.psize[2] = in->linesize[6] * in->height / 2;
            plane_param.data_ptr[2] = (void *)in->data[6];
#else
            plane_param.psize[0] = in->linesize[0] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[0];
            plane_param.psize[1] = in->linesize[1] * in->height / 2;
            plane_param.data_ptr[1] = (void *)in->data[1];
            plane_param.psize[2] = in->linesize[2] * in->height / 2;
            plane_param.data_ptr[2] = (void *)in->data[2];
#endif
            plane_param.offset[0] = 0;
            plane_param.offset[1] = plane_param.psize[0];
            plane_param.offset[2] = plane_param.psize[0] + plane_param.psize[1];
        } else if (in->format == AV_PIX_FMT_NV12 || in->format == AV_PIX_FMT_NV21){
            plane_param.num_planes = 2; 
#if WITH_BM1684
            plane_param.psize[0] = in->linesize[4] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[4];
            plane_param.psize[1] = in->linesize[5] * in->height / 2;        
            plane_param.data_ptr[1] = (void *)in->data[5];
#else 
            plane_param.psize[0] = in->linesize[0] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[0];
            plane_param.psize[1] = in->linesize[1] * in->height / 2;        
            plane_param.data_ptr[1] = (void *)in->data[1];
#endif
            plane_param.offset[0] = 0;
            plane_param.offset[1] = plane_param.psize[0];
        }
        

        param.plane_params = plane_param;
#if WITH_TS
        VIDEO_FRAME_INFO_S *pstFrameInfo = (VIDEO_FRAME_INFO_S *)dst_surf->surface_list[0].data_ptr;
        // auto t0 = std::chrono::high_resolution_clock::now();
        auto size = av_image_get_buffer_size((AVPixelFormat)in->format, in->width, in->height, 32);
        auto ret = av_image_copy_to_buffer((uint8_t*)pstFrameInfo->stVFrame.u64VirAddr[0], size, (const uint8_t *const *)in->data,
                        (const int *)in->linesize, (AVPixelFormat)in->format, in->width, in->height, 1);
        // auto t1 = std::chrono::high_resolution_clock::now();
        // printf("!!!!!!!!!ffmpeg copy time: %d us\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        param.color_format = convertFormat(in->format);
        param.data_size = AVFrame_GetSize(in);
        param.width = in->width;
        param.height = in->height;
        
        src_surf.surface_list = &param;
#else
#if WITH_BM1684
        param.data_ptr = in->data[4];
#else
        param.data_ptr = in->data[0];
#endif
        param.color_format = convertFormat(in->format);
        param.data_size = AVFrame_GetSize(in);
        param.width = in->width;
        param.height = in->height;
        
        src_surf.surface_list = &param;

        BufSurfaceCopy(&src_surf, dst_surf);
#endif
    }else{
        // BufSurfaceParams *src_param = &dst_surf->surface_list[0];
        // src_param->data_ptr = reinterpret_cast<void *>(in->data[4]);

#if WITH_BM1684
        dst_surf->mem_type = GDDEPLOY_BUF_MEM_BMNN;
#elif WITH_NVIDIA
        dst_surf->mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
#else
        dst_surf->mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
#endif
        dst_surf->batch_size = 1;
        dst_surf->num_filled = 1;
        dst_surf->is_contiguous = 0;    // AVFrame的两个plane地址不一定连续

        BufSurfaceParams *param = new BufSurfaceParams;

        BufSurfacePlaneParams plane_param;
        plane_param.width[0] = in->width;
        plane_param.height[0] = in->height;
        plane_param.bytes_per_pix[0] = 1;

        if (in->format == AV_PIX_FMT_YUV420P || in->format == AV_PIX_FMT_YUVJ420P){
            plane_param.num_planes = 3; 
#if WITH_BM1684
            plane_param.psize[0] = in->linesize[0] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[4];
            plane_param.psize[1] = in->linesize[1] * in->height / 2;
            plane_param.data_ptr[1] = (void *)in->data[5];
            plane_param.psize[2] = in->linesize[2] * in->height / 2;
            plane_param.data_ptr[2] = (void *)in->data[6];
#else
            plane_param.psize[0] = in->linesize[0] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[0];
            plane_param.psize[1] = in->linesize[1] * in->height / 2;
            plane_param.data_ptr[1] = (void *)in->data[1];
            plane_param.psize[2] = in->linesize[2] * in->height / 2;
            plane_param.data_ptr[2] = (void *)in->data[2];
#endif
            plane_param.offset[0] = 0;
            plane_param.offset[1] = plane_param.psize[0];
            plane_param.offset[2] = plane_param.psize[0] + plane_param.psize[1];
        } else if (in->format == AV_PIX_FMT_NV12 || in->format == AV_PIX_FMT_NV21){
            plane_param.num_planes = 2; 
#if WITH_BM1684
            plane_param.psize[0] = in->linesize[4] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[4];
            plane_param.psize[1] = in->linesize[5] * in->height / 2;        
            plane_param.data_ptr[1] = (void *)in->data[5];
#elif WITH_TS

#else 
            plane_param.psize[0] = in->linesize[0] * in->height;        
            plane_param.data_ptr[0] = (void *)in->data[0];
            plane_param.psize[1] = in->linesize[1] * in->height / 2;        
            plane_param.data_ptr[1] = (void *)in->data[1];
#endif
            plane_param.offset[0] = 0;
            plane_param.offset[1] = plane_param.psize[0];
        }
        

        param->plane_params = plane_param;
#if WITH_BM1684
        param->data_ptr = in->data[4];
#elif WITH_TS
        param->data_ptr =  (VIDEO_FRAME_INFO_S *)in->buf[0]->data;
#else 
        param->data_ptr = in->data[0];
#endif
        param->color_format = convertFormat(in->format);
        param->data_size = AVFrame_GetSize(in);
        param->width = in->width;
        param->height = in->height;
        
        dst_surf->surface_list = param;
#if WITH_TS
        dst_surf->mem_type = GDDEPLOY_BUF_MEM_TS;
#endif
    }

    return 0;
}

static int g_stream_id = 0;
int VideoRunnerPriv::OpenStream(std::string video_path, std::string save_path, bool is_draw)
{
    int stream_id = g_stream_id++;
    save_path_ = save_path;
    // thread_handle_ = std::thread(InferVideoThread,  model_path_);  
    thread_handle_ = std::thread([&, this, stream_id, is_draw, save_path](){
        int frame_id = 0;
        while(1){
            std::unique_lock<std::mutex> lock(this->mutex_);
            this->condition_.wait(lock, [&] { return !this->input_queue_.empty() || this->is_exit_; });    //任务队列中有任务待处理 或者 线程需要退出时唤醒线程
            if (this->is_exit_){
                printf("kill thread\n");
                GDDEPLOY_INFO("[app] VideoRunner read frame thread exit.");
                break;
            }

            auto avframe = this->input_queue_.wait_for_data();    
            // BufSurfWrapperPtr surf_ptr = this->input_queue_.wait_for_data();    
            // 拷贝avframe到文件，avframe的格式的nv12
            // std::string save_path = "/root/gddeploy/preds/avframe.jpg";
            // SaveFrame(avframe.get(), save_path);

            BufSurfaceCreateParams params;
#if WITH_BM1684
            params.mem_type = GDDEPLOY_BUF_MEM_BMNN;
#elif WITH_NVIDIA
            // params.mem_type = GDDEPLOY_BUF_MEM_NVIDIA;
            params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
#elif WITH_TS
            params.mem_type = GDDEPLOY_BUF_MEM_TS;
#else 
            params.mem_type = GDDEPLOY_BUF_MEM_SYSTEM;
#endif
            params.device_id = 0;
            params.width = avframe->width;
            params.height = avframe->height;
            params.color_format = convertFormat(avframe->format);
            params.force_align_1 = 1;
            params.bytes_per_pix = 1;
            params.size = 0;
            // params.size = params.width * params.height * 3 / 2;  // 这里要注意不要赋值，内部会根据是否需要对齐调整大小
            params.batch_size = 1;
#if WITH_TS
            gddeploy::BufSurfWrapperPtr surf_ptr(new BufSurfaceWrapper(new BufSurface, true));
            BufSurface *surf = surf_ptr->GetBufSurface();
            memset(surf, 0, sizeof(BufSurface));
            CreateSurface(&params, surf);
            convertAVFrame2BufSurface1(avframe.get(), surf_ptr, true);
#else
            gddeploy::BufSurfWrapperPtr surf_ptr = nullptr;
            if (this->pool_surf_.size() < 1){
                std::shared_ptr<BufSurfaceWrapper> tmp(new BufSurfaceWrapper(new BufSurface, true));
                BufSurface *surf = tmp->GetBufSurface();
                memset(surf, 0, sizeof(BufSurface));
                CreateSurface(&params, surf);
                this->pool_surf_.push(tmp);
            }
            // } else {
                surf_ptr = this->pool_surf_.wait_for_data();
                BufSurface *surf = surf_ptr->GetBufSurface();
                // memset(surf, 0, sizeof(BufSurface));
            // }
            // gddeploy::BufSurfWrapperPtr surf_ptr(new BufSurfaceWrapper(new BufSurface, true));
            // // BufSurfWrapperPtr surf_ptr = pool_->GetBufSurfaceWrapper();
            // BufSurface *surf = surf_ptr->GetBufSurface();
            // memset(surf, 0, sizeof(BufSurface));
            // CreateSurface(&params, surf);
            
            const int max_pool_size = 2;
            // printf("!!!!!!!!!!!!!!!!#######Stream push %d frame, queue size:%ld\n", stream_id, this->pool_surf_.size());
            if (this->pool_surf_.size() > max_pool_size){
                // 删除5个以后的surf，保持最多5个
                int  delete_num = this->pool_surf_.size() - max_pool_size;
                for (int i = 0; i < delete_num; i++){
                    this->pool_surf_.pop();
                }
            }

            convertAVFrame2BufSurface1(avframe.get(), surf_ptr, true);
#endif
            // cv::Mat in_mat = avframeToCvmat(avframe.get());
            
            // gddeploy::BufSurfWrapperPtr surf_ptr;
            // convertMat2BufSurface(in_mat, surf, true);

            gddeploy::PackagePtr in = gddeploy::Package::Create(1);
            in->data[0]->Set(surf_ptr);

            gddeploy::PackagePtr out = gddeploy::Package::Create(1);
            if (infer_api_v_.size() > 0){
                for (auto &infer_api : infer_api_v_){
                    // infer_api.InferSync(in, out);
                    // if (false == out->data[0]->HasMetaValue())
                    //     break;
                    
                    // gddeploy::InferResult result = out->data[0]->GetMetaData<gddeploy::InferResult>();

                    // PrintResult(result);
                    // infer_api.InferAsync(in, asyncCallback);
                    infer_api.InferAsync(in, [this, stream_id](gddeploy::Status status, gddeploy::PackagePtr data, gddeploy::any user_data){
                        if (data->data.size() < 1)
                            return;
                        if (data->data[0] == nullptr)
                            return ;
                        gddeploy::InferResult result = data->data[0]->GetMetaData<gddeploy::InferResult>(); 
                        PrintResult(result);

                        gddeploy::BufSurfWrapperPtr surf_ptr = data->data[0]->GetLref<gddeploy::BufSurfWrapperPtr>();
                        // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@###########Stream push %d frame, queue size:%ld\n", stream_id, this->pool_surf_.size());
                        this->pool_surf_.push(surf_ptr);
                    });
                }
            } else {
                // infer_api_.InferSync(in, out);
                // if (false == out->data[0]->HasMetaValue())
                //     break;
                
                // gddeploy::InferResult result = out->data[0]->GetMetaData<gddeploy::InferResult>();

                // PrintResult(result);
                infer_api_.InferAsync(in, asyncCallback);
            }
            // infer_api_.InferAsync(in, asyncCallback);

            // gddeploy::InferResult result = any_cast<gddeploy::InferResult>(out->result); 
            // gddeploy::InferResult result = out->data[0]->GetLref<gddeploy::InferResult>(); 
            // PrintResult(result);

            // if (is_draw && save_path != ""){
            //     cv::Mat in_mat = avframeToCvmat(avframe.get());
            //     auto pic_paths = std::vector<std::string>{};
            //     auto surfs = std::vector<gddeploy::BufSurfWrapperPtr>{surf_ptr};
            //     auto in_mats = std::vector<cv::Mat>{in_mat};
            //     std::string save_path = this->save_path_ + "frame_" + std::to_string(frame_id)+".jpg";
            //     cv::imwrite(save_path, in_mat);
            //     // DrawBbox(result, pic_paths, in_mats, surfs, save_path);
            // }
            frame_id++;
        }
    });
    usleep(2000);
    // thread_read_frame_ = std::thread(DecodeVideoFFmpeg,  model_path_, stread_id_);
    thread_read_frame_ = std::thread([&, this, video_path](){
        int frame_num = 0;
        decoder_ = std::make_shared<Decoder>();   
        if (-1 == decoder_->open_stream(video_path, [this, &frame_num](const int64_t frame_idx, const std::shared_ptr<AVFrame> &avframe) {
                frame_num++;
                // if (frame_num % 6 == 1 || frame_num % 6 == 2 || frame_num % 6 == 4 || frame_num % 6 == 5)
                //     return ;
                // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@###########Stream push %d frame, queue size:%ld\n", frame_num, this->input_queue_.size());
                // BufSurfWrapperPtr surf_ptr = pool_->GetBufSurfaceWrapper();
                // convertAVFrame2BufSurface1(avframe.get(), surf_ptr, true);
                this->input_queue_.push(avframe);
                this->condition_.notify_one();
                // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@###########Stream push %d frame, queue size:%ld\n", frame_num, this->input_queue_.size());
            //     return true;
            })){
                this->is_exit_ = true;
                this->condition_.notify_one();
            }

        // std::this_thread::sleep_for(std::chrono::seconds(6000));
    });

    return 0;
}

int VideoRunnerPriv::OpencvOpen(std::string video_path, std::string save_path, bool is_draw)
{
    save_path_ = save_path;

    cv::VideoCapture cap;
    cap.open(video_path.c_str());
    if (!cap.isOpened()) {
        GDDEPLOY_ERROR("Open {} fail", video_path);
        return -1;
    }

    int frame_id = 0;
    while(1){
        cv::Mat in_mat;
        bool ret = cap.read(in_mat);
        if (!ret){
            GDDEPLOY_ERROR("Read frame fail");
            break ;
        }

        gddeploy::BufSurfWrapperPtr surf;
        convertMat2BufSurface(in_mat, surf, true);

        gddeploy::PackagePtr in = gddeploy::Package::Create(1);
        in->data[0]->Set(surf);

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        infer_api_.InferSync(in, out);

        // gddeploy::InferResult result = any_cast<gddeploy::InferResult>(out->result); 
        gddeploy::InferResult result = out->data[0]->GetLref<gddeploy::InferResult>(); 
        PrintResult(result);

        if (is_draw && save_path != ""){
            auto pic_paths = std::vector<std::string>{};
            auto surfs = std::vector<gddeploy::BufSurfWrapperPtr>{surf};
            auto in_mats = std::vector<cv::Mat>{in_mat};
            std::string save_path = this->save_path_ + "frame_" + std::to_string(frame_id)+".jpg";
            DrawBbox(result, pic_paths, in_mats, surfs, save_path);
        }
        frame_id++;
    }

    return 0;
}

VideoRunner::VideoRunner()
{
    priv_ = std::make_shared<VideoRunnerPriv>();
}

int VideoRunner::Init(const std::string config, std::string model_path, std::string license_path)
{
    priv_->Init(config, model_path, license_path);

    return 0;
}

int VideoRunner::Init(const std::string config, std::vector<std::string> model_paths, std::vector<std::string> license_paths)
{
    priv_->Init(config, model_paths, license_paths);

    return 0;
}

int VideoRunner::OpencvOpen(std::string video_path, std::string save_path, bool is_draw)
{
    priv_->OpencvOpen(video_path, save_path, is_draw);

    return 0;
}

int VideoRunner::OpenStream(std::string video_path, std::string save_path, bool is_draw)
{
    priv_->OpenStream(video_path, save_path, is_draw);

    return 0;
}

int VideoRunner::Join()
{
    priv_->Join();

    return 0;
}