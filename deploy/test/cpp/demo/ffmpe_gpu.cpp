#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>

#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

#define MAX_QUEUE_SIZE 10
#define MAX_WIDTH 1920
#define MAX_HEIGHT 1080

// 定义一个帧结构体
struct AVFrameDeleter {
    void operator() (AVFrame* frame) {
        av_frame_free(&frame);
    }
};
typedef std::unique_ptr<AVFrame, AVFrameDeleter> AVFramePtr;

// 定义一个GPU缓存结构体
struct GPUCache {
    uint8_t* data[MAX_QUEUE_SIZE];
    size_t pitch[MAX_QUEUE_SIZE];
    int width;
    int height;
    int size = 0;

    // 构造函数
    GPUCache(int w, int h) : width(w), height(h) {}

    // 析构函数
    ~GPUCache() {
        for (int i = 0; i < size; i++) {
            cudaFree(data[i]);
        }
    }

    // 添加一帧GPU缓存
    void addData(const uint8_t* buf, size_t p, int w, int h) {
        if (size >= MAX_QUEUE_SIZE) return;
        size_t size = p * h;
        cudaMalloc(&data[size], size);
        cudaMemcpy2D(data[size], pitch[size], buf, p, p, h, cudaMemcpyHostToDevice);
        pitch[size] = p;
        width = w;
        height = h;
        size++;
    }

    // 获取一帧GPU缓存
    uint8_t* getData(int index) {
        if (index >= size) return nullptr;
        return data[index];
    }
};

// 定义一个解码器
class VideoDecoder {
private:
    AVCodecContext* m_codec_context;
    AVFormatContext* m_format_context;
    int m_video_stream_idx = -1;

    std::mutex m_mutex;
    std::queue<AVFramePtr> m_frame_queue;
    std::unique_ptr<GPUCache> m_gpu_cache;

public:
    // 构造函数
    VideoDecoder(const std::string& filename) {
        av_register_all();
        avcodec_register_all();

        AVFormatContext* format_context = NULL;
        if (avformat_open_input(&format_context, filename.c_str(), NULL, NULL) < 0) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        if (avformat_find_stream_info(format_context, NULL) < 0) {
            throw std::runtime_error("Could not find stream information");
        }

        int video_stream_index = -1;
        AVCodecContext* codec_context = NULL;

        for (int i = 0; i < format_context->nb_streams; i++) {
            if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_stream_index = i;
                codec_context = avcodec_alloc_context3(NULL);
                if (!codec_context) {
                    throw std::runtime_error("Could not allocate codec context");
                }

                if (avcodec_parameters_to_context(codec_context, format_context->streams[i]->codecpar) < 0) {
                    throw std::runtime_error("Could not copy codec parameters to context");
                }

                AVCodec* codec = avcodec_find_decoder(codec_context->codec_id);
                if (!codec) {
                    throw std::runtime_error("Unsupported codec");
                }

                if (avcodec_open2(codec_context, codec, NULL) < 0) {
                    throw std::runtime_error("Could not open codec");
                }
            }
        }

        if (video_stream_index == -1 || !codec_context) {
            throw std::runtime_error("Could not find video stream");
        }

        m_video_stream_idx = video_stream_index;
        m_codec_context = codec_context;
        m_format_context = format_context;

        m_gpu_cache = std::make_unique<GPUCache>(MAX_WIDTH, MAX_HEIGHT);
    }

    // 解码线程
    void decodeThread() {
        AVPacket packet;
        while (av_read_frame(m_format_context, &packet) >= 0) {
            if (packet.stream_index == m_video_stream_idx) {
                // 解码视频帧
                AVFramePtr frame(av_frame_alloc());
                int ret = avcodec_send_packet(m_codec_context, &packet);
                if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                    std::cerr << "Error sending a packet for decoding: " << ret << std::endl;
                    continue;
                }

                ret = avcodec_receive_frame(m_codec_context, frame.get());
                if (ret < 0 && ret != AVERROR_EOF) {
                    std::cerr << "Error during decoding: " << ret << std::endl;
                    continue;
                }

                // 将视频帧加入帧队列
                std::unique_lock<std::mutex> lock(m_mutex);
                m_frame_queue.push(std::move(frame));
                if (m_frame_queue.size() > MAX_QUEUE_SIZE) {
                    m_frame_queue.pop();
                }

                // 将视频帧缓存到GPU
                int pitch = AVFrame::linesize[0];
                m_gpu_cache->addData(AVFrame::data[0], pitch, m_codec_context->width, m_codec_context->height);
            }
            av_packet_unref(&packet);
        }
    }

    // 从帧队列中获取一帧视频帧
    AVFramePtr getFrame() {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_frame_queue.empty()) {
            return nullptr;
        }
        AVFramePtr frame = std::move(m_frame_queue.front());
        m_frame_queue.pop();
        return frame;
    }

    // 获取GPU缓存
    uint8_t* getGPUFrame(int index) {
        return m_gpu_cache->getData(index);
    }

    // 获取缓存的GPU帧数
    int getGPUFrameCount() {
        return m_gpu_cache->size;
    }

    // 获取视频帧宽度
    int getWidth() const {
        return m_codec_context->width;
    }

    // 获取视频帧高度
    int getHeight() const {
        return m_codec_context->height;
    }

    // 获取视频帧时间间隔
    int getFrameInterval() const {
        return 1000000.0 / m_codec_context->framerate.den * m_codec_context->framerate.num;
    }

    // 获取视频帧格式
    AVPixelFormat getPixelFormat() const {
        return m_codec_context->pix_fmt;
    }

    // 获取视频帧时间基
    AVRational getTimebase() const {
        return m_format_context->streams[m_video_stream_idx]->time_base;
    }

    // 析构函数
    ~VideoDecoder() {
        avcodec_free_context(&m_codec_context);
        avformat_close_input(&m_format_context);
    }
};