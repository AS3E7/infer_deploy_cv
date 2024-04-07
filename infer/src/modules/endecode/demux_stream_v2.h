#ifndef __DEMUXER_STREAM_H__
#define __DEMUXER_STREAM_H__

#include "av_lib.h"
#include <map>

namespace av_wrapper {
class Demuxer_v2 {
public:
    typedef std::function<void(const AVStream *)> on_open_t;
    typedef std::function<void(const std::shared_ptr<AVPacket> &)> on_packet_t;

    struct demuxer_options {
        bool tcp_transport{true};// 是否使用tcp来传输rtsp，默认 true
        // 是否跳过第一个I-Frame，默认 true, 第一个I-Frame貌似会pts错误
        bool jump_first_video_i_frame{true};
        int readrate_speed{1};                   // 1: 使用视频内置帧率
        std::function<void(bool)> on_stream_exit;// demux 过程退出, true 正常结束。false 打开失败
    };

public:
    Demuxer_v2();
    Demuxer_v2(Demuxer_v2 &&demuxer);                  // Movable
    Demuxer_v2(const Demuxer_v2 &) = delete;           // No Copy
    Demuxer_v2 &operator=(const Demuxer_v2 &) = delete;// No Copy
    ~Demuxer_v2();

    /**
         * @brief thread-safe RTSP stream demuxer
         * 
         * @param stream_url RTSP video url
         * @param on_open 视频打开成功后的回调，返回false会让退出视频demuxer过程
         * @param on_packet 视频数据帧的回调
         * @param options 
         */
    void open_stream(const std::string &stream_url, const on_open_t &on_open,
                     const on_packet_t &on_packet, const demuxer_options &options);

    /**
         * @brief 停止视频解包。 注意这个不能在回调函数内调用。
         * 
         */
    void stop_stream();

    void dump_demuxer_stat(std::ostream &oss);

    std::string get_iformat_name();
    double get_video_frame_rate();

protected:
    /**
         * @brief 这是内部函数，在 stop_stream()使用的。
         * 
         */
    bool open_stream_impl(int &real_stream_index);
    void read_stream_packet();
    void stop_stream_impl();

    bool is_video_file();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}// namespace av_wrapper

#endif