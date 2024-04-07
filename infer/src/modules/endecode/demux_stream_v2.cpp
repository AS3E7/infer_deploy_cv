#include "demux_stream_v2.h"
#include "basic_logs.hpp"
#include <algorithm>

namespace av_wrapper {
struct OpaqueOptions {
    bool demux_on;
    std::chrono::steady_clock::time_point time_point;
};

struct Demuxer_v2::Impl {
    std::mutex mtx;

    on_open_t on_open;
    on_packet_t on_packet;

    AVFormatContext *fmt_ctx = nullptr;
    AVDictionary *dicts = nullptr;

    std::thread demuxer_worker_thread;
    demuxer_options options;// demux options
    std::string stream_url; // RTSP的地址
    // bool demux_on;                   // 用于循环控制的标志
    OpaqueOptions opaque;
    // AVRational timebase;             // AVRational
    int video_stream_index;// Video Stream的索引
    // int frame_rate;                  // fps

    int64_t packet_count;              // 读取包计数
    int64_t packet_video_count;        // 视频帧统计
    int64_t packet_video_i_frame_count;// 视频I-Frame帧统计

    std::pair<bool, std::map<int, std::string>> stream_status;

    // uint64_t nb_frames;
};

Demuxer_v2::Demuxer_v2() : impl_(std::make_unique<Demuxer_v2::Impl>()) {}

Demuxer_v2::~Demuxer_v2() { stop_stream_impl(); }

Demuxer_v2::Demuxer_v2(Demuxer_v2 &&demuxer) : impl_(std::move(demuxer.impl_)) {}

void Demuxer_v2::open_stream(const std::string &stream_url, const on_open_t &on_open,
                             const on_packet_t &on_packet, const demuxer_options &options) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    // 1. stop if opened
    stop_stream_impl();

    // 2. setup priv data
    impl_->stream_url = stream_url;
    impl_->on_open = on_open;
    impl_->on_packet = on_packet;
    impl_->options = options;
    impl_->opaque.demux_on = true;

    // 3. run the demuxer
    impl_->demuxer_worker_thread = std::thread([this]() {
        int retry_time = 0;
        while (impl_->opaque.demux_on) {
            // try open the rtsp stream
            if (open_stream_impl(impl_->video_stream_index)) {
                retry_time = 0;
                // save the time_base
                // impl_->timebase = fmt_ctx->streams[impl_->video_stream_index]->time_base;
                // impl_->nb_frames = fmt_ctx->streams[impl_->video_stream_index]->nb_frames;
                // impl_->frame_rate = av_q2d(fmt_ctx->streams[impl_->video_stream_index]->r_frame_rate);

                if (impl_->on_open) {
                    impl_->on_open(impl_->fmt_ctx->streams[impl_->video_stream_index]);
                }

                impl_->packet_count = 0;
                impl_->packet_video_count = 0;
                impl_->packet_video_i_frame_count = 0;

                // loop the packet.
                read_stream_packet();
            } else {
                std::this_thread::sleep_for(std::chrono::seconds(3));

                if (!impl_->opaque.demux_on) {
                    if (impl_->options.on_stream_exit) { impl_->options.on_stream_exit(false); }
                }
            }

            if (is_video_file()) { break; }
        }

        if (impl_->options.on_stream_exit) { impl_->options.on_stream_exit(true); }

        std::cout << "Stream exit: " << impl_->stream_url << ", packeds: " << impl_->packet_count
                  << std::endl;
        std::cout.flush();
    });
}

void Demuxer_v2::dump_demuxer_stat(std::ostream &oss) {
    std::lock_guard<std::mutex> lgk(impl_->mtx);

    oss << std::setw(12) << "total packet" << std::setw(12) << impl_->packet_count << '\n'
        << std::setw(12) << "video packet" << std::setw(12) << impl_->packet_video_count << '\n'
        << std::setw(12) << "video-i packet" << std::setw(12) << impl_->packet_video_i_frame_count
        << '\n';
}

std::string Demuxer_v2::get_iformat_name() {
    if (impl_->fmt_ctx) { return impl_->fmt_ctx->iformat->name; }
    return {};
}

double Demuxer_v2::get_video_frame_rate() {
    return av_q2d(impl_->fmt_ctx->streams[impl_->video_stream_index]->r_frame_rate);
}

bool Demuxer_v2::open_stream_impl(int &real_stream_index) {
    int ret = 0;

    impl_->stream_url.erase(std::remove_if(impl_->stream_url.begin(), impl_->stream_url.end(),
                                           [](char x) -> bool {
                                               return x == ' ' || x == '\t' || x == '\r'
                                                   || x == '\n';
                                           }),
                            impl_->stream_url.end());

    impl_->fmt_ctx = avformat_alloc_context();

    // setup rtsp demuxer Options
    if (impl_->options.tcp_transport) {
        av_dict_set(&impl_->dicts, "buffer_size", "1024000", 0);
        av_dict_set(&impl_->dicts, "rtsp_transport", "tcp", 0);
        av_dict_set(&impl_->dicts, "stimeout", "2000000", 0);
        av_dict_set(&impl_->dicts, "max_delay", "5000000", 0);
    }

#ifdef WITH_BM1684
    av_dict_set(&impl_->dicts, "rtsp_flags", "prefer_tcp", 0);

#ifdef BM_PCIE_MODE
    av_dict_set_int(&opts, "zero_copy", pcie_no_copyback, 0);
    av_dict_set_int(&opts, "sophon_idx", sophon_idx, 0);
#endif

#endif

    impl_->opaque.time_point = std::chrono::steady_clock::now();
    /*-------------- set callback, avoid blocking --------------*/
    impl_->fmt_ctx->interrupt_callback.callback = [](void *opaque) {
        OpaqueOptions *p_opaque = (OpaqueOptions *)opaque;
        if (!p_opaque->demux_on
            || std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now()
                                                                - p_opaque->time_point)
                    .count()
                > 3) {
            return 1;
        }
        return 0;
    };
    impl_->fmt_ctx->interrupt_callback.opaque = &impl_->opaque;
    /*--------------------------- end --------------------------*/

    try {
        if ((ret = avformat_open_input(&impl_->fmt_ctx, impl_->stream_url.c_str(), NULL,
                                       &impl_->dicts))
            != 0) {
            char errmsg[256];
            av_strerror(ret, errmsg, sizeof(errmsg));
            throw std::runtime_error("Couldn't open input stream: " + impl_->stream_url
                                     + ", cause: " + errmsg);
        }

        // read packets of a media file to get stream information
        if (avformat_find_stream_info(impl_->fmt_ctx, NULL) < 0) {
            throw std::runtime_error("Couldn't find stream information: " + impl_->stream_url);
        }

        // find the "best" stream in the file
        real_stream_index =
            av_find_best_stream(impl_->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (real_stream_index < 0) {
            throw std::runtime_error("Didn't find a video stream: " + impl_->stream_url);
        }
    } catch (const std::exception &e) {
        avformat_close_input(&impl_->fmt_ctx);
        std::cerr << e.what() << '\n';
        return false;
    }

    // dump stream info
    av_dump_format(impl_->fmt_ctx, real_stream_index, impl_->stream_url.c_str(), 0);

    return true;
}

void Demuxer_v2::read_stream_packet() {
    auto packet =
        std::shared_ptr<AVPacket>(av_packet_alloc(), [](AVPacket *ptr) { av_packet_free(&ptr); });
    int64_t start_time = av_gettime();

    while (impl_->opaque.demux_on) {
        if (av_read_frame(impl_->fmt_ctx, packet.get()) >= 0) {
            impl_->opaque.time_point = std::chrono::steady_clock::now();

            impl_->packet_count++;

            // is video stream
            if (packet->stream_index == impl_->video_stream_index) {
                impl_->packet_video_count++;
                impl_->packet_video_i_frame_count +=
                    (packet->flags & AV_PKT_FLAG_KEY) == AV_PKT_FLAG_KEY ? 1 : 0;

                // Jump first I-Frame & follow P-frame
                if (impl_->options.jump_first_video_i_frame
                    && impl_->packet_video_i_frame_count < 1) {
                    continue;
                }

                if (impl_->on_packet) { impl_->on_packet(packet); }
            }
        } else {
            // < 0 on error or end of file
            break;
        }

        if (impl_->options.readrate_speed == 0) {
            av_usleep(20000);// 20 ms
        } else if (impl_->options.readrate_speed == 1) {
            if (packet->pts < 0) {
                av_usleep(
                    1000000
                        / av_q2d(impl_->fmt_ctx->streams[impl_->video_stream_index]->r_frame_rate)
                    - 1000);
            } else {
                int64_t now_time = av_gettime() - start_time;
                int64_t pts_time = av_rescale_q(
                    packet->pts, impl_->fmt_ctx->streams[impl_->video_stream_index]->time_base,
                    {1, AV_TIME_BASE});
                if (pts_time > now_time) av_usleep(pts_time - now_time);
            }
        } else {
            av_usleep(1000000 / impl_->options.readrate_speed);// sleep 1s/speed ms
        }
    }
}

void Demuxer_v2::stop_stream_impl() {
    impl_->opaque.demux_on = false;
    if (impl_->demuxer_worker_thread.joinable()) {
        impl_->demuxer_worker_thread.join();
        std::cout << "Stream joined: " << impl_->stream_url << std::endl;
        std::cout.flush();
    }

    if (impl_->fmt_ctx) { avformat_close_input(&impl_->fmt_ctx); }

    if (impl_->dicts) { av_dict_free(&impl_->dicts); }

    impl_->packet_count = 0;
    impl_->packet_video_count = 0;
    impl_->packet_video_i_frame_count = 0;
}

bool Demuxer_v2::is_video_file() {
    if (impl_->stream_url.size() > 4) {
        auto subs = impl_->stream_url.substr(0, 4);
        std::transform(subs.begin(), subs.end(), subs.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (subs == "file" || subs == "http") { return true; }
    }
    return false;
}

void Demuxer_v2::stop_stream() {
    std::lock_guard<std::mutex> lgk(impl_->mtx);

    // 不可以在回调线程里调用 stop_stream
    if (std::this_thread::get_id() == impl_->demuxer_worker_thread.get_id()) {
        throw std::runtime_error("can not stop stream in callback thread!");
    }

    stop_stream_impl();
}
}// namespace av_wrapper