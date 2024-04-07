//
// Created by cc on 2021/11/1.
//

#include "cv_tools.hpp"
#include "debug_tools.hpp"
#include <unordered_map>

namespace av_wrapper {

union UnionKey {
    struct {
        short src_pixfmt;
        short src_w;
        short src_h;
        short target_pixfmt;
    };
    int64_t hash_key;
};

class SwsContextManager {
public:
    SwsContextManager() = default;
    ~SwsContextManager() {
        for (const auto &item: ctx_table_) {
            sws_freeContext(item.second);
            DEBUG_release_view(item.second);
        }
    }
    SwsContext *get_context_for(const AVFrame *frame, AVPixelFormat target_fmt = AVPixelFormat::AV_PIX_FMT_BGR24) {
        UnionKey union_key{};
        union_key.src_pixfmt = (short)frame->format;
        union_key.src_w = (short)frame->width;
        union_key.src_h = (short)frame->height;
        union_key.target_pixfmt = (short)target_fmt;

        auto iter = ctx_table_.find(union_key.hash_key);
        if (iter != ctx_table_.end()) {
            return iter->second;
        } else {
            int dst_w = union_key.src_w;
            int dst_h = union_key.src_h;
            auto conversion = sws_getContext(union_key.src_w,
                                             union_key.src_h,
                                             (AVPixelFormat)frame->format,
                                             dst_w,
                                             dst_h,
                                             target_fmt,
                                             SWS_FAST_BILINEAR,
                                             nullptr,
                                             nullptr,
                                             nullptr);
            ctx_table_[union_key.hash_key] = conversion;
            return conversion;
        }
    }

    static AVPixelFormat pixfmt_cvt(enum AVPixelFormat src_pix_fmt) {
        switch (src_pix_fmt) {
            case AV_PIX_FMT_YUVJ420P: return AV_PIX_FMT_YUV420P;
            case AV_PIX_FMT_YUVJ422P: return AV_PIX_FMT_YUV422P;
            case AV_PIX_FMT_YUVJ444P: return AV_PIX_FMT_YUV444P;
            case AV_PIX_FMT_YUVJ440P: return AV_PIX_FMT_YUV440P;
            default:return src_pix_fmt;
        }
    }

    static std::pair<double, cv::Mat> to_cv_mat_bgr(const AVFrame *frame) {
        DebugTools::TickMetrics tick_metrics;
        tick_metrics.begin();

        int width = frame->width;
        int height = frame->height;
        cv::Mat image(height, width, CV_8UC3);
        int cv_linesizes[1];
        cv_linesizes[0] = (int)image.step1();

        auto conversion = sws_getContext(width,
                                         height,
                                         pixfmt_cvt((AVPixelFormat)frame->format),
                                         width,
                                         height,
                                         AV_PIX_FMT_BGR24,
                                         SWS_FAST_BILINEAR,
                                         nullptr,
                                         nullptr,
                                         nullptr);
        sws_scale(conversion, frame->data, frame->linesize, 0, height, &image.data, cv_linesizes);
        sws_freeContext(conversion);
        return {tick_metrics.elapsedMilliseconds(), image};
    }

    // TODO, bug, cvt to yuv always failed
//    std::pair<double, cv::Mat> to_cv_mat_yuv(const AVFrame *frame) {
//        DebugTools::TickMetrics tick_metrics;
//        tick_metrics.begin();
//
//        int width = frame->width;
//        int height = frame->height;
//        cv::Mat image(height * 3 / 2, width, CV_8UC1);
//
//        int cv_linesizes[2];
//        cv_linesizes[0] = (int)image.step1();
//        cv_linesizes[1] = (int)image.step1();
//
//        uint8_t *img_data[2];
//        img_data[0] = image.data;
//        img_data[1] = image.data + height * width;
//
//        SwsContext *conversion = get_context_for(frame, AV_PIX_FMT_YUV420P);
//        sws_scale(conversion, frame->data, frame->linesize, 0, height, img_data, cv_linesizes);
//
//        cv::Mat image_bgr;
//        cv::cvtColor(image, image_bgr, cv::COLOR_YUV2RGBA_NV12);
//        return {tick_metrics.elapsedMilliseconds(), image_bgr};
//    }

protected:
    std::unordered_map<int64_t, SwsContext *> ctx_table_;
};

SwsContextManager &context_manager() {
    static std::shared_ptr<SwsContextManager> context_manager_ptr = nullptr;
    if (!context_manager_ptr) {
        context_manager_ptr = std::make_shared<SwsContextManager>();
    }
    return *context_manager_ptr;
}

std::pair<double, cv::Mat> avframe_to_cv_mat2(const AVFrame *frame) {
    return SwsContextManager::to_cv_mat_bgr(frame);
}

cv::Mat avframe_to_cv_mat(const AVFrame *frame) {
    auto &&result = SwsContextManager::to_cv_mat_bgr(frame);

    std::cout
        << "RAW: " << av_get_pix_fmt_name((AVPixelFormat)frame->format)
        << ", time used for sws: " << result.first
        << std::endl;
    return result.second;
}

//void show_frame(const std::string &window_name, AVFrame *frame) {
//    auto &&img = avframe_to_cv_mat(frame);
//    cv::imshow(window_name, img);
//}
}

//cv_thread &cv_thread::get_instance() {
//    static std::mutex mutex;
//    static std::shared_ptr<cv_thread> this_;
//
//    std::lock_guard<std::mutex> lock_guard(mutex);
//
//    if (this_ == nullptr) {
//        this_ = std::make_shared<cv_thread>();
//        auto ui_thread = std::thread([=]() {
//            this_->loop_message();
//            std::cout << "==========cv_thread quit!\n";
//        });
//        ui_thread.detach();
//    }
//    return *this_;
//}
//
//void cv_thread::loop_message() {
//    while (true) {
//        std::pair<int, std::shared_ptr<CvAction> > message;
//        if (message_queue_.try_dequeue(message)) {
//            if (process_message(message)) {
//                break;
//            }
//        } else {
//            auto key = cv::waitKey(1);
//            if (key == 27) {
//                call_on_quit(0);
//            }
//        }
//    }
//}
//
//bool cv_thread::process_message(std::pair<int, std::shared_ptr<CvAction> > &message) {
//    if (message.first < 0) {
//        call_on_quit(message.first);
//        return true;
//    }
//
//    auto show_message = std::dynamic_pointer_cast<CvActionShow>(message.second);
//    if (show_message) {
//        cv::imshow(show_message->name, show_message->image);
//    } else {
//        auto func_message = std::dynamic_pointer_cast<CvActionFunc>(message.second);
//        if (func_message) {
//            func_message->actor();
//        }
//    }
//
//    auto key = cv::waitKey(1);
//    if (key == 27) {
//        call_on_quit(message.first);
//    }
//    return false;
//}
