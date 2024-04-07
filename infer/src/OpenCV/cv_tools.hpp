//
// Created by cc on 2021/11/1.
//

#ifndef FFMPEG_WRAPPER_SRC_OPENCV_CV_TOOLS_HPP_
#define FFMPEG_WRAPPER_SRC_OPENCV_CV_TOOLS_HPP_

#include "../modules/endecode/av_lib.h"
#include <opencv2/opencv.hpp>
#include <utility>
#include <blockingconcurrentqueue.h>
#include <unordered_map>
#include "utils.hpp"
namespace av_wrapper {

/**
 * @brief
 * @param window_name
 * @param frame
 * @remarks 需要cv::waitKey()
 */
// void show_frame(const std::string &window_name, AVFrame *frame);
std::pair<double, cv::Mat> avframe_to_cv_mat2(const AVFrame *frame);
cv::Mat avframe_to_cv_mat(const AVFrame *frame);

}

// Deprecated!!! by cc
//class cv_thread {
//private:
//    struct CvAction {
//        virtual ~CvAction() = default;
//    };
//    struct CvActionShow : public CvAction {
//        std::string name;
//        cv::Mat image;
//        explicit CvActionShow(std::string n, cv::Mat m) : name(std::move(n)), image(std::move(m)) {}
//    };
//
//    struct CvActionFunc : public CvAction {
//        std::function<void()> actor;
//        explicit CvActionFunc(std::function<void()> f) : actor(std::move(f)) {}
//    };
//
//public:
//    ~cv_thread() { std::cout << "destructor: " << gddi::utils::get_class_name(this) << std::endl; }
//
//    void imshow(const std::string &name, const cv::Mat &image) {
//        if (message_queue_.size_approx() < 10) {
//            message_queue_.enqueue({0, std::make_shared<CvActionShow>(name, image)});
//        }
//    }
//
//    void named_window(const std::string &name, int flags) {
//        message_queue_.enqueue({0, std::make_shared<CvActionFunc>([=]() {
//            cv::namedWindow(name, flags);
//            cv::setWindowProperty(name, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
//        })});
//    }
//
//    void add_listener_on_quit(std::function<void(int)> on_quit, const void *ctx) {
//        std::lock_guard<std::mutex> lock_guard(mutex_);
//        on_quit_[ctx] = std::move(on_quit);
//    }
//    void remove_listener(const void *ctx) {
//        std::lock_guard<std::mutex> lock_guard(mutex_);
//        on_quit_.erase(ctx);
//    }
//
//    static cv_thread &get_instance();
//
//private:
//    void loop_message();
//    bool process_message(std::pair<int, std::shared_ptr<CvAction> > &message);
//
//    void call_on_quit(int code) {
//        std::lock_guard<std::mutex> lock_guard(mutex_);
//        for (const auto &it: on_quit_) {
//            it.second(code);
//        }
//        cv::destroyAllWindows();
//    }
//
//private:
//    moodycamel::BlockingConcurrentQueue<std::pair<int, std::shared_ptr<CvAction> > > message_queue_;
//    std::unordered_map<const void *, std::function<void(int)>> on_quit_;
//    std::mutex mutex_;
//};

#endif //FFMPEG_WRAPPER_SRC_OPENCV_CV_TOOLS_HPP_
