//
// Created by cc on 2021/11/9.
//

#ifndef FFMPEG_WRAPPER_SRC_OPENCV_CV_IMSHOW_V2_HPP_
#define FFMPEG_WRAPPER_SRC_OPENCV_CV_IMSHOW_V2_HPP_

#include "utils.hpp"
#include "nodes/node_any_basic.hpp"
#include "nodes/message_templates.hpp"
#include "nodes/ffvcodec/decoder_node_v1.hpp"
#include "cv_tools.hpp"
#include "cv_to_cvmat.hpp"

namespace gddi {
namespace nodes {

namespace cv_nodes {

class CvImShow_v2 : public node_any_basic<CvImShow_v2> {
private:
    int window_flags_{-1}; // {cv::WINDOW_AUTOSIZE};
public:
    explicit CvImShow_v2(std::string name)
        : node_any_basic(std::move(name)) {
        bind_simple_property("window_flag", window_flags_, "");

        register_input_message_handler_<msgs::cv_mat>([this](const std::shared_ptr<msgs::cv_mat> &cv_mat) {
            cv_thread::get_instance().imshow(this->name(), cv_mat->message);
        });

        cv_thread::get_instance().add_listener_on_quit([this](int c) { quit_runner_((TaskErrorCode)c); }, this);
    }

    ~CvImShow_v2() override {
        cv_thread::get_instance().remove_listener(this);
    }

protected:
    void on_setup() override {
        if (window_flags_ >= 0) { cv_thread::get_instance().named_window(this->name(), window_flags_); }
    }
};
}

}
}

#endif //FFMPEG_WRAPPER_SRC_OPENCV_CV_IMSHOW_V2_HPP_
