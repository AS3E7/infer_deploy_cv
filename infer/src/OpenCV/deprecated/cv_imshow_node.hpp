//
// Created by cc on 2021/11/5.
//

#ifndef FFMPEG_WRAPPER_SRC_OPENCV_CV_IMSHOW_NODE_HPP_
#define FFMPEG_WRAPPER_SRC_OPENCV_CV_IMSHOW_NODE_HPP_

#include "utils.hpp"
#include "nodes/node_any_basic.hpp"
#include "nodes/message_templates.hpp"
#include "nodes/ffvcodec/decoder_node_v1.hpp"
#include "cv_tools.hpp"

namespace gddi {
namespace nodes {

namespace cv_nodes {

class CvImShow_v1 : public node_any_basic<CvImShow_v1> {
public:
    explicit CvImShow_v1(std::string name)
        : node_any_basic<CvImShow_v1>(std::move(name)) {
        register_input_message_handler_<msgs::video_frame>(&CvImShow_v1::on_frame_, this);
        register_input_message_handler_<msgs::av_status>(&CvImShow_v1::on_status_, this);
        cv_thread::get_instance().add_listener_on_quit([this](int c) { quit_runner_((TaskErrorCode)c); }, this);
    }

    ~CvImShow_v1() override { cv_thread::get_instance().remove_listener(this); }

protected:
    void on_frame_(const std::shared_ptr<msgs::video_frame> &video_frame) {
        cv_thread::get_instance().imshow(name(), av_wrapper::avframe_to_cv_mat(video_frame->message));
    }

    void on_status_(const msgs::av_status::ptr &status) {}
};
}

}
}

#endif //FFMPEG_WRAPPER_SRC_OPENCV_CV_IMSHOW_NODE_HPP_
