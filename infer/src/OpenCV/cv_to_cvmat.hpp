//
// Created by cc on 2021/11/9.
//

#ifndef FFMPEG_WRAPPER_SRC_OPENCV_CV_TO_CVMAT_HPP_
#define FFMPEG_WRAPPER_SRC_OPENCV_CV_TO_CVMAT_HPP_

#include "utils.hpp"
#include "nodes/node_any_basic.hpp"
#include "nodes/message_templates.hpp"
#include "nodes/ffvcodec/decoder_node_v1.hpp"
#include "cv_tools.hpp"
#include "yuv_rgb/yuv_rgb.h"
#include "modules/server_send/ws_notifier.hpp"
namespace gddi {
namespace nodes {

namespace msgs {
typedef simple_message<cv::Mat> cv_mat;
}

namespace cv_nodes {

class AvToCvMat_v1 : public node_any_basic<AvToCvMat_v1> {
private:
    message_pipe<msgs::cv_mat> raise_cv_mat_;
    bool show_time_info_;
public:
    explicit AvToCvMat_v1(std::string name)
        : node_any_basic(std::move(name)), show_time_info_(false) {
        bind_simple_property("print_sws_time", show_time_info_);
        raise_cv_mat_ = register_output_message_<msgs::cv_mat>();
        register_input_message_handler_<msgs::video_frame>([this](const std::shared_ptr<msgs::video_frame> &video_frame) {
            auto &&result = av_wrapper::avframe_to_cv_mat2(video_frame->message);
            auto cv_mat_msg = msgs::cv_mat::make_shared(result.second);
            cv_mat_msg->copy_timestamp_ref_info(video_frame);
            raise_cv_mat_(cv_mat_msg);

            if (show_time_info_) {
                std::cout
                    << "RAW: " << av_get_pix_fmt_name((AVPixelFormat)video_frame->message->format)
                    << std::setw(12) << video_frame->message->pts
                    << ", time used for sws: " << result.first
                    << std::endl;
            }
        });
    }
};

class AvToCvMat_v1_1 : public node_any_basic<AvToCvMat_v1_1> {
private:
    message_pipe<msgs::cv_mat> raise_cv_mat_;
    bool show_time_info_{false};
    bool ws_send{true};
    std::string ws_view_id;
public:
    explicit AvToCvMat_v1_1(std::string name)
        : node_any_basic(std::move(name)), show_time_info_(false) {
        bind_simple_property("print_sws_time", show_time_info_);
        bind_simple_property("ws_send", ws_send);
        bind_simple_property("ws_view_id", ws_view_id, "");
        ws_view_id = this->name();

        raise_cv_mat_ = register_output_message_<msgs::cv_mat>();
        register_input_message_handler_<msgs::video_frame>([this](const std::shared_ptr<msgs::video_frame> &video_frame) {
            auto &&result = av_wrapper::avframe_to_cv_mat2(video_frame->message);
            auto cv_mat_msg = msgs::cv_mat::make_shared(result.second);
            cv_mat_msg->copy_timestamp_ref_info(video_frame);
            raise_cv_mat_(cv_mat_msg);

            if (show_time_info_) {
                std::cout
                    << "RAW: " << av_get_pix_fmt_name((AVPixelFormat)video_frame->message->format)
                    << std::setw(12) << video_frame->message->pts
                    << ", time used for sws: " << result.first
                    << std::endl;
            }

            if (ws_send) {
                std::vector<unsigned char> img_data;
                cv::imencode(".jpg", result.second, img_data);
                ws_notifier::push_image(ws_view_id, img_data);
            }
        });
    }
};

class AvToCvMat_opencv2 : public node_any_basic<AvToCvMat_opencv2> {
private:
    message_pipe<msgs::cv_mat> raise_cv_mat_;

public:
    explicit AvToCvMat_opencv2(std::string name)
        : node_any_basic<AvToCvMat_opencv2>(std::move(name)) {
        // 1. create output endpoint
        raise_cv_mat_ = register_output_message_<msgs::cv_mat>();

        // 2. create input endpoint
        register_input_message_handler_<msgs::video_frame>([this](const std::shared_ptr<msgs::video_frame> &video_frame) {
            auto height = video_frame->message->height;
            auto width = video_frame->message->width;
            cv::Mat yuv_img(height * 3 / 2, width, CV_8UC1);

            memcpy(yuv_img.data, video_frame->message->data[0], height * width);
            memcpy(yuv_img.data + height * width, video_frame->message->data[1], height / 2 * width);

            cv::Mat rgb_img;
            cv::cvtColor(yuv_img, rgb_img, cv::COLOR_YUV2BGRA_NV12);
            auto final_msg = msgs::cv_mat::make_shared(rgb_img);
            final_msg->copy_timestamp_ref_info(video_frame);
            raise_cv_mat_(final_msg);
        });
    }
};

class AvToCvMat_yuv2rgb : public node_any_basic<AvToCvMat_yuv2rgb> {
private:
    message_pipe<msgs::cv_mat> raise_cv_mat_;

public:
    explicit AvToCvMat_yuv2rgb(std::string name)
        : node_any_basic<AvToCvMat_yuv2rgb>(std::move(name)) {
        // 1. create output endpoint
        raise_cv_mat_ = register_output_message_<msgs::cv_mat>();

        // 2. create input endpoint
        register_input_message_handler_<msgs::video_frame>([this](const std::shared_ptr<msgs::video_frame> &video_frame) {
            auto height = video_frame->message->height;
            auto width = video_frame->message->width;
            auto frame = video_frame->message;
            cv::Mat rgb_img(height, width, CV_8UC3);

            auto rgb_stride = rgb_img.step1(0);
            nv12_rgb24_std(width,
                           height,
                           frame->data[0],
                           frame->data[1],
                           frame->linesize[0],
                           frame->linesize[1],
                           rgb_img.data,
                           (uint32_t)rgb_stride,
                           YCBCR_601);
            cv::cvtColor(rgb_img, rgb_img, cv::COLOR_RGB2BGR);

            auto final_msg = msgs::cv_mat::make_shared(rgb_img);
            final_msg->copy_timestamp_ref_info(video_frame);
            raise_cv_mat_(final_msg);
        });
    }
};

}

}
}

#endif //FFMPEG_WRAPPER_SRC_OPENCV_CV_TO_CVMAT_HPP_
