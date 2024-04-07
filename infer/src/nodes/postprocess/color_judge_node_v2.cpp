#include "color_judge_node_v2.h"
#include "spdlog/spdlog.h"
#include "types.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace gddi {
namespace nodes {

void ColorJudge_v2::on_setup() {}

void ColorJudge_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_result_(frame);
        return;
    }

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
    auto &back_ext_info = clone_frame->frame_info->ext_info.back();
    int flag = 0;
    int class_id_ = back_ext_info.map_class_label.size();
    Scalar redColor = {1.0f, 0.0f, 0.0f, 1.0f}, blueColor = {0.0f, 0.0f, 1.0f, 1.0f};
    auto src_frame = image_wrapper::image_to_mat(frame->frame_info->src_frame->data);
    cv::Mat hsv, mask, rotated, cropped;
    cv::Scalar lower_yellow(20, 100, 100);
    cv::Scalar upper_yellow(50, 255, 255);

    for (auto iter = back_ext_info.map_key_points.begin(); iter != back_ext_info.map_key_points.end(); iter++) {
        std::vector<cv::Point2f> points;
        for (int i = 0; i < iter->second.size(); i++) {
            cv::Point2f point(iter->second[i].x, iter->second[i].y);
            points.emplace_back(point);
        }
        cv::RotatedRect rect = cv::minAreaRect(points);//还原旋转矩形区域
        cv::warpAffine(src_frame, rotated, cv::getRotationMatrix2D(rect.center, rect.angle, 1.0), src_frame.size());
        cv::getRectSubPix(rotated, rect.size, rect.center, cropped);
        cv::cvtColor(cropped, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, lower_yellow, upper_yellow, mask);
        int yellow_pixel_count = cv::countNonZero(mask);
        int class_id = back_ext_info.map_target_box[iter->first].class_id;
        // double yellow_ratio = static_cast<double>(yellow_pixel_count) / (mask.rows * mask.cols);
        if (yellow_pixel_count > 0) {
            if (back_ext_info.map_class_label.at(class_id) == label_) {
                back_ext_info.map_target_box[iter->first].class_id = class_id_;
                flag++;
            }
            if (back_ext_info.map_class_label.at(class_id) == "2") {
                back_ext_info.map_target_box[iter->first].class_id = class_id_ + 1;
                flag++;
            }
        }
        // spdlog::info("size:{},class_id:{},yellow_pixel_count:{}", back_ext_info.map_target_box.size(),
        //              back_ext_info.map_target_box[iter->first].class_id, yellow_pixel_count);
    }
    if (flag > 0) {
        back_ext_info.map_class_label[class_id_] = std::to_string(class_id_);        //黄色实线
        back_ext_info.map_class_label[class_id_ + 1] = std::to_string(class_id_ + 1);//黄色虚线
        back_ext_info.map_class_color[class_id_] = redColor;                         //红色
        back_ext_info.map_class_color[class_id_ + 1] = blueColor;                    //蓝色
    }
    // for (const auto &[_, item] : back_ext_info.map_target_box) {
    //     spdlog::info("size:{},class_id:{}", back_ext_info.map_target_box.size(), item.class_id);
    // }

    output_result_(clone_frame);
}

}// namespace nodes
}// namespace gddi
