#include "status_update_node_v2.h"
#include "spdlog/spdlog.h"
#include "types.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace gddi {
namespace nodes {

void StatusUpdate_v2::on_setup() {}

std::tuple<int, int> is_report(const std::shared_ptr<msgs::cv_frame> &clone_frame,
                               const std::shared_ptr<msgs::cv_frame> &cache_frame_, const std::string &label_,
                               const float iou_threshold_, const float angle__threshold_) {

    auto &back_ext_info = clone_frame->frame_info->ext_info.back();
    auto &cache_ext_info = cache_frame_->frame_info->ext_info.back();
    auto &box_ext_info = clone_frame->frame_info->ext_info.front();
    float bigest_area = 0;
    float max_prob = 0;
    int flag = 0;
    int number = 0;
    int cross = 0;
    Rect2f hook_box;
    Rect2f car_box;
    // 1.找最大框
    for (const auto &[_, item] : box_ext_info.map_target_box) {
        if (box_ext_info.map_class_label.at(item.class_id) != label_) {
            bigest_area = std::max(bigest_area, (float)item.box.area());
        } else {
            max_prob = std::max(max_prob, (float)item.prob);
            if (std::abs(max_prob - (float)item.prob) < 1e-6) { hook_box = item.box; }
        }
    }
    // 2.找交叉框,保留上报框
    for (auto iter1 = box_ext_info.map_target_box.begin(); iter1 != box_ext_info.map_target_box.end();) {
        if (box_ext_info.map_class_label.at(iter1->second.class_id) != label_
            && std::abs(bigest_area - (float)iter1->second.box.area()) < 1e-6) {
            //iou
            float w = std::min(hook_box.x + hook_box.width, iter1->second.box.x + iter1->second.box.width)
                - std::max(hook_box.x, iter1->second.box.x);
            float h = std::min(hook_box.y + hook_box.height, iter1->second.box.y + iter1->second.box.height)
                - std::max(hook_box.y, iter1->second.box.y);
            float intersection = (w > 0 && h > 0) ? w * h : 0;
            float union_area = (float)hook_box.area() + (float)iter1->second.box.area() - intersection;
            float iou_frame = (union_area > 0) ? intersection / union_area : 0;
            //angle
            float dx = iter1->second.box.x + iter1->second.box.width / 2 - hook_box.x - hook_box.width / 2;
            float dy = iter1->second.box.y + iter1->second.box.height / 2 - hook_box.y - hook_box.height / 2;
            float angle_frame = std::abs(180 * std::atan2(dy, dx) / std::acos(-1));
            //select
            if ((hook_box.y + hook_box.height / 2 < iter1->second.box.y + iter1->second.box.height / 2)
                && (iou_frame > iou_threshold_) && (angle_frame < angle__threshold_)) {
                car_box = iter1->second.box;
                cross++;
                iter1++;
            } else {
                iter1 = box_ext_info.map_target_box.erase(iter1);
            }
        } else {
            iter1 = box_ext_info.map_target_box.erase(iter1);
        }
    }
    // 3.对比前后帧编号,保留上报编号
    if (cross > 0) {
        for (auto iter2 = back_ext_info.map_ocr_info.begin(); iter2 != back_ext_info.map_ocr_info.end();) {
            number++;
            std::string label;
            for (auto &value : iter2->second.labels) { label += value.str; }
            if ((iter2->second.points[0].x >= car_box.x && iter2->second.points[0].x <= car_box.x + car_box.width)
                && (iter2->second.points[0].y >= car_box.y && iter2->second.points[0].y <= car_box.y + car_box.height)
                && (label.length() == 3)) {
                iter2++;
                for (const auto &[i_cache, cache_ocr_info] : cache_ext_info.map_ocr_info) {
                    std::string cache_label;
                    for (auto &value : cache_ocr_info.labels) { cache_label += value.str; }
                    //spdlog::info("========= label: {}, cache_label: {}", label, cache_label);
                    if (cache_label.length() == 3 && cache_label != label) { flag++; }
                }
            } else {
                iter2 = back_ext_info.map_ocr_info.erase(iter2);
            }
        }
    }

    return std::make_tuple(flag, number);
}

void StatusUpdate_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_result_(frame);
        return;
    }
    if (!cache_frame_) {
        cache_frame_ = frame;
        return;
    }

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
    auto [flag, number] = is_report(clone_frame, cache_frame_, label_, iou_threshold_, angle__threshold_);
    if (flag > 0) {//上报结果
        cache_frame_->check_report_callback_ = [](const std::vector<FrameExtInfo> &) { return FrameType::kReport; };
        clone_frame = cache_frame_;
    }
    if (number > 0) {//更新编号
        cache_frame_ = frame;
    }
    output_result_(clone_frame);
}

}// namespace nodes
}// namespace gddi
