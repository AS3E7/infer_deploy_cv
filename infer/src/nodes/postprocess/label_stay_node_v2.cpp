#include "label_stay_node_v2.h"
#include "spdlog/spdlog.h"
#include "types.hpp"
#include <memory>
#include <numeric>
#include <vector>

namespace gddi {
namespace nodes {

void LabelStay_v2::on_setup() {}

void LabelStay_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_result_(frame);
        return;
    }

    auto &front_ext_info = frame->frame_info->ext_info.front();
    auto &back_ext_info = frame->frame_info->ext_info.back();

    int status = 0;
    for (const auto &[_, item] : back_ext_info.map_target_box) {
        if (back_ext_info.map_class_label.at(item.class_id) != label_) { continue; }

        int item_center_x = item.box.x + item.box.width / 2;
        int item_center_y = item.box.y + item.box.height / 2;

        int match_track_id = -1;
        int min_distance = frame->frame_info->width() * frame->frame_info->height();
        for (const auto &[track_id, target] : front_ext_info.tracked_box) {
            // 计算距离
            int target_center_x = target.box.x + target.box.width / 2;
            int target_center_y = target.box.y + target.box.height / 2;
            auto distance = std::pow(item_center_x - target_center_x, 2) + std::pow(item_center_y - target_center_y, 2);
            if (distance < min_distance) {
                min_distance = distance;
                match_track_id = track_id;
            }
        }
        if (match_track_id != -1) { ++stay_count_[match_track_id]; }
    }

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
    auto result = clone_frame->check_report_callback_(frame->frame_info->ext_info);
    if (result >= FrameType::kReport) {
        result = FrameType::kBase;
        for (const auto &[track_id, info] : front_ext_info.tracked_box) {
            if (!info.cross) { continue; }

            if (stay_count_.count(track_id) == 0 || stay_count_.at(track_id) < (stay_time_ * frame->infer_frame_rate)) {
                result = FrameType::kReport;
                if (stay_count_.count(track_id) > 0) { stay_count_.at(track_id) = 0; }
            }
        }

        // 移除已经离开的目标
        auto iter = stay_count_.begin();
        while (iter != stay_count_.end()) {
            if (front_ext_info.tracked_box.count(iter->first) == 0) {
                iter = stay_count_.erase(iter);
            } else {
                ++iter;
            }
        }

        clone_frame->check_report_callback_ = [result](const std::vector<FrameExtInfo> &) { return result; };
    }

    output_result_(clone_frame);
}

}// namespace nodes
}// namespace gddi
