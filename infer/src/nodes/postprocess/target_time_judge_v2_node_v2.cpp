#include "target_time_judge_v2_node_v2.h"
#include "spdlog/spdlog.h"
#include "types.hpp"
#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>

namespace gddi {
namespace nodes {

void TargetTimeJudge_v2::on_setup() {}

int get_min() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto tm = std::localtime(&t);
    return tm->tm_hour * 60 * 60 + tm->tm_min * 60 + tm->tm_sec;
}

bool Equal_Characters(const std::string &str1, const std::string &str2, int n) {
    if (str1.size() < n || str2.size() < n) { return false; }

    for (int i = 0; i < str1.size() - n + 1; ++i) {
        for (int j = 0; j < str2.size() - n + 1; ++j) {
            bool equal = true;
            for (int k = 0; k < n; ++k) {
                if (std::tolower(str1[i + k]) != std::tolower(str2[j + k])) {
                    equal = false;
                    break;
                }
            }

            if (equal) { return true; }
        }
    }

    return false;
}
//目标过线时间判断逻辑
void TargetTimeJudge_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_result_(frame);
        return;
    }
    // output_result_(frame);
    // return;

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
    auto &back_ext_info = clone_frame->frame_info->ext_info.back();
    //目标消失
    if (target_disappear_) {
        int status = 0;//正常情况:门关/门开有人
        int flag = 0;
        int pflag = 0;
        for (const auto &[_, item] : back_ext_info.map_target_box) {
            if (back_ext_info.map_class_label.at(item.class_id) == "open") { flag++; }//"open"
            if (back_ext_info.map_class_label.at(item.class_id) == label_) { pflag++; }
        }
        if (flag > 0 && pflag == 0) {//门开无人
            status = 1;              //异常情况
        }

        if (time_sum_.size() > clone_frame->infer_frame_rate) {
            time_sum_.pop_front();
            time_sum_.push_back(status);
        } else {
            time_sum_.push_back(status);
        }

        if (std::accumulate(time_sum_.cbegin(), time_sum_.cend(), 0) > frame->infer_frame_rate / 2) {//当前帧异常
            count++;
        }
        if (pflag > 0) { count = 0; }
        if (count / clone_frame->infer_frame_rate > duration_time_ && pflag == 0) {
            count = 0;
            clone_frame->check_report_callback_ = [](const std::vector<FrameExtInfo> &) { return FrameType::kReport; };
        }
        // output_result_(clone_frame);
    }
    //目标停留
    if (target_retention_) {

        int flag = 0;
        for (const auto &[_, item] : back_ext_info.map_target_box) {
            if (back_ext_info.map_class_label.at(item.class_id) == label_) {//检测到车牌
                time_sum_.push_back(get_min());
                cache_frame_ = clone_frame;
            }
        }
        auto result = frame->check_report_callback_(frame->frame_info->ext_info);
        if (result >= FrameType::kReport) {
            
            int appear = time_sum_.front();
            if (get_min() - appear > duration_time_) {
                if (cache_frame_) {
                    flag++;
                    cache_frame_->check_report_callback_ = [](const std::vector<FrameExtInfo> &) {
                        return FrameType::kReport;
                    };
                }
                // output_result_(cache_frame_);
            }
            time_sum_.clear();
        }
        if (flag > 0) { clone_frame = cache_frame_; }//保证检测到车牌}
    }
    output_result_(clone_frame);
}

}// namespace nodes
}// namespace gddi
