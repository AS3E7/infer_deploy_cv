#include "target_cover_logic_node_v2.h"
#include "modules/cvrelate/geometry.h"
#include "node_struct_def.h"
#include "spdlog/spdlog.h"
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace gddi {
namespace nodes {

void TargetCoverLogic_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    auto &back_ext_info = frame->frame_info->ext_info.back();

    if (back_ext_info.map_target_box.empty()) {
        output_result_(frame);
        return;
    }

    if (!is_valid_expression_) {
        expression_.register_symbol_table(symbol_table_);
        for (auto &[_, value] : back_ext_info.map_class_label) { variables_[utils::strim_str(value)] = 0; }
        for (auto &[key, value] : variables_) { symbol_table_.add_variable(key.c_str(), value); }
        parser_t parser;
        if (!parser.compile(str_expression_, expression_)) {
            spdlog::error("Failed to compile expression: {}", str_expression_);
            quit_runner_(TaskErrorCode::kInvalidArgument);
            return;
        }
        is_valid_expression_ = true;
    }

    // 获取主标签和表达式中的变量
    std::map<int, std::vector<Point2i>> main_label_points;
    std::map<int, std::vector<Point2i>> other_label_points;
    for (const auto &[target_id, bbox] : back_ext_info.map_target_box) {
        if (main_label_.count(back_ext_info.map_class_label[bbox.class_id]) > 0) {
            main_label_points[target_id].emplace_back(int(bbox.box.x), int(bbox.box.y));
            main_label_points[target_id].emplace_back(int(bbox.box.x), int(bbox.box.y + bbox.box.height));
            main_label_points[target_id].emplace_back(int(bbox.box.x + bbox.box.width),
                                                      int(bbox.box.y + bbox.box.height));
            main_label_points[target_id].emplace_back(int(bbox.box.x + bbox.box.width), int(bbox.box.y));
        } else if (variables_.count(back_ext_info.map_class_label[bbox.class_id]) > 0) {
            other_label_points[target_id].emplace_back(int(bbox.box.x), int(bbox.box.y));
            other_label_points[target_id].emplace_back(int(bbox.box.x), int(bbox.box.y + bbox.box.height));
            other_label_points[target_id].emplace_back(int(bbox.box.x + bbox.box.width),
                                                       int(bbox.box.y + bbox.box.height));
            other_label_points[target_id].emplace_back(int(bbox.box.x + bbox.box.width), int(bbox.box.y));
        }
    }

    // 判断主标签是否覆盖其他标签
    std::map<int, std::vector<std::string>> main_contains_labels;
    for (const auto &[target_id, other_points] : other_label_points) {
        int most_max_cover_id = -1;
        float max_threshold = 0;
        for (const auto &[main_target_id, main_points] : main_label_points) {
            auto cur_threshold = geometry::area_cover_rate(other_points, main_points);
            if (cur_threshold >= iou_threshold_ && cur_threshold > max_threshold) {
                max_threshold = cur_threshold;
                most_max_cover_id = main_target_id;
            }
        }
        if (most_max_cover_id != -1) {
            auto class_id = back_ext_info.map_target_box.at(target_id).class_id;
            main_contains_labels[most_max_cover_id].emplace_back(back_ext_info.map_class_label.at(class_id));
        }
    }

    // 判断表达式是否成立
    for (const auto &[target_id, labels] : main_contains_labels) {
        for (const auto &label : labels) { ++variables_[label]; }

        // expression_.value() == 1 means the expression is true
        if (expression_.value() == 1) {
            frame->frame_info->frame_event_result = 1;
            break;
        }

        for (auto &[key, value] : variables_) { value = 0; }
    }

    for (auto &[key, value] : variables_) { value = 0; }

    output_result_(frame);
}

}// namespace nodes
}// namespace gddi