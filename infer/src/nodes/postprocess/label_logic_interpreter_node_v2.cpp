#include "label_logic_interpreter_node_v2.h"
#include "spdlog/spdlog.h"
#include <regex>

namespace gddi {
namespace nodes {

void LabelLogicInterpreter_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    auto &back_ext_info = frame->frame_info->ext_info.back();

    if (!is_valid_expression_) {
        expression_.register_symbol_table(symbol_table_);
        for (auto &[_, value] : back_ext_info.map_class_label) { variables_[utils::strim_str(value)] = 0; }
        for (auto &[key, value] : variables_) { symbol_table_.add_variable(key.c_str(), value); }
        parser_t parser;
        if (!parser.compile(str_expression_, expression_)) {
            quit_runner_(TaskErrorCode::kInvalidArgument);
            return;
        }
        is_valid_expression_ = true;
    }

    // 一阶段全局目标
    if (frame->frame_info->ext_info.size() == 1) {
        for (const auto &[_, bbox] : back_ext_info.map_target_box) {
            if (variables_.count(back_ext_info.map_class_label.at(bbox.class_id)) > 0) {
                ++variables_[back_ext_info.map_class_label.at(bbox.class_id)];
            }
        }
    } else {
        // 多阶段取前一阶段同个目标的目标框
        std::map<int, std::vector<BoxInfo>> target_box;
        for (const auto &[_, bbox] : back_ext_info.map_target_box) { target_box[bbox.prev_id].emplace_back(bbox); }

        for (const auto &[_, bboxs] : target_box) {
            for (const auto &bbox : bboxs) {
                if (variables_.count(back_ext_info.map_class_label.at(bbox.class_id)) > 0) {
                    ++variables_[back_ext_info.map_class_label.at(bbox.class_id)];
                }
            }
        }
    }

    for (const auto &[_, item] : back_ext_info.map_target_box) {
        spdlog::debug("class_id: {}, class_label: {}, prob: {}, prev_id: {}, bbox: [{}, {}, {}, {}]", item.class_id,
                      back_ext_info.map_class_label.at(item.class_id), item.prob, item.prev_id, item.box.x, item.box.y,
                      item.box.width, item.box.height);
    }

    auto expression_result = expression_.value();

    if (map_label_) {
        float avg_prob = 1;
        if (!back_ext_info.map_target_box.empty()) {
            avg_prob = back_ext_info.map_target_box.begin()->second.prob;

            for (const auto &[_, bbox] : back_ext_info.map_target_box) { avg_prob = (avg_prob + bbox.prob) / 2; }
        }

        back_ext_info.map_target_box.clear();
        back_ext_info.map_class_label.clear();
        back_ext_info.map_class_color.clear();

        if (expression_result && !expression_true_.empty()) {
            back_ext_info.map_class_label[0] = expression_true_;
            back_ext_info.map_class_color[0] = {255, 0, 0, 255};
            back_ext_info.map_target_box[0] =
                BoxInfo{0, 0, avg_prob, Rect2f{0, 0, frame->frame_info->width(), frame->frame_info->height()}, "", 0};
        } else if (!expression_result && !expression_false_.empty()) {
            back_ext_info.map_class_label[0] = expression_false_;
            back_ext_info.map_class_color[0] = {0, 0, 255, 255};
            back_ext_info.map_target_box[0] =
                BoxInfo{0, 0, 0, Rect2f{0, 0, frame->frame_info->width(), frame->frame_info->height()}, "", 0};
        }
    }

    frame->frame_info->frame_event_result = expression_result;

    for (auto &[key, value] : variables_) { value = 0; }

    output_result_(frame);
}

}// namespace nodes
}// namespace gddi