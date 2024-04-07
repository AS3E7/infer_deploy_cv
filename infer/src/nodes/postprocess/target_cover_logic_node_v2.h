/**
 * @file rect_cover_logic_node_v2.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-08-16
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#pragma once

#include "exprtk.hpp"
#include "message_templates.hpp"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"
#include <vector>

namespace gddi {
namespace nodes {

typedef exprtk::symbol_table<float> symbol_table_t;
typedef exprtk::expression<float> expression_t;
typedef exprtk::parser<float> parser_t;

class TargetCoverLogic_v2 : public node_any_basic<TargetCoverLogic_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit TargetCoverLogic_v2(std::string name) : node_any_basic(std::move(name)) {
        bind_simple_property("main_label", main_label_, "主标签");
        bind_simple_property("iou_threshold", iou_threshold_, "重叠阈值");
        bind_simple_property("expression", str_expression_, "表达式");

        register_input_message_handler_(&TargetCoverLogic_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~TargetCoverLogic_v2() = default;

protected:
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

    bool parse_expression(const std::string &expression);

private:
    std::set<std::string> main_label_;
    float iou_threshold_{0.6};
    std::string str_expression_;
    std::unordered_map<std::string, float> variables_;

    bool is_valid_expression_{false};
    symbol_table_t symbol_table_;
    expression_t expression_;

    std::shared_ptr<msgs::cv_frame> last_one_frame_;
    std::shared_ptr<msgs::cv_frame> last_zero_frame_;

    std::vector<int> event_group_;
    int last_group_status_{0};
};

}// namespace nodes
}// namespace gddi