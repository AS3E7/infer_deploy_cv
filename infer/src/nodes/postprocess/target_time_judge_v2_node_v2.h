/**
 * @file target_time_judge_v2_node_v2.h
 * @author Minwell
 * @brief 
 * @version 0.1
 * @date 2023-09-21
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#pragma once

#include "json.hpp"
#include "message_templates.hpp"
#include "modules/postprocess/cross_border.h"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"
#include <array>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gddi {
namespace nodes {

class TargetTimeJudge_v2 : public node_any_basic<TargetTimeJudge_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit TargetTimeJudge_v2(std::string name) : node_any_basic(std::move(name)) {

        bind_simple_property("label", label_, "标签");
        bind_simple_property("duration_time", duration_time_, "持续时长");
        bind_simple_property("target_retention", target_retention_, "目标停留");
        bind_simple_property("target_disappear", target_disappear_, "目标消失");

        register_input_message_handler_(&TargetTimeJudge_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~TargetTimeJudge_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    bool target_retention_{true};
    bool target_disappear_{false};
    std::string label_;
    int duration_time_{0};
    int count = 0;
    std::list<int> time_sum_;
    std::shared_ptr<msgs::cv_frame> cache_frame_;
};

}// namespace nodes
}// namespace gddi
