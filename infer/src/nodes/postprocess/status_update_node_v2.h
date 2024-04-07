/**
 * @file status_update_node_v2.h
 * @author Minwell
 * @brief 
 * @version 0.1
 * @date 2023-09-04
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
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

namespace gddi {
namespace nodes {

class StatusUpdate_v2 : public node_any_basic<StatusUpdate_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit StatusUpdate_v2(std::string name) : node_any_basic(std::move(name)) {

        bind_simple_property("iou_threshold", iou_threshold_, "交并比");
        bind_simple_property("angle_threshold", angle__threshold_, "夹角度数");
        bind_simple_property("label", label_, "标签");

        register_input_message_handler_(&StatusUpdate_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~StatusUpdate_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    std::shared_ptr<msgs::cv_frame> cache_frame_;
    std::string label_;
    float iou_threshold_{0};
    float angle__threshold_{0};
};

}// namespace nodes
}// namespace gddi
