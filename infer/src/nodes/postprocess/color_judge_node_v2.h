/**
 * @file color_judge_node_v2.h
 * @author Minwell
 * @brief 
 * @version 0.1
 * @date 2023-10-12
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
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <vector>

namespace gddi {
namespace nodes {

class ColorJudge_v2 : public node_any_basic<ColorJudge_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit ColorJudge_v2(std::string name) : node_any_basic(std::move(name)) {

        bind_simple_property("label", label_, "标签");

        register_input_message_handler_(&ColorJudge_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~ColorJudge_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    std::string label_;
    
};

}// namespace nodes
}// namespace gddi
