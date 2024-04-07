/**
 * @file cross_stay_node_v2.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-08-30
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
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gddi {
namespace nodes {

class LabelStay_v2 : public node_any_basic<LabelStay_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit LabelStay_v2(std::string name) : node_any_basic(std::move(name)) {
        bind_simple_property("label", label_, "标签名称");
        bind_simple_property("stay_time", stay_time_, "停留时间");
        
        register_input_message_handler_(&LabelStay_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~LabelStay_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    std::string label_;
    int stay_time_;

    std::map<int, int> stay_count_;
};

}// namespace nodes
}// namespace gddi
