/**
 * @file box_scaler_node_v2.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-09-04
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#pragma once

#include "message_templates.hpp"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include <set>

namespace gddi {
namespace nodes {
class BoxScaler_v2 : public node_any_basic<BoxScaler_v2> {
protected:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit BoxScaler_v2(std::string name) : node_any_basic<BoxScaler_v2>(std::move(name)) {
        bind_simple_property("scaler", scaler_, "缩放比例");

        register_input_message_handler_(&BoxScaler_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }

    ~BoxScaler_v2() override = default;

private:
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    float scaler_ = 1;
};

}// namespace nodes
}// namespace gddi