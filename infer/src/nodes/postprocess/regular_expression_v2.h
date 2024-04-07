/**
 * @file regular_expression_v2.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#pragma once

#include "message_templates.hpp"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include <regex>

namespace gddi {
namespace nodes {
class RegularExpression_v2 : public node_any_basic<RegularExpression_v2> {
private:
    message_pipe<msgs::cv_frame> output_image_;

public:
    explicit RegularExpression_v2(std::string name) : node_any_basic(std::move(name)) {
        bind_simple_property("expression", str_expression_, "表达式");

        bind_simple_flags("support_preview", true);

        register_input_message_handler_(&RegularExpression_v2::on_cv_image, this);
        output_image_ = register_output_message_<msgs::cv_frame>();
    }
    ~RegularExpression_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    std::string str_expression_;

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter_;
    std::wregex pattern_;
};
}// namespace nodes
}// namespace gddi