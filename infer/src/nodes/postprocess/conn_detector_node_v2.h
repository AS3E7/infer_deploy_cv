/**
 * @file conn_detector_node_v2.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-07-19
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#pragma once

#include "message_templates.hpp"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"
#include <chrono>
#include <cstdint>
#include <map>
#include <vector>

namespace gddi {
namespace nodes {

class ConnDetector_v2 : public node_any_basic<ConnDetector_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit ConnDetector_v2(std::string name) : node_any_basic(std::move(name)) {
        register_input_message_handler_(&ConnDetector_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~ConnDetector_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    float unclip_ratio_ = 1.5;

    std::string front_title_file_{"config/front_title.json"};
    std::string back_title_file_{"config/back_title.json"};

    uint32_t g_index_;
    cv::Mat front_ingore_mask_;
    cv::Mat back_ingore_mask_;
};

}// namespace nodes
}// namespace gddi