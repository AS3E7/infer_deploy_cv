/**
 * @file vehicle_license_spliter_node_v2.h
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

class VehicleLicenseSpliter_v2 : public node_any_basic<VehicleLicenseSpliter_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit VehicleLicenseSpliter_v2(std::string name) : node_any_basic(std::move(name)) {
        register_input_message_handler_(&VehicleLicenseSpliter_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~VehicleLicenseSpliter_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

    std::vector<cv::Mat> split_font_back_img(const cv::Mat &image,
                                             const std::vector<std::vector<float>> &vec_key_points);

private:
    std::string front_mask_path_{"config/front_premask_v2.json"};
    std::string back_mask_path_{"config/back_premask_v2.json"};

    std::vector<std::vector<cv::Point>> front_mask_points_;
    std::vector<std::vector<cv::Point>> back_mask_points_;
};

}// namespace nodes
}// namespace gddi