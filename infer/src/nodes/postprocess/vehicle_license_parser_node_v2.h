/**
 * @file vehicle_license_parser_node_v2.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-07-20
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
#include <json.hpp>
#include <map>
#include <string>
#include <vector>
#include <regex>

namespace gddi {
namespace nodes {

class VehicleLicenseParser_v2 : public node_any_basic<VehicleLicenseParser_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit VehicleLicenseParser_v2(std::string name) : node_any_basic(std::move(name)) {
        register_input_message_handler_(&VehicleLicenseParser_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~VehicleLicenseParser_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

    void find_front_overlap(const FrameExtInfo &ext_info);
    void find_back_overlap(const FrameExtInfo &ext_info);

private:
    uint32_t index = 0;
    std::string front_box_path_{"config/front_box.json"};
    std::string back_box_path_{"config/back_box.json"};

    std::map<std::string, std::vector<float>> front_box_map_;
    std::map<std::string, std::vector<float>> back_box_map_;

    nlohmann::json metadata_;

    std::regex regex_;
};

}// namespace nodes
}// namespace gddi