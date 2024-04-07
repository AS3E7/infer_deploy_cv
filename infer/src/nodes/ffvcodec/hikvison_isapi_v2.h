/**
 * @file hikvison_isapi_v2.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-05-22
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#ifndef HIKVISON_ISAPI_V2_HPP
#define HIKVISON_ISAPI_V2_HPP

#include "message_templates.hpp"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"
#include <cstdint>
#include <curl/curl.h>
#include <map>
#include <memory>

namespace gddi {
namespace nodes {

class HikvisonISAPI_v2 : public node_any_basic<HikvisonISAPI_v2> {
private:
    message_pipe<msgs::cv_frame> output_image_;

public:
    explicit HikvisonISAPI_v2(std::string name) : node_any_basic(std::move(name)) {
        output_image_ = register_output_message_<msgs::cv_frame>();

        bind_simple_property("task_name", task_name_, ngraph::PropAccess::kPrivate);
        bind_simple_property("camera_ip", camera_ip_, "摄像头IP");
        bind_simple_property("username:password", username_password_, "用户名:密码");
        bind_simple_property("presets", str_presets_, "预置点信息");
        bind_simple_property("staing_time", staing_time_, "预置点停留时间(s)");
    }

    ~HikvisonISAPI_v2();

protected:
    void on_setup() override;

private:
    std::string task_name_;
    std::string camera_ip_;
    std::string username_password_;
    std::string str_presets_;
    uint32_t staing_time_ {5};

    std::map<std::string, std::vector<std::vector<int>>> presets_;

    int64_t frame_idx_ = 0;
    ImagePool mem_pool_;

    std::shared_ptr<CURL> curl_;
};
}// namespace nodes

}// namespace gddi

#endif//HIKVISON_ISAPI_V2_HPP
