/**
 * @file connected_components_node_v2.h
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
#include "utils.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace gddi {
namespace nodes {

class ConnectedComponents_v2 : public node_any_basic<ConnectedComponents_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit ConnectedComponents_v2(std::string name) : node_any_basic(std::move(name)) {
        bind_simple_property("area_threshold", area_threshold_, "面积阈值");
        bind_simple_property("aspect_ratio_threshold", aspect_ratio_threshold_,
                             "长宽比阈值");// 如果选择了多边形区域，这个参数无效

        bind_simple_property("polygon", polygon_, "多边形区域");                    // 是否输出多边形区域
        bind_simple_property("polygon_threshold", polygon_threshold_, "多边形阈值");// 阈值越大边越少

        bind_simple_property("distance_threshold", distance_threshold_, "连通域合并距离阈值");

        register_input_message_handler_(&ConnectedComponents_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~ConnectedComponents_v2() = default;

protected:
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    // 过滤阈值
    float area_threshold_{0};
    float aspect_ratio_threshold_{0};

    // 多边形区域
    bool polygon_{false};
    float polygon_threshold_{0.02};

    float distance_threshold_{100};
};

}// namespace nodes
}// namespace gddi