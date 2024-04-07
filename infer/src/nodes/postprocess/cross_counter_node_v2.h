/**
 * @file cross_counter_node_v2.h
 * @author zhdotcai
 * @brief 
 * @version 0.1
 * @date 2022-12-19
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __CROSS_COUNTER_NODE_V2__
#define __CROSS_COUNTER_NODE_V2__

#include "json.hpp"
#include "message_templates.hpp"
#include "modules/postprocess/cross_border.h"
#include "node_any_basic.hpp"
#include "node_msg_def.h"
#include "utils.hpp"
#include <bits/types/time_t.h>
#include <map>
#include <memory>
#include <vector>

namespace gddi {
namespace nodes {

class CrossCounter_v2 : public node_any_basic<CrossCounter_v2> {
private:
    message_pipe<msgs::cv_frame> output_result_;

public:
    explicit CrossCounter_v2(std::string name) : node_any_basic(std::move(name)) {
        bind_simple_property("left_cross", left_cross_, "左侧穿过");
        bind_simple_property("right_cross", right_cross_, "右侧穿过");
        bind_simple_property("regions_with_label", regions_with_label_, "边界区域");
        bind_simple_property("margin", margin_, "边界宽度");
        bind_simple_property("point_pos", point_pos_, "越线位置");
        bind_simple_property("internal_time", interval_time_, "间隔时间");

        bind_simple_flags("support_preview", true);

        register_input_message_handler_(&CrossCounter_v2::on_cv_image, this);
        output_result_ = register_output_message_<msgs::cv_frame>();
    }
    ~CrossCounter_v2() = default;

protected:
    void on_setup() override;
    void on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame);

private:
    bool left_cross_{true};
    bool right_cross_{false};
    float margin_{0};
    int interval_time_{0};
    Position point_pos_{Position::kCenter};

    std::vector<std::vector<Point2i>> border_points_;
    std::vector<std::map<std::string, std::array<int, 2>>> cross_count_;// 边界 - 标签 - 数量
    std::map<std::string, std::vector<std::vector<float>>> regions_with_label_;

    std::unique_ptr<CrossBorder> cross_border_;

    std::set<int> tracked_cross_;
    std::time_t pre_time = std::time(nullptr);
};
}// namespace nodes
}// namespace gddi

#endif// __CROSS_COUNTER_NODE_V2__
