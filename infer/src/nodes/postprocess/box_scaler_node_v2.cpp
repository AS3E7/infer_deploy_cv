#include "box_scaler_node_v2.h"

namespace gddi {
namespace nodes {

void BoxScaler_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_result_(frame);
        return;
    }

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
    auto &back_ext_info = clone_frame->frame_info->ext_info.back();

    for (auto &[target_id, item] : back_ext_info.map_target_box) {
        auto center = cv::Point2f(item.box.x + item.box.width / 2, item.box.y + item.box.height / 2);
        item.box.x = (item.box.x - center.x) * scaler_ + center.x;
        item.box.y = (item.box.y - center.y) * scaler_ + center.y;
        item.box.width *= scaler_;
        item.box.height *= scaler_;

        if (back_ext_info.map_key_points.count(target_id) > 0) {
            // 计算所有关键点的中心
            cv::Point2f center_key_points;
            for (auto &key_point : back_ext_info.map_key_points.at(target_id)) {
                center_key_points.x += key_point.x;
                center_key_points.y += key_point.y;
            }
            center_key_points.x /= back_ext_info.map_key_points.at(target_id).size();
            center_key_points.y /= back_ext_info.map_key_points.at(target_id).size();

            for (auto &key_point : back_ext_info.map_key_points.at(target_id)) {
                key_point.x = (key_point.x - center_key_points.x) * scaler_ + center_key_points.x;
                key_point.y = (key_point.y - center_key_points.y) * scaler_ + center_key_points.y;
            }
        }
    }

    output_result_(clone_frame);
}

}// namespace nodes
}// namespace gddi
