#include "aggregation_node_v2.h"
#include "modules/cvrelate/geometry.h"

namespace gddi {
namespace nodes {

int find_best_match(const BoxInfo &target_boxinfo, std::map<int, BoxInfo> &cache_boxinfo, const float threshold) {
    auto target_points =
        std::vector<Point2i>({{(int)target_boxinfo.box.x, (int)target_boxinfo.box.y},
                              {(int)(target_boxinfo.box.x + target_boxinfo.box.width), (int)target_boxinfo.box.y},
                              {(int)(target_boxinfo.box.x + target_boxinfo.box.width),
                               (int)(target_boxinfo.box.y + target_boxinfo.box.height)},
                              {(int)target_boxinfo.box.x, (int)(target_boxinfo.box.y + target_boxinfo.box.height)}});

    int best_match_id = -1;
    float best_match_rate = 0.0;
    for (auto &[target_id, box_info] : cache_boxinfo) {
        auto match_points = std::vector<Point2i>(
            {{(int)box_info.box.x, (int)box_info.box.y},
             {(int)(box_info.box.x + box_info.box.width), (int)box_info.box.y},
             {(int)(box_info.box.x + box_info.box.width), (int)(box_info.box.y + box_info.box.height)},
             {(int)box_info.box.x, (int)(box_info.box.y + box_info.box.height)}});
        auto match_rate = geometry::area_cover_rate(target_points, match_points);
        if (match_rate >= threshold && match_rate > best_match_rate) {
            best_match_id = target_id;
            best_match_rate = match_rate;
        }
    }

    return best_match_id;
}

void Aggregation_v2::on_setup() { input_endpoint_count_ = get_input_endpoint_count(); }

void Aggregation_v2::on_cv_image_(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_image_(frame);
        return;
    }

    if (cache_frame_info_[frame->frame_info->video_frame_idx].size() < input_endpoint_count_ - 1) {
        cache_frame_info_[frame->frame_info->video_frame_idx].emplace_back(frame->frame_info);
        return;
    }

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);

    if (merge_same_) {
        if (clone_frame->frame_info->ext_info.back().algo_type
            < cache_frame_info_[clone_frame->frame_info->video_frame_idx].back()->ext_info.back().algo_type) {
            clone_frame->frame_info = cache_frame_info_[clone_frame->frame_info->video_frame_idx].back();
            cache_frame_info_[clone_frame->frame_info->video_frame_idx].back() = frame->frame_info;
        }
        auto &first_ext_info = clone_frame->frame_info->ext_info.back();
        auto &second_ext_info = cache_frame_info_[clone_frame->frame_info->video_frame_idx].back()->ext_info.back();

        // 处理多个模型的情况
        // for (auto &item : cache_frame_info_[clone_frame->frame_info->video_frame_idx]) {
        //     spdlog::info("item->algo_type: {}, back_ext_info.algo_type: {}", item->ext_info.back().algo_type, back_ext_info.algo_type);
        //     if (item->ext_info.back().algo_type > back_ext_info.algo_type) {
        //         std::swap(back_ext_info, item->ext_info.back());
        //     }
        // }

        // 目前只处理两个模型的情况
        if (cache_frame_info_.size() == 1) {
            auto class_index = first_ext_info.map_class_label.size();
            for (auto &[class_id, label] : second_ext_info.map_class_label) {
                first_ext_info.map_class_label[class_id + class_index] = label;
                first_ext_info.map_class_color[class_id + class_index] = second_ext_info.map_class_color.at(class_id);
            }

            for (auto iter = first_ext_info.map_target_box.begin(); iter != first_ext_info.map_target_box.end();) {
                auto back_target_points = std::vector<Point2i>(
                    {{(int)iter->second.box.x, (int)iter->second.box.y},
                     {(int)(iter->second.box.x + iter->second.box.width), (int)iter->second.box.y},
                     {(int)(iter->second.box.x + iter->second.box.width),
                      (int)(iter->second.box.y + iter->second.box.height)},
                     {(int)iter->second.box.x, (int)(iter->second.box.y + iter->second.box.height)}});

                int best_match_id = find_best_match(iter->second, second_ext_info.map_target_box, threshold_);
                if (best_match_id >= 0) {
                    iter->second.box = second_ext_info.map_target_box.at(best_match_id).box;
                    iter->second.class_id = second_ext_info.map_target_box.at(best_match_id).class_id + class_index;
                    ++iter;
                } else {
                    if (first_ext_info.algo_type == AlgoType::kPose) {
                        first_ext_info.map_key_points.erase(iter->first);
                    }
                    iter = first_ext_info.map_target_box.erase(iter);
                }
            }

            auto target_index =
                std::max_element(first_ext_info.map_target_box.begin(), first_ext_info.map_target_box.end(),
                                 [](const auto &a, const auto &b) { return a.first < b.first; })
                    ->first;
            for (auto &[target_id, item] : second_ext_info.map_key_points) {
                second_ext_info.map_target_box.at(target_id).class_id += class_index;
                first_ext_info.map_target_box[target_index] = second_ext_info.map_target_box.at(target_id);
                first_ext_info.map_key_points[target_index] = item;
                ++target_index;
            }
        }
    } else {
        auto &back_ext_info = clone_frame->frame_info->ext_info.back();
        for (auto &frame_info : cache_frame_info_[clone_frame->frame_info->video_frame_idx]) {
            size_t target_index = back_ext_info.map_target_box.size();
            size_t class_index = back_ext_info.map_class_label.size();

            for (auto &[target_id, box_info] : frame_info->ext_info.back().map_target_box) {
                box_info.class_id += class_index;
                back_ext_info.map_target_box[target_index] = box_info;

                if (frame_info->ext_info.back().map_key_points.count(target_id) > 0) {
                    back_ext_info.map_key_points[target_index] =
                        frame_info->ext_info.back().map_key_points.at(target_id);
                }

                ++target_index;
            }
            for (const auto &[key, value] : frame_info->ext_info.back().map_class_label) {
                back_ext_info.map_class_label[key + class_index] = value;
            }
            for (const auto &[key, value] : frame_info->ext_info.back().map_class_color) {
                back_ext_info.map_class_color[key + class_index] = value;
            }

            for (const auto &region : frame_info->roi_points) {
                auto iter = clone_frame->frame_info->roi_points.begin();
                for (; iter != clone_frame->frame_info->roi_points.end(); ++iter) {
                    if (geometry::has_intersection(iter->second, region.second)) { break; }
                }
                if (iter != clone_frame->frame_info->roi_points.end()) {
                    iter->second = geometry::merge_region(iter->second, region.second);
                } else {
                    clone_frame->frame_info->roi_points.insert(region);
                }
            }
        }
    }

    cache_frame_info_.erase(clone_frame->frame_info->video_frame_idx);

    if (only_most_match_) {
        auto &back_ext_info = clone_frame->frame_info->ext_info.back();

        std::pair<int, BoxInfo> match_target{-1, BoxInfo{0, 0, 0.0}};
        for (const auto &[target_id, box_info] : back_ext_info.map_target_box) {
            if (box_info.prob > match_target.second.prob) {
                match_target.first = target_id;
                match_target.second = box_info;
            }
        }

        if (match_target.first != -1) {
            back_ext_info.map_target_box.clear();
            back_ext_info.map_target_box.insert(match_target);
        }
    }

    // // 打印 map_target_box 成员信息
    // for (const auto &item : back_ext_info.map_target_box) {
    //     spdlog::info("map_target_box: {}, {}", item.first, item.second.class_id);
    // }

    // // 打印 map_key_points 成员信息
    // for (const auto &item : back_ext_info.map_key_points) {
    //     spdlog::info("map_key_points: {}, {}", item.first, item.second.size());
    // }

    // for (const auto &[key, value] : back_ext_info.map_class_label) {
    //     spdlog::info("map_class_label: {}, {}", key, value);
    // }

    // if (back_ext_info.map_target_box.size() > 0) {
    //     spdlog::info("============================================================");
    // }

    output_image_(clone_frame);
}

}// namespace nodes
}// namespace gddi