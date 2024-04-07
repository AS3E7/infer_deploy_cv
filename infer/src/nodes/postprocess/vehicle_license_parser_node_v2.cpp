#include "vehicle_license_parser_node_v2.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <fstream>
#include <vector>

#define DIVISED_RATIO 3

namespace gddi {
namespace nodes {

float calculate_intersection_area(cv::Rect2f bbox1, cv::Rect2f bbox2) {
    // 计算两个矩形的交集区域
    cv::Rect2f intersection = bbox1 & bbox2;
    return intersection.area();// Calculate and return the area
}

std::map<std::string, std::vector<OcrInfo>>
find_highest_overlap(const std::map<std::string, std::vector<float>> &bbox_map, const FrameExtInfo &ext_info) {
    std::map<std::string, std::vector<OcrInfo>> results;

    for (const auto &[target_id, item] : ext_info.map_ocr_info) {
        float highest_overlap = 0;
        std::string highest_overlap_bbox_key;

        cv::Rect2f target_bbox{item.points[0].x, item.points[0].y, item.points[2].x - item.points[0].x,
                               item.points[2].y - item.points[0].y};

        for (const auto &[key, bbox_item] : bbox_map) {
            cv::Rect2f map_bbox{bbox_item[0], bbox_item[1], bbox_item[2], bbox_item[3]};

            // 计算重叠面积
            float intersection_area = calculate_intersection_area(target_bbox, map_bbox);

            // 计算矩形x和当前bbox的面积之和
            float union_area = target_bbox.area() + map_bbox.area() - intersection_area;

            // 计算重叠度
            float overlap = intersection_area / union_area;

            // 更新最高重叠度和对应的矩形
            if (overlap > highest_overlap) {
                highest_overlap = overlap;
                highest_overlap_bbox_key = key;
            }
        }

        results[highest_overlap_bbox_key].emplace_back(item);
    }

    for (auto &[key, value] : results) {
        std::sort(value.begin(), value.end(),
                  [](const OcrInfo &a, const OcrInfo &b) { return a.points[0].y < b.points[0].y; });
    }

    return results;
}

void VehicleLicenseParser_v2::on_setup() {
    // 解析正面框位置信息
    std::fstream front_box_file(front_box_path_);
    if (!front_box_file.is_open()) {
        spdlog::error("front_box_file is not exist");
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }
    // 解析文件内容
    auto font_box = nlohmann::json::parse(front_box_file);
    for (const auto &[key, value] : font_box.items()) {
        front_box_map_[key] = value.get<std::vector<float>>();
        for (auto &item : front_box_map_[key]) { item /= DIVISED_RATIO; }
    }

    // 解析反面框位置信息
    std::fstream back_box_file(back_box_path_);
    if (!back_box_file.is_open()) {
        spdlog::error("back_box_file is not exist");
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }
    auto back_box = nlohmann::json::parse(back_box_file);
    for (const auto &[key, value] : back_box.items()) {
        back_box_map_[key] = value.get<std::vector<float>>();
        for (auto &item : back_box_map_[key]) { item /= DIVISED_RATIO; }
    }

    // 正则表达式
    regex_ = std::regex("(19\\d{2}|20\\d{2})[0-9]{2}[0-9]{2}");
}

void VehicleLicenseParser_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    auto &back_ext_info = frame->frame_info->ext_info.back();

    std::map<std::string, std::vector<OcrInfo>> results;

    auto type = frame->frame_info->frame_meta["type"].get<std::string>();
    if (type == "front-back") {
        if (index == 0) {
            find_front_overlap(back_ext_info);
        } else if (index == 1) {
            find_back_overlap(back_ext_info);
            back_ext_info.metadata = metadata_;
            back_ext_info.map_target_box.clear();
            index = 0;
            metadata_.clear();
            output_result_(frame);
            return;
        }
    } else if (type == "front") {
        find_front_overlap(back_ext_info);
        back_ext_info.metadata = metadata_;
        back_ext_info.map_target_box.clear();
        index = 0;
        metadata_.clear();
        output_result_(frame);
        return;
    } else if (type == "back") {
        find_back_overlap(back_ext_info);
        back_ext_info.metadata = metadata_;
        back_ext_info.map_target_box.clear();
        index = 0;
        metadata_.clear();
        output_result_(frame);
        return;
    } else {
        output_result_(frame);
        return;
    }

    ++index;
}

void VehicleLicenseParser_v2::find_front_overlap(const FrameExtInfo &ext_info) {
    auto results = find_highest_overlap(front_box_map_, ext_info);
    for (const auto &item : front_box_map_) { metadata_["front"][item.first] = std::string(); }
    for (const auto &[key, ocr_info] : results) {
        for (const auto &item : ocr_info) {
            if (metadata_["front"].count(key) > 0) {
                if (key == "注册日期" || key == "发证日期") {
                    if (std::regex_match(item.labels[0].str, regex_)) {
                        // 格式化日期
                        std::string date = item.labels[0].str;
                        date.insert(4, "-");
                        date.insert(7, "-");
                        metadata_["front"][key] = date;
                    }
                } else {
                    metadata_["front"][key] = metadata_["front"][key].get<std::string>() + item.labels[0].str;
                }
            }
        }
    }
}

void VehicleLicenseParser_v2::find_back_overlap(const FrameExtInfo &ext_info) {
    auto results = find_highest_overlap(back_box_map_, ext_info);
    for (const auto &item : back_box_map_) { metadata_["back"][item.first] = std::string(); }
    for (const auto &[key, ocr_info] : results) {
        for (const auto &item : ocr_info) {
            if (metadata_["back"].count(key) > 0) {
                if (key == "检验记录") {
                    if (item.labels[0].str.size() > 5) {
                        metadata_["back"][key] = item.labels[0].str;
                    } else {
                        metadata_["back"]["燃油类型"] = item.labels[0].str;
                    }
                } else {
                    metadata_["back"][key] = metadata_["back"][key].get<std::string>() + item.labels[0].str;
                }
            }
        }
    }
}

}// namespace nodes
}// namespace gddi