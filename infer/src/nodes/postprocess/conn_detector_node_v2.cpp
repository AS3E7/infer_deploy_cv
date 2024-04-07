#include "conn_detector_node_v2.h"
#include "node_struct_def.h"
#include <algorithm>
#include <clipper2/clipper.h>
#include <clipper2/clipper.offset.h>
#include <cstdint>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))

#define DIVISED_RATIO 3

namespace gddi {
namespace nodes {

float BoxScoreFast(const cv::Mat &bitmap, cv::RotatedRect _box) {
    int h = bitmap.rows;
    int w = bitmap.cols;

    // Convert RotatedRect to a normal Rect for simplicity
    cv::Rect box = _box.boundingRect();

    // Ensure box is within the image
    box.x = std::max(0, std::min(box.x, w - 1));
    box.y = std::max(0, std::min(box.y, h - 1));
    box.width = std::max(0, std::min(box.width, w - box.x));
    box.height = std::max(0, std::min(box.height, h - box.y));

    // Create mask
    cv::Mat mask = cv::Mat::zeros(box.height, box.width, CV_8UC1);

    // Create polygon from RotatedRect
    cv::Point2f vertices[4];
    _box.points(vertices);
    std::vector<std::vector<cv::Point>> pts;
    std::vector<cv::Point> pt(4);
    for (int i = 0; i < 4; ++i) {
        pt[i].x = static_cast<int>(vertices[i].x - box.x);
        pt[i].y = static_cast<int>(vertices[i].y - box.y);
    }
    pts.push_back(pt);

    // Fill polygon in mask
    cv::fillPoly(mask, pts, cv::Scalar(1));

    // Compute mean with mask
    cv::Scalar mean = cv::mean(bitmap(box), mask);

    return static_cast<float>(mean[0]);
}

float GetPerimeter(const cv::RotatedRect &rotated_rect) {
    return 2 * (rotated_rect.size.width + rotated_rect.size.height);
}

double PolygonArea(const std::vector<cv::Point> &contour) { return cv::contourArea(contour); }

bool isInside(const cv::RotatedRect &rect, const cv::Size &size) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; ++i) {
        if (vertices[i].x < 0 || vertices[i].y < 0 || vertices[i].x >= size.width || vertices[i].y >= size.height) {
            return false;
        }
    }
    return true;
}

cv::RotatedRect Unclip(const cv::RotatedRect &box, const float unclip_ratio, const cv::Size &max_size) {
    cv::RotatedRect rotated_bbox = box;
    std::vector<cv::Point2f> pts;
    pts.resize(4);
    rotated_bbox.points(pts.data());

    std::vector<cv::Point> pts_int;
    for (auto point : pts) { pts_int.emplace_back(cv::Point{(int)point.x, (int)point.y}); }

    // float max_length = GetMaxLengthInContours(pts_int);
    float perimeter = GetPerimeter(rotated_bbox);
    float area = PolygonArea(pts_int);
    double distance = area * unclip_ratio / perimeter;

    std::vector<Clipper2Lib::Point<double>> subj;
    for (auto point : pts) { subj.emplace_back(Clipper2Lib::Point<double>(point.x, point.y)); }

    Clipper2Lib::ClipperOffset co;
    co.AddPath(subj, Clipper2Lib::JoinType::Miter, Clipper2Lib::EndType::Polygon);
    auto solutions = co.Execute(distance);

    // If we have a valid expanded polygon
    if (!solutions.empty()) {
        // Convert the expanded polygon points back to cv::Point2f
        std::vector<cv::Point2f> expanded_pts;
        for (const auto &point : solutions[0]) {
            expanded_pts.emplace_back(cv::Point2f{(float)point.x, (float)point.y});
        }

        // Create a cv::RotatedRect from the expanded points
        cv::RotatedRect expanded_rect = cv::minAreaRect(expanded_pts);

        // Check if the expanded rectangle fits within max_size, if not, shrink it
        while (!isInside(expanded_rect, max_size)) {
            expanded_rect.size.width *= 0.99;
            expanded_rect.size.height *= 0.99;
        }
        return expanded_rect;

    } else {
        // If no valid expanded polygon, return the original box
        return box;
    }
}

int sort_point_v2(std::vector<cv::Point2f> &sort_point) {
    std::vector<cv::Point2f> new_sort_point;

    std::sort(sort_point.begin(), sort_point.end(), [](cv::Point2f a, cv::Point2f b) {
        if (a.y < b.y) { return true; }
        if (a.y == b.y) {
            if (a.x < b.x) return true;
        }
        return false;
    });

    std::sort(sort_point.begin(), sort_point.begin() + 2, [](cv::Point2f a, cv::Point2f b) {
        if (a.x < b.x) { return true; }
        if (a.x == b.x) {
            if (a.y > b.y) return true;
        }
        return false;
    });

    new_sort_point.emplace_back(sort_point[0]);

    new_sort_point.emplace_back(sort_point[1]);

    std::sort(sort_point.begin() + 2, sort_point.end(), [](cv::Point2f a, cv::Point2f b) {
        if (a.x > b.x) { return true; }
        if (a.x == b.x) {
            if (a.y > b.y) return true;
        }
        return false;
    });

    new_sort_point.emplace_back(sort_point[2]);

    new_sort_point.emplace_back(sort_point[3]);

    sort_point = new_sort_point;

    return 0;
}

int get_distance(cv::Point2f pointO, cv::Point2f point1) {
    int distance;
    distance = powf((pointO.x - point1.x), 2) + powf((pointO.y - point1.y), 2);
    distance = sqrtf(distance);
    return distance;
}

std::map<int, cv::Mat> extract_and_rotate_regions(const cv::Mat &img, const std::vector<cv::RotatedRect> &rects) {
    std::map<int, cv::Mat> rotatedMats;
    int index = 0;
    for (const auto &rect : rects) {
        std::vector<cv::Point2f> sort_point(4);
        rect.points(sort_point.data());
        sort_point_v2(sort_point);
        int distance_1 = get_distance(sort_point[0], sort_point[1]);
        int distance_2 = get_distance(sort_point[0], sort_point[3]);

        std::vector<cv::Point2f> src_points;
        src_points.resize(4);

        // if (sort_point[0].x < sort_point[3].x){
        //     src_points[0].x = sort_point[1].x;
        //     src_points[0].y = sort_point[1].y;
        //     src_points[1].x = sort_point[2].x;
        //     src_points[1].y = sort_point[2].y;
        //     src_points[2].x = sort_point[3].x;
        //     src_points[2].y = sort_point[3].y;
        //     src_points[3].x = sort_point[0].x;
        //     src_points[3].y = sort_point[0].y;
        // }else{  //2310
        src_points[0].x = sort_point[3].x;
        src_points[0].y = sort_point[3].y;
        src_points[1].x = sort_point[0].x;
        src_points[1].y = sort_point[0].y;
        src_points[2].x = sort_point[1].x;
        src_points[2].y = sort_point[1].y;
        src_points[3].x = sort_point[2].x;
        src_points[3].y = sort_point[2].y;
        // }
        std::vector<cv::Point2f> src_points_tmp = src_points;
        src_points[0].x = src_points_tmp[1].x;
        src_points[0].y = src_points_tmp[1].y;
        src_points[1].x = src_points_tmp[0].x;
        src_points[1].y = src_points_tmp[0].y;
        src_points[2].x = src_points_tmp[3].x;
        src_points[2].y = src_points_tmp[3].y;
        src_points[3].x = src_points_tmp[2].x;
        src_points[3].y = src_points_tmp[2].y;

        int dst_h = std::min(distance_1, distance_2);
        int dst_w = std::max(distance_1, distance_2);
        // printf("dst_h: %d, dst_w: %d\n", dst_h, dst_w);
        std::vector<cv::Point2f> dst_points = {{0.0f, 0.0f},
                                               {0.0f, (float)dst_h},
                                               {(float)dst_w, (float)dst_h},
                                               {(float)dst_w, 0.0f}};

        cv::Mat M = cv::getPerspectiveTransform(src_points, dst_points);
        cv::Mat coord_mat;
        cv::warpPerspective(img, coord_mat, M, cv::Size(dst_w, dst_h));
        rotatedMats[index++] = coord_mat;
    }
    return rotatedMats;
}

bool get_ingore_mask(std::ifstream &ifs, cv::Mat &mask) {
    auto front_title = nlohmann::json::parse(ifs);

    try {
        auto shape_list = front_title["shapes"];
        // auto image_width = front_title["imageWidth"].get<int>();
        // auto image_height = front_title["imageHeight"].get<int>();
        // mask = cv::Mat::ones(static_cast<int>(std::ceil(image_height / DIVISED_RATIO)),
        //                      static_cast<int>(std::ceil(image_width / DIVISED_RATIO)), CV_8UC1);
        mask = cv::Mat::ones(294, 440, CV_8UC1);

        spdlog::debug("mask_width: {}, mask_height: {}", mask.cols, mask.rows);

        for (const auto &item : shape_list) {
            std::vector<cv::Point> polygon;
            for (const auto &point : item["points"].get<std::vector<std::vector<float>>>()) {
                polygon.emplace_back(std::ceil(point[0] / DIVISED_RATIO), std::ceil(point[1] / DIVISED_RATIO));
            }
            std::vector<std::vector<cv::Point>> polygon_vector(1, polygon);
            cv::fillPoly(mask, polygon_vector, cv::Scalar(0));
        }
    } catch (std::exception &e) {
        spdlog::error("{}", e.what());
        return false;
    }

    return true;
}

void ConnDetector_v2::on_setup() {
    std::ifstream ifs(front_title_file_);
    if (!ifs.is_open()) {
        spdlog::error("{} not exist", front_title_file_);
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }

    if (!get_ingore_mask(ifs, front_ingore_mask_)) {
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }

    ifs = std::ifstream(back_title_file_);
    if (!ifs.is_open()) {
        spdlog::error("{} not exist", back_title_file_);
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }

    if (!get_ingore_mask(ifs, back_ingore_mask_)) {
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }

    // front_ingore_mask_ *= 255;
    // back_ingore_mask_ *= 255;

    // cv::imwrite("front_ingore_mask_.jpg", front_ingore_mask_);
    // cv::imwrite("back_ingore_mask_.jpg", back_ingore_mask_);
}

void ConnDetector_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    auto &last_ext_info = *(frame->frame_info->ext_info.crbegin() + 1);
    auto &back_ext_info = frame->frame_info->ext_info.back();

    cv::Mat ignore_mask;
    auto type = frame->frame_info->frame_meta["type"].get<std::string>();
    if (type == "front-back") {
        if (g_index_ == 0) {
            spdlog::debug("front");
            ignore_mask = front_ingore_mask_;
        } else {
            spdlog::debug("back");
            ignore_mask = back_ingore_mask_;
        }
        g_index_ = (g_index_ + 1) % 2;
    } else if (type == "front") {
        spdlog::debug("front");
        ignore_mask = front_ingore_mask_;
    } else if (type == "back") {
        spdlog::debug("back");
        ignore_mask = back_ingore_mask_;
    } else {
        output_result_(frame);
        return;
    }

    for (const auto &[idx, item] : back_ext_info.map_seg_info) {
        cv::Mat thresh_mat(item.seg_height, item.seg_width, CV_8UC1, const_cast<uint8_t *>(item.seg_map.data()));

        cv::Mat thresh_int8_mat;
        thresh_mat.convertTo(thresh_int8_mat, CV_8UC1, 255.0);

        // cv::Mat color_img;
        // cv::cvtColor(thresh_int8_mat, color_img, cv::COLOR_GRAY2BGR);
        // cv::imwrite("thresh_int8_mat.jpg", color_img);

        std::vector<std::vector<cv::Point>> contours;

        spdlog::debug("front_ingore_mask, width: {}, height: {}", front_ingore_mask_.cols, front_ingore_mask_.rows);
        spdlog::debug("thresh_int8_mat, width: {}, height: {}", thresh_int8_mat.cols, thresh_int8_mat.rows);

        cv::bitwise_and(thresh_int8_mat, ignore_mask, thresh_int8_mat);

        //水平方向膨胀
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(56, 4));
        cv::dilate(thresh_int8_mat, thresh_int8_mat, element);

        //腐蚀操作
        element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 2));
        cv::erode(thresh_int8_mat, thresh_int8_mat, element);

        // cv::imwrite("thresh_int8_mat" + std::to_string(g_index_) + ".jpg", thresh_int8_mat);

        cv::findContours(thresh_int8_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.size() > 1000) { contours.erase(contours.begin() + 1000, contours.end()); }

        int index = 0;
        std::vector<cv::RotatedRect> rects;
        for (auto contour : contours) {
            // 2. 找最小框
            cv::RotatedRect rotated_bbox = cv::minAreaRect(contour);//带旋转矩阵框
            // rotated_bbox.angle = std::clamp(rotated_bbox.angle, -95.0f, -85.0f);
            // 3. 计算box score
            float score = BoxScoreFast(thresh_mat, rotated_bbox);
            // 4. unclip
            // printf("contour score %f angel %f\n", __FILE__, __LINE__, score, rotated_bbox.angle);
            cv::RotatedRect res_rotated_bbox =
                Unclip(rotated_bbox, unclip_ratio_, cv::Size(item.seg_width, item.seg_height));

            rects.emplace_back(res_rotated_bbox);

            auto bbox = res_rotated_bbox.boundingRect2f();
            back_ext_info.map_target_box[index].box = {bbox.x, bbox.y, bbox.width, bbox.height};
            back_ext_info.map_target_box[index].prob = score;
            back_ext_info.crop_rects[index] = {bbox.x, bbox.y, bbox.width, bbox.height};

            ++index;
        }

        back_ext_info.crop_images = extract_and_rotate_regions(last_ext_info.crop_images.at(idx), rects);

        // static int count = 0;
        // for (const auto &item : rects) {
        //     auto bbox = item.boundingRect2f();
        //     cv::rectangle(last_ext_info.crop_images.at(0), bbox, cv::Scalar(0, 0, 255), 2);
        // }
        // cv::imwrite(std::to_string(g_index_) + "_crop_" + std::to_string(count++) + ".jpg",
        //             last_ext_info.crop_images.at(0));
    }

    output_result_(frame);
}

}// namespace nodes
}// namespace gddi
