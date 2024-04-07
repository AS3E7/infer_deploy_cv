#include "connected_components_node_v2.h"
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace gddi {
namespace nodes {

float calculate_distance(const cv::Vec4i &line1, const cv::Vec4i &line2) {
    cv::Point p1(line1[0], line1[1]);
    cv::Point p2(line1[2], line1[3]);
    cv::Point p3(line2[0], line2[1]);
    cv::Point p4(line2[2], line2[3]);

    // 计算两条直线的距离（欧氏距离）
    return cv::norm(p1 - p3);
}

bool find_intersection(const cv::Vec4i &line1, const cv::Vec4i &line2, cv::Point2f &intersection) {
    cv::Point2f p1(line1[0], line1[1]);
    cv::Point2f p2(line1[2], line1[3]);
    cv::Point2f q1(line2[0], line2[1]);
    cv::Point2f q2(line2[2], line2[3]);

    float d = (p1.x - p2.x) * (q1.y - q2.y) - (p1.y - p2.y) * (q1.x - q2.x);
    if (d == 0) {
        // Lines are parallel
        return false;
    }

    float t = ((q1.x - p1.x) * (q1.y - q2.y) - (q1.y - p1.y) * (q1.x - q2.x)) / d;

    intersection.x = p1.x + t * (p2.x - p1.x);
    intersection.y = p1.y + t * (p2.y - p1.y);

    return true;
}

void ConnectedComponents_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_result_(frame);
        return;
    }

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
    auto &back_ext_info = clone_frame->frame_info->ext_info.back();

    // cv::Mat bgr_image;
    // frame->frame_info->src_frame->data->download(bgr_image);

    int target_id = 0;
    for (auto &[_, seg_info] : back_ext_info.map_seg_info) {
        for (const auto &[class_id, label] : back_ext_info.map_class_label) {
            if (class_id == 0) { continue; }

            auto binary_mask = cv::Mat(seg_info.seg_height, seg_info.seg_width, CV_8UC1, cv::Scalar(0));
            for (int i = 0; i < seg_info.seg_height; i++) {
                for (int j = 0; j < seg_info.seg_width; j++) {
                    if (seg_info.seg_map[i * seg_info.seg_width + j] == class_id) { binary_mask.at<uchar>(i, j) = 255; }
                }
            }

            // // 应用高斯滤波
            // int kernel_size = 5;
            // cv::GaussianBlur(binary_mask, binary_mask, cv::Size(kernel_size, kernel_size), 0, 0);

            int kernelSize = 5;
            auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
            cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN, kernel);

            // if (class_id == 3) { cv::imwrite("binary.png", binary_mask); }

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            if (polygon_) {
                for (auto contour : contours) {
                    double area = fabs(cv::contourArea(contour));
                    if (area < area_threshold_) { continue; }

                    std::vector<cv::Point> approx;
                    cv::approxPolyDP(contour, approx, polygon_threshold_ * cv::arcLength(contour, true), true);

                    auto bounding_rect = cv::boundingRect(approx);
                    back_ext_info.map_target_box[target_id].box = {std::max(0, bounding_rect.x),
                                                                   std::max(0, bounding_rect.y), bounding_rect.width,
                                                                   bounding_rect.height};
                    back_ext_info.map_target_box[target_id].class_id = class_id;

                    auto approx_size = approx.size();
                    for (int i = 0; i < approx_size; i++) {
                        back_ext_info.map_key_points[target_id].emplace_back(
                            PoseKeyPoint{i, approx[i].x, approx[i].y, 1});
                    }

                    // auto color = back_ext_info.map_class_color.at(class_id);
                    // cv::drawContours(bgr_image, std::vector<std::vector<cv::Point>>{approx}, 0,
                    //                  cv::Scalar(color.b, color.g, color.r), 2);

                    ++target_id;
                }
            } else {
                std::vector<cv::RotatedRect> rotated_rects;
                for (const auto contour : contours) {
                    double area = fabs(cv::contourArea(contour));
                    if (area < area_threshold_) { continue; }
                    auto rect = cv::minAreaRect(contour);
                    if (rect.size.width > rect.size.height) {
                        rect.angle = 90 + rect.angle;
                        std::swap(rect.size.width, rect.size.height);

                        if (rect.angle < 0) { rect.angle += 180; }
                    }
                    rotated_rects.emplace_back(rect);
                }

                if (class_id == 3) {
                    std::vector<float> areas;
                    for (const auto &rect : rotated_rects) { areas.emplace_back(rect.size.area()); }
                    std::sort(areas.begin(), areas.end());

                    // 移除与中位数差距过大的矩形
                    if (areas.size() > 0) {
                        auto midmedian = areas[areas.size() / 2];
                        auto iter = rotated_rects.begin();
                        while (iter != rotated_rects.end()) {
                            if (std::abs(iter->size.area() - midmedian) > midmedian * 0.8) {
                                iter = rotated_rects.erase(iter);
                            } else {
                                iter++;
                            }
                        }
                    }

                    // 合并同角度，且中心点距离较近的矩形
                    std::vector<std::vector<cv::RotatedRect>> grouped_rotated_rects;
                    for (const auto &item : rotated_rects) {
                        std::vector<cv::RotatedRect> group;
                        group.emplace_back(item);

                        auto iter = grouped_rotated_rects.begin();
                        while (iter != grouped_rotated_rects.end()) {
                            bool merge_flag = false;
                            for (const auto &rect : *iter) {
                                auto distance = cv::norm(item.center - rect.center);
                                auto height = std::min(item.size.width, rect.size.height);
                                if (std::abs(item.angle - rect.angle) < 10 && distance < height * 6) {
                                    merge_flag = true;
                                }
                            }

                            if (merge_flag) {
                                group.insert(group.end(), iter->begin(), iter->end());
                                iter = grouped_rotated_rects.erase(iter);
                            } else {
                                iter++;
                            }
                        }

                        grouped_rotated_rects.emplace_back(group);
                    }

                    rotated_rects.clear();
                    for (const auto &item : grouped_rotated_rects) {
                        std::vector<cv::Point2f> all_points;
                        for (const auto &rect : item) {
                            cv::Point2f points[4];
                            rect.points(points);
                            for (int i = 0; i < 4; ++i) { all_points.push_back(points[i]); }
                        }
                        rotated_rects.emplace_back(cv::minAreaRect(all_points));
                    }

                    // 移除面积过小的矩形
                    std::vector<float> areas2;
                    for (const auto &rect : rotated_rects) { areas2.emplace_back(rect.size.area()); }
                    auto average = std::accumulate(areas2.begin(), areas2.end(), 0.0) / areas2.size();
                    auto iter = rotated_rects.begin();
                    while (iter != rotated_rects.end()) {
                        if (iter->size.area() < average * 0.3) {
                            iter = rotated_rects.erase(iter);
                        } else {
                            iter++;
                        }
                    }

                    // for (const auto &item : rotated_rects) {
                    //     // 随机颜色
                    //     auto color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
                    //     cv::Point2f points[4];
                    //     item.points(points);
                    //     for (int i = 0; i < 4; i++) { cv::line(bgr_image, points[i], points[(i + 1) % 4], color, 2); }
                    // }
                } else if (aspect_ratio_threshold_ > 0) {
                    auto iter = rotated_rects.begin();
                    while (iter != rotated_rects.end()) {
                        if (iter->size.height > iter->size.width) {
                            if (iter->size.height / iter->size.width < aspect_ratio_threshold_) {
                                iter = rotated_rects.erase(iter);
                            } else {
                                iter++;
                            }
                        } else {
                            if (iter->size.width / iter->size.height < aspect_ratio_threshold_) {
                                iter = rotated_rects.erase(iter);
                            } else {
                                iter++;
                            }
                        }
                    }
                }

                auto color = back_ext_info.map_class_color.at(class_id);
                for (auto &rect : rotated_rects) {
                    cv::Point2f vertices[4];
                    rect.points(vertices);

                    std::vector<cv::Point2f> points;
                    for (int i = 0; i < 4; i++) { points.emplace_back(vertices[i]); }

                    auto bounding_rect = cv::boundingRect(points);
                    back_ext_info.map_target_box[target_id].box = {std::max(0, bounding_rect.x),
                                                                   std::max(0, bounding_rect.y), bounding_rect.width,
                                                                   bounding_rect.height};
                    back_ext_info.map_target_box[target_id].class_id = class_id;

                    for (int i = 0; i < 4; i++) {
                        if (rect.size.width > rect.size.height) {
                            back_ext_info.map_key_points[target_id].emplace_back(
                                PoseKeyPoint{i, vertices[i].x, vertices[i].y, 1});
                        } else {
                            back_ext_info.map_key_points[target_id].emplace_back(
                                PoseKeyPoint{i, vertices[(i + 1) % 4].x, vertices[(i + 1) % 4].y, 1});
                        }
                    }

                    // for (int i = 0; i < 4; i++) {
                    //     cv::line(bgr_image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(color.b, color.g, color.r),
                    //              2);
                    // }

                    ++target_id;
                }
            }
            // } else {
            // 边缘检测（可以使用Canny边缘检测或其他方法）
            // cv::Mat edges;
            // cv::Canny(binary_mask, edges, 50, 150);

            // // 霍夫直线变换
            // std::vector<cv::Vec4i> lines;
            // cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 20, 20, 30);

            // for (size_t i = 0; i < lines.size(); i++) {
            //     cv::Vec4i line = lines[i];
            //     auto color = back_ext_info.map_class_color.at(class_id);
            //     cv::line(bgr_image, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]),
            //              cv::Scalar(color.b, color.g, color.r), 2);
            // }
            // int kernelSize = 5;
            // auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
            // cv::dilate(binary_mask, binary_mask, kernel);

            // cv::imwrite("binary.png", binary_mask);

            //     std::vector<std::vector<cv::Point>> contours;
            //     cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            //     for (size_t i = 0; i < contours.size(); i++) {
            //         std::vector<cv::Point> approx;
            //         cv::approxPolyDP(contours[i], approx, 0.01 * cv::arcLength(contours[i], true), true);

            //         // 计算多边形的面积
            //         double area = fabs(cv::contourArea(contours[i]));
            //         if (area < area_threshold_) { continue; }

            //         auto rotated_bbox = cv::minAreaRect(approx);

            //         // 多边形的边数
            //         for (auto &item : approx) {
            //             frame->frame_info->frame_meta["con_comp"][std::to_string(class_id)].emplace_back(
            //                 std::map<std::string, float>{
            //                     {"x", item.x},
            //                     {"y", item.y},
            //                 });
            //         }

            //         auto color = back_ext_info.map_class_color.at(class_id);
            //         cv::drawContours(bgr_image, std::vector<std::vector<cv::Point>>{approx}, 0,
            //                          cv::Scalar(color.b, color.g, color.r), 2);
            //     }
            // }
        }
        // cv::Mat binary;
        // cv::Mat seg_dst_mat(seg_info.seg_height, seg_info.seg_width, CV_8UC1, seg_info.seg_map.data());
        // cv::threshold(seg_dst_mat, binary, 0, 255, cv::THRESH_BINARY);
        // cv::imwrite("binary.png", binary);
    }
    // cv::imwrite("test.png", bgr_image);

    output_result_(clone_frame);
}

}// namespace nodes
}// namespace gddi
