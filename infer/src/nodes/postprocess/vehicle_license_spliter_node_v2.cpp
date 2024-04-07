#include "vehicle_license_spliter_node_v2.h"
#include <algorithm>
#include <spdlog/spdlog.h>
#include <vector>

namespace gddi {
namespace nodes {

#define DIVISED_RATIO 3

int count_red_pixels(const cv::Mat &image) {
    // 将图像转换为HSV颜色空间
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);

    // 在HSV颜色空间中定义红色的范围
    cv::Scalar lower_red(0, 100, 100);
    cv::Scalar upper_red(10, 255, 255);

    // 创建一个掩码，将红色像素标记为白色，其他像素标记为黑色
    cv::Mat mask;
    cv::inRange(hsv_image, lower_red, upper_red, mask);

    // 计算并返回红色像素的数量
    return cv::countNonZero(mask);
}

int count_black_pixels(const cv::Mat &image) {
    auto crop_image = image(cv::Rect(176, 222, 392 - 176, 265 - 222));
    cv::Mat gray_image, binary_image;
    cv::cvtColor(crop_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_image, binary_image, 81, 255, cv::THRESH_BINARY_INV);
    return cv::countNonZero(binary_image);
}

cv::Mat meanAxis0(const cv::Mat &src) {
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1, dim, CV_32F);
    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) { sum += src.at<float>(j, i); }
        output.at<float>(0, i) = sum / num;
    }

    return output;
}

cv::Mat elementwiseMinus(const cv::Mat &A, const cv::Mat &B) {
    cv::Mat output(A.rows, A.cols, A.type());

    assert(B.cols == A.cols);
    if (B.cols == A.cols) {
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < B.cols; j++) { output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j); }
        }
    }
    return output;
}

cv::Mat varAxis0(const cv::Mat &src) {
    cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
    cv::multiply(temp_, temp_, temp_);
    return meanAxis0(temp_);
}

int MatrixRank(cv::Mat M) {
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = cv::countNonZero(nonZeroSingularValues);
    return rank;
}

cv::Mat similarTransform(cv::Mat src, cv::Mat dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = meanAxis0(src);
    cv::Mat dst_mean = meanAxis0(dst);
    cv::Mat src_demean = elementwiseMinus(src, src_mean);
    cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) { d.at<float>(dim - 1, 0) = -1; }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S, U, V);

    // the SVD function in opencv differ from scipy .

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);
    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V;// np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U * twp;
            d.at<float>(dim - 1, 0) = s;
        }
    } else {
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_ * V.t();// np.dot(np.diag(d), V.T)
        cv::Mat res = U * twp;      // U
        T.rowRange(0, dim).colRange(0, dim) = U * diag_ * V;
    }
    cv::Mat var_ = varAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d, S, res);
    float scale = 1.0 / val * cv::sum(res).val[0];
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim) * src_mean.t();
    cv::Mat temp2 = scale * temp1;
    cv::Mat temp3 = dst_mean - temp2.t();
    T.at<float>(0, 2) = temp3.at<float>(0);
    T.at<float>(1, 2) = temp3.at<float>(1);
    T.rowRange(0, dim).colRange(0, dim) *= scale;// T[:dim, :dim] *= scale

    return T;
}

void VehicleLicenseSpliter_v2::on_setup() {
    // 解析正面框位置信息
    std::fstream front_mask_file(front_mask_path_);
    if (!front_mask_file.is_open()) {
        spdlog::error("front_mask_file is not exist");
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }
    // 解析文件内容
    auto front_mask = nlohmann::json::parse(front_mask_file);
    for (const auto &item : front_mask["shapes"]) {
        std::vector<cv::Point> mask_points;
        for (const auto &point : item["points"].get<std::vector<std::vector<int>>>()) {
            mask_points.emplace_back(static_cast<int>(point[0] / DIVISED_RATIO),
                                     static_cast<int>(point[1] / DIVISED_RATIO));
        }
        front_mask_points_.emplace_back(std::move(mask_points));
    }

    // 解析反面框位置信息
    std::fstream back_mask_file(back_mask_path_);
    if (!back_mask_file.is_open()) {
        spdlog::error("back_mask_file is not exist");
        quit_runner_(TaskErrorCode::kInvalidArgument);
        return;
    }
    auto back_mask = nlohmann::json::parse(back_mask_file);
    for (const auto &item : back_mask["shapes"]) {
        std::vector<cv::Point> mask_points;
        for (const auto &point : item["points"].get<std::vector<std::vector<int>>>()) {
            mask_points.emplace_back(static_cast<int>(point[0] / DIVISED_RATIO),
                                     static_cast<int>(point[1] / DIVISED_RATIO));
        }
        back_mask_points_.emplace_back(std::move(mask_points));
    }
}

void VehicleLicenseSpliter_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    auto back_ext_info = frame->frame_info->ext_info.back();

    std::vector<std::vector<float>> vec_key_points;
    for (const auto &[_, key_points] : back_ext_info.map_key_points) {
        std::vector<float> points;
        for (const auto &item : key_points) {
            points.emplace_back(item.x);
            points.emplace_back(item.y);
        }
        vec_key_points.emplace_back(points);
    }

    auto transformed_imgs = split_font_back_img(*frame->frame_info->src_frame->data, vec_key_points);

    auto type = frame->frame_info->frame_meta["type"].get<std::string>();
    if (type == "front-back") {
        if (transformed_imgs.size() != 2) {
            // 这里需要输出两次，因为后面的节点需要两次结果
            output_result_(frame);
            output_result_(frame);
            return;
        }
    } else if (type == "front" || type == "back") {
        if (transformed_imgs.size() != 1) {
            output_result_(frame);
            return;
        }
    } else {
        output_result_(frame);
        return;
    }

    int index = 0;
    for (auto &image : transformed_imgs) {
        spdlog::debug("count_black_pixels: {}", count_black_pixels(image));

        // cv::imwrite("transformed_imgs_" + std::to_string(index) + ".jpg", image);

        auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
        clone_frame->frame_info->ext_info.back().flag_crop = true;
        clone_frame->frame_info->ext_info.back().crop_images[0] = image;
        clone_frame->frame_info->ext_info.back().crop_rects[0] = {vec_key_points[index][0], vec_key_points[index][1],
                                                                  vec_key_points[index][2] - vec_key_points[index][0],
                                                                  vec_key_points[index][3] - vec_key_points[index][1]};

        if (index == 0) {
            clone_frame->frame_info->ext_info.back().mask_points = front_mask_points_;
        } else {
            clone_frame->frame_info->ext_info.back().mask_points = back_mask_points_;
        }

        ++index;
        output_result_(clone_frame);
    }
}

std::vector<cv::Mat>
VehicleLicenseSpliter_v2::split_font_back_img(const cv::Mat &image,
                                              const std::vector<std::vector<float>> &vec_key_points) {
    std::vector<cv::Mat> transformed_imgs;
    for (const auto &key_points : vec_key_points) {
        float arcface_srcf[8] = {60 / DIVISED_RATIO,
                                 40 / DIVISED_RATIO,
                                 1260 / DIVISED_RATIO,
                                 40 / DIVISED_RATIO,
                                 1260 / DIVISED_RATIO,
                                 840 / DIVISED_RATIO,
                                 0,
                                 840 / DIVISED_RATIO};
        cv::Mat arcface_src(4, 2, CV_32FC1, arcface_srcf);

        cv::Mat kps(4, 2, CV_32FC1, const_cast<float *>(key_points.data()));
        // std::cout << "kps: " << kps << std::endl;
        cv::Mat M = similarTransform(kps, arcface_src);
        // TOC("similarTransform:");
        M = M(cv::Rect(0, 0, 3, 2));
        // std::cout << "M: " << M << std::endl;
        cv::Mat transformed_img;
        int h = static_cast<int>(std::ceil(880.0 / DIVISED_RATIO));
        int w = static_cast<int>(std::ceil(1320.0 / DIVISED_RATIO));
        cv::warpAffine(image, transformed_img, M, cv::Size(w, h));

        transformed_imgs.push_back(transformed_img);
    }

    std::sort(transformed_imgs.begin(), transformed_imgs.end(),
              [](const cv::Mat &a, const cv::Mat &b) { return count_black_pixels(a) < count_black_pixels(b); });

    return transformed_imgs;
}
}// namespace nodes
}// namespace gddi
