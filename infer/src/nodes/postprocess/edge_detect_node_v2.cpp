#include "edge_detect_node_v2.h"
#include "spdlog/spdlog.h"
#include "types.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace gddi {
namespace nodes {

void EdgeDetect_v2::on_setup() {}

//5×5高斯滤波
cv::Mat _gaussian_filter(const cv::Mat &mat) {
    cv::Mat matDouble;
    mat.convertTo(matDouble, CV_64FC1);
    cv::Mat kernel =
        (cv::Mat_<double>(5, 5) << 2, 4, 5, 4, 2, 4, 9, 12, 9, 4, 5, 12, 15, 12, 5, 4, 9, 12, 9, 4, 2, 4, 5, 4, 2);
    kernel = kernel / 159;
    cv::Mat resDouble;
    cv::filter2D(matDouble, resDouble, -1, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT101);
    cv::Mat res;
    resDouble.convertTo(res, CV_8UC1);
    return res;
}

//对滤波后的图利用sobel计算梯度，通过梯度角的tan值与tan22.5进行一些比较获取梯度角所属分区
//angle = 0-> horizontal, 1 -> vertical, 2 -> diagonal
void _sobel_gradient(const cv::Mat &mat, cv::Mat &dx, cv::Mat &dy, cv::Mat &magnitudes, cv::Mat &angles,
                     int apertureSize, bool L2gradient) {
    CV_Assert(apertureSize == 3 || apertureSize == 5);

    double scale = 1.0;
    cv::Sobel(mat, dx, CV_16S, 1, 0, apertureSize, scale, cv::BORDER_REPLICATE);
    cv::Sobel(mat, dy, CV_16S, 0, 1, apertureSize, scale, cv::BORDER_REPLICATE);

    const int TAN225 = 13573;//tan22.5 * 2^15(2 << 15)

    angles = cv::Mat(mat.size(), CV_8UC1);// 0-> horizontal, 1 -> vertical, 2 -> diagonal
    magnitudes = cv::Mat::zeros(mat.rows + 2, mat.cols + 2, CV_32SC1);
    cv::Mat magROI = cv::Mat(magnitudes, cv::Rect(1, 1, mat.cols, mat.rows));

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            short xs = dx.ptr<short>(i)[j];
            short ys = dy.ptr<short>(i)[j];
            int x = (int)std::abs(xs);
            int y = (int)std::abs(ys) << 15;

            if (L2gradient) {
                //magROI.ptr<int>(i)[j] = int(xs) * xs + int(ys) * ys;
                magROI.ptr<int>(i)[j] = (int)std::sqrt(xs * xs + ys * ys);
            } else {
                magROI.ptr<int>(i)[j] = std::abs(int(xs)) + std::abs(int(ys));
            }

            int tan225x = x * TAN225;
            if (y < tan225x) {// horizontal
                angles.ptr<uchar>(i)[j] = 0;
            } else {
                int tan675x = tan225x + (x << 16);
                if (y > tan675x) {// vertical
                    angles.ptr<uchar>(i)[j] = 1;
                } else {// diagonal
                    angles.ptr<uchar>(i)[j] = 2;
                }
            }
        }
    }
}

//根据angles将梯度图进行非极大值抑制得到NMSImage，对其利用OTSU算法计算阈值，
//计算得到的阈值为高阈值high，低阈值取0.5*high
void _calculate_hysteresis_threshold_value(const cv::Mat &dx, const cv::Mat &dy, const cv::Mat &magnitudes,
                                           const cv::Mat &angles, cv::Mat &NMSImage, int &low, int &high) {
    NMSImage = cv::Mat::zeros(magnitudes.size(), magnitudes.type());//CV_32SC1

    for (int i = 0; i < dx.rows; ++i) {
        int r = i + 1;
        for (int j = 0; j < dx.cols; ++j) {
            int c = j + 1;
            int m = magnitudes.ptr<int>(r)[c];
            uchar angle = angles.ptr<uchar>(i)[j];

            if (angle == 0)//horizontal
            {
                if (m > magnitudes.ptr<int>(r)[c - 1] && m >= magnitudes.ptr<int>(r)[c + 1])
                    NMSImage.ptr<int>(r)[c] = m;
            } else if (angle == 1)//vertical
            {
                if (m > magnitudes.ptr<int>(r - 1)[c] && m >= magnitudes.ptr<int>(r + 1)[c])
                    NMSImage.ptr<int>(r)[c] = m;
            } else if (angle == 2)//diagonal
            {
                short xs = dx.ptr<short>(i)[j];
                short ys = dy.ptr<short>(i)[j];
                if ((xs > 0 && ys > 0) || (xs < 0 && ys < 0)) {//45 degree
                    if (m > magnitudes.ptr<int>(r - 1)[c - 1] && m > magnitudes.ptr<int>(r + 1)[c + 1])
                        NMSImage.ptr<int>(r)[c] = m;
                } else {//135 degree
                    if (m > magnitudes.ptr<int>(r - 1)[c + 1] && m > magnitudes.ptr<int>(r + 1)[c - 1])
                        NMSImage.ptr<int>(r)[c] = m;
                }
            }
        }
    }

    //利用Otsu对非极大值抑制图像进行处理，将计算得到的阈值作为高阈值high, 低阈值取高阈值的0.5倍
    cv::normalize(NMSImage, NMSImage, 0, 255, cv::NORM_MINMAX);
    NMSImage.convertTo(NMSImage, CV_8UC1);

    cv::Mat temp;
    high = (int)cv::threshold(NMSImage, temp, 0, 255, cv::THRESH_OTSU);
    low = (int)(0.5 * high);
}

//对非极大值抑制后的图根据高低阈值进行标记，当当前像素小于low，则标记为1，当当前像素大于low且大于high，则标记为2
//当大于low小于high时标记为0，并将标记为2的像素坐标压入队列
void _non_maximum_suppression(const cv::Mat &NMSImage, cv::Mat &map, std::deque<int> &mapIndicesX,
                              std::deque<int> &mapIndicesY, int low, int high) {
    // 0 -> the pixel may be edge
    // 1 -> the pixel is not edge
    // 2 -> the pixel is edge
    map = cv::Mat::ones(NMSImage.size(), CV_8UC1);

    for (int i = 0; i < NMSImage.rows; ++i) {
        for (int j = 0; j < NMSImage.cols; ++j) {
            int m = NMSImage.ptr<uchar>(i)[j];//nms -> CV_8UC1
            if (m > low) {
                if (m > high) {
                    map.ptr<uchar>(i)[j] = 2;
                    mapIndicesX.push_back(j);
                    mapIndicesY.push_back(i);
                } else
                    map.ptr<uchar>(i)[j] = 0;
            }
        }
    }
}

//双阈值滞后处理：根据队列中的像素坐标，进行8领域边缘点寻找，即在map中与2相连的0均认作为边缘点
void _hysteresis_thresholding(std::deque<int> &mapIndicesX, std::deque<int> &mapIndicesY, cv::Mat &map) {
    while (!mapIndicesX.empty()) {
        int r = mapIndicesY.back();
        int c = mapIndicesX.back();
        //获取到边缘点之后要将其弹出
        mapIndicesX.pop_back();
        mapIndicesY.pop_back();

        // top left
        if (map.ptr<uchar>(r - 1)[c - 1] == 0) {
            mapIndicesX.push_back(c - 1);
            mapIndicesY.push_back(r - 1);
            map.ptr<uchar>(r - 1)[c - 1] = 2;
        }
        // top
        if (map.ptr<uchar>(r - 1)[c] == 0) {
            mapIndicesX.push_back(c);
            mapIndicesY.push_back(r - 1);
            map.ptr<uchar>(r - 1)[c] = 2;
        }
        // top right
        if (map.ptr<uchar>(r - 1)[c + 1] == 0) {
            mapIndicesX.push_back(c + 1);
            mapIndicesY.push_back(r - 1);
            map.ptr<uchar>(r - 1)[c + 1] = 2;
        }
        // left
        if (map.ptr<uchar>(r)[c - 1] == 0) {
            mapIndicesX.push_back(c - 1);
            mapIndicesY.push_back(r);
            map.ptr<uchar>(r)[c - 1] = 2;
        }
        // right
        if (map.ptr<uchar>(r)[c + 1] == 0) {
            mapIndicesX.push_back(c + 1);
            mapIndicesY.push_back(r);
            map.ptr<uchar>(r)[c + 1] = 2;
        }
        // bottom left
        if (map.ptr<uchar>(r + 1)[c - 1] == 0) {
            mapIndicesX.push_back(c - 1);
            mapIndicesY.push_back(r + 1);
            map.ptr<uchar>(r + 1)[c - 1] = 2;
        }
        // bottom
        if (map.ptr<uchar>(r + 1)[c] == 0) {
            mapIndicesX.push_back(c);
            mapIndicesY.push_back(r + 1);
            map.ptr<uchar>(r + 1)[c] = 2;
        }
        // bottom right
        if (map.ptr<uchar>(r + 1)[c + 1] == 0) {
            mapIndicesX.push_back(c + 1);
            mapIndicesY.push_back(r + 1);
            map.ptr<uchar>(r + 1)[c + 1] = 2;
        }
    }
}

cv::Mat _get_canny_result(const cv::Mat &map) {
    cv::Mat dst(map.rows - 2, map.cols - 2, CV_8UC1);
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) { dst.ptr<uchar>(i)[j] = (map.ptr<uchar>(i + 1)[j + 1] == 2 ? 255 : 0); }
    }
    return dst;
}

// /*--------函数封装---------*/
// //自适应阈值canny plus版本
// cv::Mat Adaptive_Canny(const cv::Mat& src, int apertureSize, bool L2gradient)
// {
// 	CV_Assert(src.type() == CV_8UC1);
// 	CV_Assert(apertureSize == 3 || apertureSize == 5);

// 	cv::Mat gaussianSrc = _gaussian_filter(src);

// 	cv::Mat dx, dy, magnitudes, angles;
// 	_sobel_gradient(gaussianSrc, dx, dy, magnitudes, angles, apertureSize, L2gradient);

// 	//非极大值抑制计算高低阈值
// 	int low, high;
// 	cv::Mat NMSImage;
// 	_calculate_hysteresis_threshold_value(dx, dy, magnitudes, angles, NMSImage, low, high);

// 	cv::Mat map;
// 	std::deque<int> mapIndicesX, mapIndicesY;
// 	_non_maximum_suppression(NMSImage, map, mapIndicesX, mapIndicesY, low, high);

// 	_hysteresis_thresholding(mapIndicesX, mapIndicesY, map);
// 	cv::Mat dst = _get_canny_result(map);

// 	return dst;
// }

void EdgeDetect_v2::on_cv_image(const std::shared_ptr<msgs::cv_frame> &frame) {
    if (frame->frame_type == FrameType::kNone) {
        output_result_(frame);
        return;
    }

    auto clone_frame = std::make_shared<msgs::cv_frame>(frame);
    auto &back_ext_info = clone_frame->frame_info->ext_info.back();
    // 1. 图像边缘检测
    auto src_frame = image_wrapper::image_to_mat(frame->frame_info->src_frame->data);
    cv::Mat gray, blurred, edge;
    cv::cvtColor(src_frame, gray, cv::COLOR_BGR2GRAY);
    int ksize = 5;
    cv::GaussianBlur(gray, blurred, cv::Size(ksize, ksize), 0);
    //Canny(blurred, edge, 50, 150);
    //edge = Adaptive_Canny(gray,3,true);
    cv::Canny(blurred, edge, 0, 0);
    double otsu_threshold = cv::threshold(blurred, blurred, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Canny(blurred, edge, otsu_threshold * 0.5, otsu_threshold);
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(edge, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // 2. 生成目标朝向的最小外接矩形，保留该框
    for (const auto &[key, item] : back_ext_info.map_target_box) {
        std::vector<cv::Point2f> point;
        if (back_ext_info.map_class_label.at(item.class_id) == label_) {
            for (int i = 0; i < cnts.size(); i++) {
                float area = cv::contourArea(cnts[i]);
                cv::RotatedRect rect = cv::minAreaRect(cnts[i]);
                cv::Point2f center = rect.center;
                if (center.x >= item.box.x && center.x <= item.box.x + item.box.width && center.y >= item.box.y
                    && center.y <= item.box.y + item.box.height) {
                    std::vector<cv::Point2f> box(4);
                    rect.points(box.data());
                    point.insert(point.end(), box.begin(), box.end());
                }
            }
            if (!point.empty()) {
                cv::RotatedRect rect = cv::minAreaRect(point);
                cv::Point2f vertex[4];
                rect.points(vertex);
                spdlog::debug("Vertex: {},{},{},{},{},{},{},{}", vertex[0].x, vertex[0].y, vertex[1].x, vertex[1].y,
                             vertex[2].x, vertex[2].y, vertex[3].x, vertex[3].y);
                for (int i = 0; i < 4; i++) {
                    if (rect.size.width > rect.size.height) {
                        back_ext_info.map_key_points[key].emplace_back(PoseKeyPoint{i, vertex[i].x, vertex[i].y, 1});
                    } else {
                        back_ext_info.map_key_points[key].emplace_back(
                            PoseKeyPoint{i, vertex[(i + 1) % 4].x, vertex[(i + 1) % 4].y, 1});
                    }
                }
            }
        }
    }

    output_result_(clone_frame);
}

}// namespace nodes
}// namespace gddi
