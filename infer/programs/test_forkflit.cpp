#include <cstring>
#include <iostream>

#include <opencv2/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <vector>
#include <cmath>


class ForkflitPostProcess
{
private:
    int image_width_;
    int image_height_;
    cv::Mat mask_img_;

public:
    ForkflitPostProcess(int image_width, int image_height);
    void SetParam(cv::Mat cam_mtx, cv::Mat cam_dist, cv::Mat r_vec, cv::Mat t_vec, cv::Point3d cam_pos, double w_dis_one, double w_dis_two, double w_dis_three);
    int PostProcess(const std::vector<cv::Rect2d> &input_bbox);
    ~ForkflitPostProcess();
};

template <typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0)
    {
        return linspaced;
    }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                              // are exactly the same as the input
    return linspaced;
}

std::vector<cv::Point> img_point_filter(std::vector<cv::Point> points_2d)
{
    std::vector<cv::Point> out_points;
    auto last_p = points_2d.cend();
    for (auto it = points_2d.cbegin(); it != points_2d.cend() - 1; it++)
    {
        if (it->x < (it + 1)->x)
        {
            out_points.push_back(*it);
            last_p = it + 1;
        }
    }
    if (last_p != points_2d.cend() and last_p != points_2d.cend() - 1)
    {
        out_points.push_back(*last_p);
    }
    return out_points;
}

void fill_circle(cv::Mat image, cv::Mat cam_mtx, cv::Mat cam_dist, cv::Mat r_vec, cv::Mat t_vec, int sample_nums, double radius, cv::Point3d cam_pos, cv::Scalar color)
{
    std::vector<double> a = linspace(-M_PI / 2., M_PI / 2, sample_nums);
    std::vector<cv::Point3d> points_3d;
    for (auto it = a.cbegin(); it != a.cend(); it++)
    {
        points_3d.push_back(cv::Point3d(radius * cos(*it) + cam_pos.x, radius * sin(*it) + cam_pos.y, 0. + cam_pos.z));
        /* code */
    }
    int img_h = image.size[0];
    int img_w = image.size[1];
    std::vector<cv::Point2d> points_2d;
    cv::projectPoints(points_3d, r_vec, t_vec, cam_mtx, cam_dist, points_2d);

    std::vector<cv::Point> points_2d_int;
    std::transform(points_2d.begin(), points_2d.end(), std::back_inserter(points_2d_int), [](const cv::Point2d &p)
                   { return (cv::Point)p; });

    // 去掉不在画面内的点
    std::vector<cv::Point> points_2d_filter;
    std::copy_if(points_2d_int.begin(), points_2d_int.end(), std::back_inserter(points_2d_filter), [img_w, img_h](const cv::Point &p)
                 { return p.x > 0 and p.x < img_w and p.y > 0 and p.y < img_h; });

    // 去掉边缘不正常的点
    points_2d_filter = img_point_filter(points_2d_filter);

    // 加上闭合区域
    points_2d_filter.push_back(cv::Point(img_w - 1, points_2d_filter.back().y));
    points_2d_filter.push_back(cv::Point(img_w - 1, img_h - 1));
    points_2d_filter.push_back(cv::Point(0, img_h - 1));
    points_2d_filter.push_back(cv::Point(0, points_2d_filter.front().y));
    cv::fillConvexPoly(image, points_2d_filter, color);
}

ForkflitPostProcess::ForkflitPostProcess(int image_width, int image_height)
{
    image_height_ = image_height;
    image_width_ = image_width;

    mask_img_ = cv::Mat::zeros(image_height, image_width, CV_8UC1);
}

int ForkflitPostProcess::PostProcess(const std::vector<cv::Rect2d> &input_bbox)
{
    std::vector<int> result_v;
    for (auto it = input_bbox.cbegin(); it != input_bbox.cend(); it++)
    {
        int bottom_cx = it->width / 2 + it->x;
        int bottom_cy = it->height + it->y;
        if (bottom_cx > 0 and bottom_cx < mask_img_.size[1] and bottom_cy > 0 and bottom_cy < mask_img_.size[0])
        {
            result_v.push_back(mask_img_.data[bottom_cy * mask_img_.cols + bottom_cx + 0]);
        }
        else
        {
            result_v.push_back(0);
        }
    }
    int max_value = 0;
    if (!result_v.empty())
    {
        max_value = *std::max_element(result_v.begin(), result_v.end());
    }
    return max_value;
}

void ForkflitPostProcess::SetParam(cv::Mat cam_mtx, cv::Mat cam_dist, cv::Mat r_vec, cv::Mat t_vec, cv::Point3d cam_pos, double w_dis_one, double w_dis_two, double w_dis_three)
{
    mask_img_.setTo(0);

    fill_circle(mask_img_, cam_mtx, cam_dist, r_vec, t_vec, 50, w_dis_one, cam_pos, cv::Scalar(1));
    fill_circle(mask_img_, cam_mtx, cam_dist, r_vec, t_vec, 50, w_dis_two, cam_pos, cv::Scalar(2));
    fill_circle(mask_img_, cam_mtx, cam_dist, r_vec, t_vec, 50, w_dis_three, cam_pos, cv::Scalar(3));

    // debug mask
    // cv::imwrite("figure.png", mask_img_);
}

ForkflitPostProcess::~ForkflitPostProcess()
{
}

std::vector<cv::Rect2d> get_test_bbox()
{
    std::vector<cv::Rect2d> out_bbox;

    out_bbox.push_back(cv::Rect2d(94, 104, 109, 162));   // 0
    out_bbox.push_back(cv::Rect2d(512, 121, 140, 233));  // 1
    out_bbox.push_back(cv::Rect2d(71, 314, 117, 147));   // 1
    out_bbox.push_back(cv::Rect2d(1676, 76, 102, 210));  // 0
    out_bbox.push_back(cv::Rect2d(901, 32, 75, 141));    // 0
    out_bbox.push_back(cv::Rect2d(253, 162, 203, 413));  // 2
    out_bbox.push_back(cv::Rect2d(1009, 133, 167, 250)); // 2
    out_bbox.push_back(cv::Rect2d(578, 230, 176, 551));  // 3
    out_bbox.push_back(cv::Rect2d(647, 66, 209, 592));   // 3

    return out_bbox;
}

int main()
{
    // 初始化参数，这部分参数需要有平台下发功能
    // cv::Mat mtx = (cv::Mat_<float>(3, 3) << 1148.0678727231252, 0.0, 968.0883806940227, 0.0, 1155.8458179063227, 539.3126117576774, 0.0, 0.0, 1.0);
    // cv::Mat dist = (cv::Mat_<float>(5, 1) << -0.42022691705122023, 0.2110163003990064, -0.0005615454442203457, -0.00034488052319524005, -0.05416083319699509);
    // cv::Mat r_vec = (cv::Mat_<float>(3, 1) << -0.7844210658650189, -0.7820100265658598, -1.457766991763434);
    // cv::Mat t_vec = (cv::Mat_<float>(3, 1) << -16.647841493742657, 9.682050266632663, 225.67105129376233);

    cv::Mat_<float> mtx(3, 3);
    auto data = std::vector<float>{1148.0678727231252, 0.0, 968.0883806940227, 0.0, 1155.8458179063227, 539.3126117576774, 0.0, 0.0, 1.0};
    memcpy(mtx.data, data.data(), data.size() * sizeof(float));
    cv::Mat_<float> dist(5, 1);
    data = std::vector<float>{-0.42022691705122023, 0.2110163003990064, -0.0005615454442203457, -0.00034488052319524005, -0.05416083319699509};
    memcpy(dist.data, data.data(), data.size() * sizeof(float));
    cv::Mat_<float> r_vec(3, 1);
    data = std::vector<float>{-0.7844210658650189, -0.7820100265658598, -1.457766991763434};
    memcpy(r_vec.data, data.data(), data.size() * sizeof(float));
    cv::Mat_<float> t_vec(3, 1);
    data = std::vector<float>{-16.647841493742657, 9.682050266632663, 225.67105129376233};
    memcpy(t_vec.data, data.data(), data.size() * sizeof(float));
    
    cv::Point3d cam_pos = cv::Point3d(-200, 0, 0);//相机位置 unit:cm
    double warning_distance_one = 500.; // 报警1距离 unit: cm
    double warning_distance_two = 300.;// 报警2距离 unit: cm
    double warning_distance_three = 200.;// 报警3距离 unit: cm

    // 初始化模块
    int image_width = 1920;
    int image_height = 1080;
    ForkflitPostProcess f_post(image_width, image_height);
    f_post.SetParam(mtx, dist, r_vec, t_vec, cam_pos,warning_distance_one,warning_distance_two,warning_distance_three);

    // 测试后处理
    std::vector<cv::Rect2d> test_bbox = get_test_bbox();
    int result = f_post.PostProcess(test_bbox);

    // 打印结果
    std::cout << result << std::endl;
}