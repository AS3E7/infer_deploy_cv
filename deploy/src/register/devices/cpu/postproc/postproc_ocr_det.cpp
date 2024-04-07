#include "postproc_ocr_det.h"

#include <math.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace gddeploy;

#include <stdlib.h>
#include <stdio.h>
#include "clipper2/clipper.h"  
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>


double get_max_length_in_contours(const std::vector<cv::Point> &contour)
{
    double max_length = -1.;
    for (auto it = contour.cbegin(); it != contour.cend(); it++)
    {
        for (auto it2 = it + 1; it2 != contour.cend(); it2++)
        {
            max_length = std::max(max_length, sqrt((it->x - it2->x)*(it->x - it2->x) + (it->y - it2->y)*(it->y - it2->y)));
        }
    }
    return max_length;
}

namespace bg = boost::geometry;
using point_type = bg::model::d2::point_xy<float>;
using polygon_type = bg::model::polygon<point_type>;

polygon_type convert_polygon(const std::vector<cv::Point> &region) {
    polygon_type poly;
    for (auto &point : region) { bg::append(poly, bg::make<bg::model::d2::point_xy<float>>(point.x, point.y)); }
    bg::correct(poly);
    return poly;
}

float polygon_area(const std::vector<cv::Point> &region) { 
    return bg::area(convert_polygon(region)); 
}


//
int OcrDetectDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    float threshold = any_cast<float>(param);

    float box_thresh=0.7;
    float bitmap_thresh = 0.3f;
    float unclip_ratio = 1.25f;

    auto output_shape = model_ptr->OutputShape(0);

    int out_w = output_shape[2];
    int out_h = output_shape[3];
    int out_c = output_shape[1];

    auto input_shape = model_ptr->InputShape(0);
    int model_w = input_shape[2];
    int model_h = input_shape[3];
    int model_b = input_shape[0];

    for (size_t b = 0; b < out_data.size(); b++) {
        OcrDetectImg ocr_det_img;
        ocr_det_img.img_id = b;
        ocr_det_img.img_w = frame_info[b].width;
        ocr_det_img.img_h = frame_info[b].height;

        float *data_ptr = static_cast<float*>(out_data[0]->GetHostData(0, b));

        // 1.找contours
        // 1.1 threshold获得掩码图
        cv::Mat out_mat(out_h, out_w, CV_32FC1, data_ptr);
        cv::Mat thresh_mat;
        cv::threshold(out_mat, thresh_mat, bitmap_thresh, 1, cv::THRESH_TOZERO);
        cv::Mat thresh_int8_mat;
        thresh_mat.convertTo(thresh_int8_mat, CV_8UC1, 255.0);
        // cv::imwrite("/data/preds/thresh_int8_mat.jpg", thresh_int8_mat);

        //找连通域
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh_int8_mat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE );

        if (contours.size() > 1000){
            contours.erase(contours.begin() + 1000, contours.end());
        }

        int index = 0; 
        
        for (auto contour : contours){
            // 2. 找最小框
            cv::RotatedRect rotated_bbox = cv::minAreaRect(contour); //带旋转矩阵框
            std::vector<cv::Point2f> pts;
            pts.resize(4);
            rotated_bbox.points(pts.data());
            std::vector<cv::Point> pts_int;
            for (auto point : pts){
                pts_int.emplace_back(cv::Point{(int)point.x, (int)point.y});
            }
            std::vector<cv::Point2d> pts_double;
            for (auto point : pts){
                pts_double.emplace_back(cv::Point2d{(double)point.x, (double)point.y});
            }

            //圈中这个旋转矩形的外矩形
            // cv::Rect brect = rotated_bbox.boundingRect();

            // 3. 计算box score
            cv::Mat bbox_mask;
            thresh_int8_mat.copyTo(bbox_mask);
            bbox_mask.setTo(cv::Scalar::all(0));

            int npt[] = { (int)contour.size() };
            const cv::Point *pp = contour.data();
            cv::fillPoly(bbox_mask, &pp, npt, 1, cv::Scalar(255));

            // cv::imwrite("/data/preds/beforeadd"+std::to_string(index++)+".jpg", bbox_mask);

            cv::Scalar mean = cv::mean(thresh_mat, bbox_mask);
            float mean_score = mean.val[0];
            if (mean_score < box_thresh){
                continue;
            }

            // 4. unclip
            // 找旋转矩形框的最长边，用面积/最长边得到短边长度
            float max_length = get_max_length_in_contours(pts_int);
            float area = polygon_area(pts_int);
            double distance = area * unclip_ratio / max_length;
            if (area == 0)
                continue;
            // printf("max_length:%f, area:%f, distance:%f\n", max_length, area, distance);

            // 扩大矩形到1.5倍
            // Clipper2Lib::Path subj;
            std::vector<Clipper2Lib::Point<double>> subj;
            // Clipper2Lib::Paths solution;
            // printf("old point: ");
            for (auto point : pts_double){
                subj.emplace_back(Clipper2Lib::Point<double>(point.x, point.y));
                // printf("(%ld, %ld)\t", (int)point.x, (int)point.y);
            }
            // printf("\n");
            Clipper2Lib::ClipperOffset co;
            co.AddPath(subj, Clipper2Lib::JoinType::Miter, Clipper2Lib::EndType::Polygon);
            auto solutions = co.Execute(distance);

            // std::vector<cv::Point> test_point;
            // printf("new point: ");
            // for (auto new_point : solutions[0]){
            //     test_point.emplace_back(cv::Point{(int)new_point.x, (int)new_point.y});
            //     printf("(%d, %d)\t", (int)new_point.x, (int)new_point.y);
            //     cv::circle(thresh_int8_mat, cv::Point{(int)new_point.x, (int)new_point.y}, 10, cv::Scalar(255));
            // }
            // printf("\n");
            // cv::imwrite("/gddeploy/preds/"+std::to_string(index++)+"2.jpg", thresh_int8_mat);

            // 6. save result
            for (auto pts : solutions){
                OcrDetectObject ocr_obj;
                ocr_obj.detect_id = index++;
                ocr_obj.class_id = 0;
                ocr_obj.score = mean_score;

                // 注意：这里的点的顺序为右上、右下、左下、左上的逆时针方向
                for (auto point : pts){
                    // 换换为原来图形wh
                    int x = (int)point.x * ((float)ocr_det_img.img_w / model_w);
                    int y = (int)point.y * ((float)ocr_det_img.img_h / model_h);

                    // 5. clip, 不能超过图形边界
                    x = std::max(0, std::min(ocr_det_img.img_w, x));
                    y = std::max(0, std::min(ocr_det_img.img_h, y));
                    
                    PoseKeyPoint p = {x, y, 0, mean_score};
                    ocr_obj.point.emplace_back(p);    
                }
                //保存最小矩形框
                std::vector<cv::Point2f> pts_cv;
                for (auto point : pts){
                    int x = (int)point.x * ((float)ocr_det_img.img_w / model_w);
                    int y = (int)point.y * ((float)ocr_det_img.img_h / model_h);

                    // 5. clip, 不能超过图形边界
                    x = std::max(0, std::min(ocr_det_img.img_w, x));
                    y = std::max(0, std::min(ocr_det_img.img_h, y));
                    pts_cv.emplace_back(cv::Point{x, y});
                }
                auto rect = cv::boundingRect(pts_cv);
                ocr_obj.bbox = {(float)rect.x, (float)rect.y, (float)rect.width, (float)rect.height};

                ocr_det_img.ocr_objs.emplace_back(ocr_obj);
            }
        }

        result.ocr_detect_result.batch_size++;
        result.ocr_detect_result.ocr_detect_imgs.emplace_back(ocr_det_img);
    }
    result.result_type.emplace_back(GDD_RESULT_TYPE_OCR_DETECT);
    
    return 0;
}

int OcrDetectDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::any param,  
                                    std::vector<gddeploy::FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{   
    OcrDetectDecodeOutputNCHW(out_data, result, param, frame_info, model_ptr);

    return 0;
}
