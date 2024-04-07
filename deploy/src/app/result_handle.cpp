#include "result_handle.h"
#include "core/result_def.h"
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

using namespace gddeploy;

int PrintResult(gddeploy::InferResult &result)
{
    for (auto result_type : result.result_type){
        if (result_type == GDD_RESULT_TYPE_DETECT_POSE){
            for (uint32_t i = 0; i < result.detect_pose_result.detect_imgs.size(); i++){
                for (auto &obj : result.detect_pose_result.detect_imgs[i].detect_objs) {
                        std::cout << "detect result: " << "box[" << obj.bbox.x \
                                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                                << obj.bbox.h << "]" \
                                << "   score: " << obj.score << std::endl;
                }
                std::cout << "----------------------------------------------" << std::endl;
            }
        } else if (result_type == GDD_RESULT_TYPE_DETECT){
            for (uint32_t i = 0; i < result.detect_result.detect_imgs.size(); i++){
                for (auto &obj : result.detect_result.detect_imgs[0].detect_objs) {
                    std::cout << "detect result: " << "box[" << obj.bbox.x \
                            << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                            << obj.bbox.h << "]" \
                            << "   score: " << obj.score << std::endl;
                }
            std::cout << "----------------------------------------------" << std::endl;
            }
        } else if (result_type == GDD_RESULT_TYPE_CLASSIFY){
            for (auto &obj : result.classify_result.detect_imgs[0].detect_objs) {
                std::cout << "Classify result: "
                    << "   score: " << obj.score 
                    << "   class id: " << obj.class_id << std::endl;
            }
        } else if (result_type == GDD_RESULT_TYPE_IMAGE_RETRIEVAL){
            for (auto &image_retrieval_img : result.image_retrieval_result.image_retrieval_imgs) {
                std::cout << "Image retrieval result: ";
                for (auto feature : image_retrieval_img.feature){
                    std::cout << feature << ", ";
                }
                std::cout << std::endl;
            }
        } else if (result_type == GDD_RESULT_TYPE_FACE_RETRIEVAL){
            for (auto &face_retrieval_img : result.face_retrieval_result.face_retrieval_imgs) {
                std::cout << "Face retrieval result: ";
                for (auto feature : face_retrieval_img.feature){
                    std::cout << feature << ", ";
                }
                std::cout << std::endl;
            }
        } else if (result_type == GDD_RESULT_TYPE_OCR_DETECT){
            for (auto &ocr_detect_img : result.ocr_detect_result.ocr_detect_imgs) {
                for (auto obj : ocr_detect_img.ocr_objs){ 
                    std::cout << "Ocr detect result: " << "box[" << obj.bbox.x \
                                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                                << obj.bbox.h << "]" \
                                << "   score: " << obj.score
                                << "   point:[ " 
                                << "(" << obj.point[0].x << ", " << obj.point[0].y << "), "
                                << "(" << obj.point[1].x << ", " << obj.point[1].y << "), "
                                << "(" << obj.point[2].x << ", " << obj.point[2].y << "), "
                                << "(" << obj.point[3].x << ", " << obj.point[3].y << ") ]"
                                << std::endl;
                }
                std::cout << "----------------------------------------------" << std::endl;
            }
        } else if (result_type == GDD_RESULT_TYPE_OCR_RETRIEVAL){
            for (auto &ocr_rec_img : result.ocr_rec_result.ocr_rec_imgs) {
                for (auto obj : ocr_rec_img.ocr_rec_objs){
                    std::cout << "Ocr retrieval result: " << "box[" << obj.bbox.x \
                                << ", " << obj.bbox.y << ", " << obj.bbox.w << ", " \
                                << obj.bbox.h << "]" \
                                << "   score: " << obj.score
                                << "   chars: " << obj.chars_str
                                // << "   point:[ " 
                                // << "(" << obj.point[0].x << ", " << obj.point[0].y << "), "
                                // << "(" << obj.point[1].x << ", " << obj.point[1].y << "), "
                                // << "(" << obj.point[2].x << ", " << obj.point[2].y << "), "
                                // << "(" << obj.point[3].x << ", " << obj.point[3].y << ") ]"
                                << std::endl;
                }
                std::cout << "----------------------------------------------" << std::endl;
            }
        }
    }


    return 0;
}

int DrawBbox(gddeploy::InferResult &result, 
            std::vector<std::string> &pic_path, 
            std::vector<cv::Mat> &in_mats, 
            std::vector<gddeploy::BufSurfWrapperPtr> &surfs, 
            std::string &save_path)
{
    int skeleton[][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11},\
                     {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2},\
                     {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};

    if(pic_path.size() > 0){
        for (auto result_type : result.result_type){
            if (result_type == GDD_RESULT_TYPE_DETECT_POSE){
                for (uint32_t i = 0; i < result.detect_pose_result.detect_imgs.size(); i++){
                    cv::Mat in_mat = cv::imread(pic_path[i]);

                    for (auto &obj : result.detect_pose_result.detect_imgs[i].detect_objs) {
                        cv::Point p1(obj.bbox.x, obj.bbox.y);
                        cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
                        cv::rectangle(in_mat, p1, p2, cv::Scalar(0, 255, 0), 1);

                        for (const auto &bone : skeleton) {
                            cv::Scalar color;
                            if (bone[0] < 5 || bone[1] < 5)
                                color = cv::Scalar(0, 255, 0);
                            else if (bone[0] > 12 || bone[1] > 12)
                                color = cv::Scalar(255, 0, 0);
                            else if (bone[0] > 4 && bone[0] < 11 && bone[1] > 4 && bone[1] < 11)
                                color = cv::Scalar(0, 255, 255);
                            else
                                color = cv::Scalar(255, 0, 255);
                            cv::line(in_mat, cv::Point(obj.point[bone[0]].x, obj.point[bone[0]].y),
                                    cv::Point(obj.point[bone[1]].x, obj.point[bone[1]].y), color,
                                    2);
                        }
                        for(const auto &point : obj.point) {
                            cv::Scalar color;
                            if (point.number < 5)
                                color = cv::Scalar(0, 255, 0);
                            else if (point.number > 10)
                                color = cv::Scalar(255, 0, 0);
                            else
                                color = cv::Scalar(0, 255, 255);
                            cv::circle(in_mat, cv::Point(point.x, point.y), 5, color, -1, cv::LINE_8, 0);
                        }
                    }

                    cv::imwrite(save_path, in_mat);
                }
            }else if (result_type == GDD_RESULT_TYPE_DETECT){
                for (uint32_t i = 0; i < result.detect_result.detect_imgs.size(); i++){
                    cv::Mat in_mat = cv::imread(pic_path[i]);

                    for (auto &obj : result.detect_result.detect_imgs[i].detect_objs) {
                        cv::Point p1(obj.bbox.x, obj.bbox.y);
                        cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
                        cv::Scalar color = cv::Scalar(obj.class_id % 255 , 255, 255 );
                        cv::putText(in_mat, obj.label+" "+std::to_string(obj.score).substr(0, 5), cv::Point(obj.bbox.x, obj.bbox.y), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 1, cv::LINE_8);
                        cv::rectangle(in_mat, p1, p2, cv::Scalar(0, 255, 0), 1);
                        // cv::putText(in_mat, obj.label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                    }

                    cv::imwrite(save_path, in_mat);
                }
            }else if (result_type == GDD_RESULT_TYPE_CLASSIFY){
                for (uint32_t i = 0; i < result.classify_result.detect_imgs.size(); i++){
                    cv::Mat in_mat = cv::imread(pic_path[i]);
                    // 修改为打印text分类结果
                    auto obj = result.classify_result.detect_imgs[i].detect_objs;
                    std::string text = "class id: "+std::to_string(obj[0].class_id)+" "+std::to_string(obj[0].score);
                    cv::putText(in_mat,text,cv::Point(50,60),cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255,23,0),4,8);
                    cv::imwrite(save_path, in_mat);
                }
            }else if (result_type == GDD_RESULT_TYPE_SEG){

                for (uint32_t i = 0; i < result.seg_result.seg_imgs.size(); i++){

                    auto &obj = result.seg_result.seg_imgs[i];
                    cv::Mat frame = cv::imread(pic_path[i]);
                    cv::Mat seg_mat(obj.map_h, obj.map_w, CV_8UC1, obj.seg_map.data());
                    cv::Mat thresh_mat;
                    cv::threshold(seg_mat, thresh_mat, 0, 255, cv::THRESH_BINARY);
                    cv::imwrite("./seg_mask.jpg", thresh_mat);

                    cv::Mat dst_mat;
                    // frame.copyTo(dst_mat, thresh_mat);
                    cv::bitwise_and(frame, frame, dst_mat, thresh_mat);
                    // cv::addWeighted(frame, 0.9, thresh_mat, 0.1, 0, dst_mat);
                    cv::imwrite("/data/preds/seg_with_mask.jpg", dst_mat);

                    //test
                    // cv::Mat frame_test = cv::imread("/volume1/gddi-data/lgy/dataset/jishui/0/images/valid/028f8b53e6c497fc87b68b111605a92471fc6fd7.jpg");
                    // cv::Mat mask = cv::imread("/volume1/gddi-data/lgy/dataset/jishui/0/images/valid/dd09d9a361d20d6b33d40b80aae53c54f7fe63d7.png", cv::IMREAD_UNCHANGED);
                    // cv::Mat dst_mat_test;
                    // frame_test.copyTo(dst_mat_test, mask);
                    // cv::imwrite("./seg_test.jpg", dst_mat_test);
                    //test

                    // 这里只需要把结果保存为.npy文件即可
                    // std::string pic_name = "/volume1/gddi-data/lgy/cambricon/preds_seg/test.npy";

                    // std::ofstream ofile(pic_name);
                    // if(ofile.is_open()==false){
                    //     std::cout << strerror(errno) << std::endl;
                    //     continue;
                    // }
                    // ofile.write((char*)obj.seg_map.get(), (obj.map_h * obj.map_w)*sizeof(uint8_t));
                    // ofile.close();

                }

                // cv::imwrite(img_save_path, frame);  
            }else if (result_type == GDD_RESULT_TYPE_IMAGE_RETRIEVAL){
            }else if (result_type == GDD_RESULT_TYPE_OCR_DETECT){
                for (uint32_t i = 0; i < result.ocr_detect_result.ocr_detect_imgs.size(); i++){
                    cv::Mat in_mat = cv::imread(pic_path[i]);

                    for (auto &obj : result.ocr_detect_result.ocr_detect_imgs[i].ocr_objs) {
                        cv::Point p1(obj.bbox.x, obj.bbox.y);
                        cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
                        cv::rectangle(in_mat, p1, p2, cv::Scalar(0, 255, 0), 1);
                    }

                    cv::imwrite(save_path, in_mat);
                }
            }
        }
    }else if(in_mats.size() > 0){  //读取Mat来获取数据
        for (auto result_type : result.result_type){
            if (result_type == GDD_RESULT_TYPE_DETECT_POSE){
                for (uint32_t i = 0; i < result.detect_pose_result.detect_imgs.size(); i++){
                    cv::Mat in_mat = in_mats[i];

                    for (auto &obj : result.detect_pose_result.detect_imgs[i].detect_objs) {
                        cv::Point p1(obj.bbox.x, obj.bbox.y);
                        cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
                        cv::rectangle(in_mat, p1, p2, cv::Scalar(0, 255, 0), 1);

                        for (const auto &bone : skeleton) {
                            cv::Scalar color;
                            if (bone[0] < 5 || bone[1] < 5)
                                color = cv::Scalar(0, 255, 0);
                            else if (bone[0] > 12 || bone[1] > 12)
                                color = cv::Scalar(255, 0, 0);
                            else if (bone[0] > 4 && bone[0] < 11 && bone[1] > 4 && bone[1] < 11)
                                color = cv::Scalar(0, 255, 255);
                            else
                                color = cv::Scalar(255, 0, 255);
                            cv::line(in_mat, cv::Point(obj.point[bone[0]].x, obj.point[bone[0]].y),
                                    cv::Point(obj.point[bone[1]].x, obj.point[bone[1]].y), color,
                                    2);
                        }
                        for(const auto &point : obj.point) {
                            cv::Scalar color;
                            if (point.number < 5)
                                color = cv::Scalar(0, 255, 0);
                            else if (point.number > 10)
                                color = cv::Scalar(255, 0, 0);
                            else
                                color = cv::Scalar(0, 255, 255);
                            cv::circle(in_mat, cv::Point(point.x, point.y), 5, color, -1, cv::LINE_8, 0);
                        }
                    }

                    cv::imwrite(save_path, in_mat);
                }
            }else if (result_type == GDD_RESULT_TYPE_DETECT){
                for (uint32_t i = 0; i < result.detect_result.detect_imgs.size(); i++){
                    cv::Mat in_mat = in_mats[i];

                    for (auto &obj : result.detect_result.detect_imgs[i].detect_objs) {
                        cv::Point p1(obj.bbox.x, obj.bbox.y);
                        cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
                        cv::rectangle(in_mat, p1, p2, cv::Scalar(0, 255, 0), 1);
                    }

                    cv::imwrite(save_path, in_mat);
                }
            }else if (result_type == GDD_RESULT_TYPE_IMAGE_RETRIEVAL){
            }else if (result_type == GDD_RESULT_TYPE_OCR_DETECT){
                for (uint32_t i = 0; i < result.ocr_detect_result.ocr_detect_imgs.size(); i++){
                    cv::Mat in_mat = in_mats[i];

                    for (auto &obj : result.ocr_detect_result.ocr_detect_imgs[i].ocr_objs) {
                        cv::Point p1(obj.bbox.x, obj.bbox.y);
                        cv::Point p2(obj.bbox.x+obj.bbox.w, obj.bbox.y+obj.bbox.h);
                        cv::rectangle(in_mat, p1, p2, cv::Scalar(0, 255, 0), 1);
                    }

                    cv::imwrite(save_path, in_mat);
                }
            }
        }
    }else if(surfs.size() > 0){  //读取surfs来获取数据
    }

    return 0;
}