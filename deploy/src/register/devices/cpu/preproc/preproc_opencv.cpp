#include "preproc_opencv.h"
#include "opencv2/core.hpp"

#include "transform/transform.h"
#include <opencv2/imgproc.hpp>
#include "core/result_def.h"

namespace gddeploy
{

    int PreprocYolov5(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        int w, h, x, y;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_h / (input_mat_w_ * 1.0);
        float r_h = model_w / (input_mat_h_ * 1.0);

        if (r_h > r_w)
        {
            w = model_h;
            h = r_w * input_mat_h_;
            x = 0;
            y = (model_w - h) / 2;
        }
        else
        {
            w = r_h * input_mat_w_;
            h = model_w;
            x = (model_h - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        cv::Mat out_float;
        out.convertTo(out_float, CV_32FC3, 1 / 255.0);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_float, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

    int PreprocYolov5Intel(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        int w, h, x, y;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_h / (input_mat_w_ * 1.0);
        float r_h = model_w / (input_mat_h_ * 1.0);

        if (r_h > r_w)
        {
            w = model_h;
            h = r_w * input_mat_h_;
            x = 0;
            y = (model_w - h) / 2;
        }
        else
        {
            w = r_h * input_mat_w_;
            h = model_w;
            x = (model_h - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        cv::Mat out_float;
        out.convertTo(out_float, CV_32FC3, 1 / 255.0);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_float, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

     int PreprocYolov5NHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        int w, h, x, y;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_h / (input_mat_w_ * 1.0);
        float r_h = model_w / (input_mat_h_ * 1.0);

        if (r_h > r_w)
        {
            w = model_h;
            h = r_w * input_mat_h_;
            x = 0;
            y = (model_w - h) / 2;
        }
        else
        {
            w = r_h * input_mat_w_;
            h = model_w;
            x = (model_h - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        cv::Mat out_float;
        out.convertTo(out_float, CV_32FC3, 1 / 255.0);

        std::memcpy((float *)output_mat.data, out_float.data, model_w * model_h * 3 * sizeof(float));

        return 0;
    }

    int PreprocYolov5Ts(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        int w, h, x, y;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_h / (input_mat_w_*1.0);
        float r_h = model_w / (input_mat_h_*1.0);

        if (r_h > r_w) {
            w = model_h;
            h = r_w * input_mat_h_;
            x = 0;
            y = (model_w - h) / 2;
        } else {
            w = r_h * input_mat_w_;
            h = model_w;
            x = (model_h - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
        
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        std::memcpy((char *)output_mat.data, out.data, 3*model_w*model_h*sizeof(char));

        // out.convertTo(output_mat, CV_32FC3, 1.0);
        // for (int i = 0; i < 3; i++){
        //     std::memcpy((float *)output_mat.data+i*model_w*model_h, bgrChannels[i].data, model_w*model_h*sizeof(float));
        // }

        return 0;
    }

    int PreprocYolox(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        int w, h, x, y;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_h / (input_mat_w_ * 1.0);
        float r_h = model_w / (input_mat_h_ * 1.0);

        if (r_h > r_w)
        {
            w = model_h;
            h = r_w * input_mat_h_;
            x = 0;
            y = 0;
        }
        else
        {
            w = r_h * input_mat_w_;
            h = model_w;
            x = 0;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        cv::Mat out_float;
        out.convertTo(out_float, CV_32FC3, 1.0);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_float, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

    int PreprocYoloxNHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        int w, h, x, y;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_h / (input_mat_w_ * 1.0);
        float r_h = model_w / (input_mat_h_ * 1.0);

        if (r_h > r_w)
        {
            w = model_h;
            h = r_w * input_mat_h_;
            x = 0;
            y = 0;
        }
        else
        {
            w = r_h * input_mat_w_;
            h = model_w;
            x = 0;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        cv::Mat out_float;
        out.convertTo(out_float, CV_32FC3, 1.0);

        std::memcpy((float *)output_mat.data, out_float.data, model_w * model_h * 3 * sizeof(float));

        return 0;
    }

    // ===================================RTMPose PreProc================================================ //
    cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, 
        int output_image_width, int output_image_height, bool inverse = false)
    {
        // solve the affine transformation matrix

        // get the three points corresponding to the source picture and the target picture
        cv::Point2f src_point_1;
        src_point_1.x = center_x;
        src_point_1.y = center_y;

        cv::Point2f src_point_2;
        src_point_2.x = center_x;
        src_point_2.y = center_y - scale_width * 0.5;

        cv::Point2f src_point_3;
        src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
        src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);


        float alphapose_image_center_x = output_image_width / 2;
        float alphapose_image_center_y = output_image_height / 2;

        cv::Point2f dst_point_1;
        dst_point_1.x = alphapose_image_center_x;
        dst_point_1.y = alphapose_image_center_y;

        cv::Point2f dst_point_2;
        dst_point_2.x = alphapose_image_center_x;
        dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

        cv::Point2f dst_point_3;
        dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
        dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);


        cv::Point2f srcPoints[3];
        srcPoints[0] = src_point_1;
        srcPoints[1] = src_point_2;
        srcPoints[2] = src_point_3;

        cv::Point2f dstPoints[3];
        dstPoints[0] = dst_point_1;
        dstPoints[1] = dst_point_2;
        dstPoints[2] = dst_point_3;

        // get affine matrix
        cv::Mat affineTransform;
        if (inverse)
        {
            affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
        }
        else
        {
            affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
        }

        return affineTransform;
    }

    cv::Mat CropImageByDetectBox(const cv::Mat& input_image, int model_h, int model_w, const gddeploy::Bbox& box)
    {
        std::pair<cv::Mat, cv::Mat> result_pair;

        if (!input_image.data)
        {
            return cv::Mat();
        }
        // deep copy
        cv::Mat input_mat_copy = input_image;

        // calculate the width, height and center points of the human detection box
        int box_width = box.w;
        int box_height = box.h;
        int box_center_x = box.x + box_width / 2;
        int box_center_y = box.y + box_height / 2;

        float aspect_ratio = model_w * 1.0 / model_h;

        // adjust the width and height ratio of the size of the picture in the RTMPOSE input
        if (box_width > (aspect_ratio * box_height))
        {
            box_height = box_width / aspect_ratio;
        }
        else if (box_width < (aspect_ratio * box_height))
        {
            box_width = box_height * aspect_ratio;
        }

        float scale_image_width = box_width * 1.2;
        float scale_image_height = box_height * 1.2;

        // get the affine matrix
        cv::Mat affine_transform = GetAffineTransform(
            box_center_x,
            box_center_y,
            scale_image_width,
            scale_image_height,
            model_w,
            model_h
        );

        // affine transform
        cv::Mat affine_image;
        cv::warpAffine(input_mat_copy, affine_image, affine_transform, cv::Size(model_w, model_h), cv::INTER_LINEAR);
        //cv::imwrite("affine_img.jpg", affine_image);

        return affine_image;
    }

    int PreprocRTMPoseNHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w, InferResult &results)
    {
        // 解析result获取box
        DetectResult detect_result = results.detect_result;
        gddeploy::Bbox box = {0, 0, input_mat.cols, input_mat.rows};
        
        if (detect_result.detect_imgs.size() > 0 && detect_result.detect_imgs[0].detect_objs.size() > 0){
            box = detect_result.detect_imgs[0].detect_objs[0].bbox;
        }

        cv::Mat out = CropImageByDetectBox(input_mat, model_h, model_w, box);
        // cv::imwrite("../preds/rtmpose_preproc.jpg", out);

        std::memcpy((float *)output_mat.data, out.data, model_w * model_h * 3 * sizeof(char));

        return 0;
    }

    int PreprocClassify(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        cv::Mat re(model_h, model_w, CV_8UC3);

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        float mean[] = {123.675, 116.28, 103.53};
        float std[] = {1 / 58.395, 1 / 57.12, 1 / 57.375};
        cv::Mat out_float = gddeploy::transform::Normalize(re, mean, std);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_float, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

    int PreprocClassifyNHWC(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        cv::Mat re(model_h, model_w, CV_8UC3);

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        float mean[] = {123.675, 116.28, 103.53};
        float std[] = {1 / 58.395, 1 / 57.12, 1 / 57.375};
        cv::Mat out_float = gddeploy::transform::Normalize(re, mean, std);

        std::memcpy((float *)output_mat.data, out_float.data, model_w * model_h * 3 * sizeof(float));

        return 0;
    }

    int PreprocSeg(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        int w, h, x, y;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_w / (input_mat_w_ * 1.0);
        float r_h = model_h / (input_mat_h_ * 1.0);

        if (r_h > r_w)
        {
            w = model_w;
            h = r_w * input_mat_h_;
            x = 0;
            y = 0;
        }
        else
        {
            w = r_h * input_mat_w_;
            h = model_h;
            x = 0;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_h, model_w, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        float mean[] = {123.675, 116.28, 103.53};
        float std[] = {1 / 58.395, 1 / 57.12, 1 / 57.375};
        cv::Mat out_float = gddeploy::transform::Normalize(out, mean, std);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_float, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

    int PreprocAction(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {

        return 0;
    }

    /* 步骤：
        1. To RGB
        2. 补齐长边Resize
        3. /255
        4. Normalize
        5. To RGB planner
     */
    int PreprocImageRetrieval(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        // if (input_mat.channels() == 1)
        //     cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        // else
        //     cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);
        img = input_mat;

        int input_mat_w_ = img.cols;
        int input_mat_h_ = img.rows;
        float r_w = model_h / (input_mat_w_ * 1.0);
        float r_h = model_w / (input_mat_h_ * 1.0);

        int w = 0, h = 0, x = 0, y = 0;
        if (r_h > r_w)
        {
            w = model_h;
            h = r_w * input_mat_h_;
            x = 0;
            y = (model_w - h) / 2;
        }
        else
        {
            w = r_h * input_mat_w_;
            h = model_w;
            x = (model_h - w) / 2;
            y = 0;
        }

        w = w + (4 - w % 4) % 4; // align;

        cv::Mat re(h, w, CV_8UC3);
        cv::Mat out(model_w, model_h, CV_8UC3, cv::Scalar(114, 114, 114));

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

        // cv::Mat out_float;
        // out.convertTo(out_float, CV_32FC3, 1/255.0);

        // float mean[] = {0.406, 0.456, 0.485};
        // float std[] = {1/0.225, 1/0.224, 1/0.229};
        float mean[] = {103.53, 116.28, 123.675};
        float std[] = {1 / 57.375, 1 / 57.12, 1 / 58.395};
        cv::Mat out_normalize = gddeploy::transform::Normalize(out, mean, std);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_normalize, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

    int PreprocFaceRetrieval(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        cv::Mat re(model_h, model_w, CV_8UC3);

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        float mean[] = {127.5, 127.5, 127.5};
        float std[] = {1 / 127.5, 1 / 127.5, 1 / 127.5};
        cv::Mat out_normalize = gddeploy::transform::Normalize(re, mean, std);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_normalize, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

    int PreprocOcrDet(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        cv::Mat re(model_h, model_w, CV_8UC3);

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        float mean[] = {103.53, 116.28, 123.675};
        float std[] = {1 / 57.375, 1 / 57.12, 1 / 58.395};
        cv::Mat out_normalize = gddeploy::transform::Normalize(re, mean, std);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_normalize, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

    int PreprocOcrRec(cv::Mat input_mat, cv::Mat &output_mat, int model_h, int model_w)
    {
        cv::Mat img;

        if (input_mat.channels() == 1)
            cv::cvtColor(input_mat, img, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(input_mat, img, cv::COLOR_BGR2RGB);

        cv::Mat re(model_h, model_w, CV_8UC3);

        cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);

        float mean[] = {123.675, 116.28, 103.53};
        float std[] = {1 / 58.395, 1 / 57.12, 1 / 57.375};
        cv::Mat out_normalize = gddeploy::transform::Normalize(re, mean, std);

        std::vector<cv::Mat> bgrChannels;
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        bgrChannels.emplace_back(model_w, model_h, CV_32F);
        cv::split(out_normalize, bgrChannels);

        for (int i = 0; i < 3; i++)
        {
            std::memcpy((float *)output_mat.data + i * model_w * model_h, bgrChannels[i].data, model_w * model_h * sizeof(float));
        }

        return 0;
    }

}