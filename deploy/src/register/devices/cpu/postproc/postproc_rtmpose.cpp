#include "postproc_rtmpose.h"

#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"
#include "core/model.h"

#include <math.h>
#include <vector>

#include "opencv2/opencv.hpp"

#include "util/common_nms.h"

using namespace gddeploy;

namespace gddeploy {

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}
namespace rtmpose{

static cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, 
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

int get_point(int box_x, int box_y, int box_w, int box_h,int model_w, int model_h, std::vector<PoseKeyPoint> &points) {
    int box_width = box_w;
	int box_height = box_h;
	int box_center_x = box_x + box_width / 2;
	int box_center_y = box_y + box_height / 2;

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
    cv::Mat affine_transform_reverse = GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		model_w,
		model_h,
		true
	);

    for (auto &point : points){
        cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
		origin_point_Mat.at<double>(0, 0) = point.x;
		origin_point_Mat.at<double>(1, 0) = point.y;

        cv::Mat temp_result_mat = affine_transform_reverse * origin_point_Mat;
        point.x = temp_result_mat.at<double>(0, 0);
		point.y = temp_result_mat.at<double>(1, 0);
    }

    // bbox->x = std::max(0.0f, bbox->x);
    // bbox->y = std::max(0.0f, bbox->y);

    // bbox->w = std::min((float)bbox->x+img_w, bbox->x+bbox->w) - bbox->x;
    // bbox->h = std::min((float)bbox->x+img_h, bbox->y+bbox->h) - bbox->y;
}

}

int decodeNCHWInt8(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                int model_output_w, int model_output_h, float threshold, int b,
                std::vector<float> out_scales, std::vector<int> out_zps,
                DetectPoseImg &detect_img)
{
    BufSurfWrapperPtr x_surf_ptr = out_data[0];
    BufSurfWrapperPtr y_surf_ptr = out_data[1];

    int8_t* simcc_x_result = (int8_t*)x_surf_ptr->GetHostData(b, 0);
	int8_t* simcc_y_result = (int8_t*)y_surf_ptr->GetHostData(b, 0);

    detect_img.detect_objs.resize(1);
    std::vector<PoseKeyPoint> &points = detect_img.detect_objs[0].point;
	for (int i = 0; i < 17; ++i)
	{
		// find the maximum and maximum indexes in the value of each model_output_w length
		auto x_biggest_iter = std::max_element(simcc_x_result + i * model_output_w, simcc_x_result + i * model_output_w + model_output_w);
		int max_x_pos = std::distance(simcc_x_result + i * model_output_w, x_biggest_iter);
		int pose_x = max_x_pos / 2;
		float score_x = deqnt_affine_to_f32(*x_biggest_iter, out_zps[0], out_scales[0]);

		// find the maximum and maximum indexes in the value of each exten_height length
		auto y_biggest_iter = std::max_element(simcc_y_result + i * model_output_h, simcc_y_result + i * model_output_h + model_output_h);
		int max_y_pos = std::distance(simcc_y_result + i * model_output_h, y_biggest_iter);
		int pose_y = max_y_pos / 2;
		float score_y = deqnt_affine_to_f32(*y_biggest_iter, out_zps[1], out_scales[1]);

		//float score = (score_x + score_y) / 2;
		float score = std::max(score_x, score_y);

        points.emplace_back(PoseKeyPoint{pose_x, pose_y, i, score});
	}

    return 0;
}

int RTMPoseDecodeOutput(std::vector<gddeploy::BufSurfWrapperPtr> out_data, 
                                    gddeploy::InferResult &result,  
                                    gddeploy::PostParam &param,  
                                    std::vector<FrameInfo> frame_info, 
                                    gddeploy::ModelPtr model_ptr)
{
    std::vector<std::string> labels = param.labels;
    float threshold = param.conf_thresh;
    float iou_thresh = param.iou_thresh;
    std::vector<float> output_scales = param.output_scale;
    std::vector<int> output_zp = param.output_zp;

    // 1. 获取Model信息

    auto input_shape = model_ptr->InputShape(0);
    const DataLayout output_layout =  model_ptr->InputLayout(0);
    auto dtype = output_layout.dtype;
    auto order = output_layout.order;
    int model_b = input_shape[0];

    int model_w = 0, model_h = 0;

    if (order == gddeploy::DimOrder::NCHW){
        model_w = input_shape[3];
        model_h = input_shape[2];
    } else {
        model_w = input_shape[2];
        model_h = input_shape[1];
    }
    
    for (int b = 0; b < frame_info.size(); b++) {
        DetectPoseImg detect_img; 
        detect_img.img_id = frame_info[b].frame_id;

        detect_img.img_w = frame_info[b].width;
        detect_img.img_h = frame_info[b].height; 
        detect_img.detect_objs.resize(1);

        if (result.detect_result.detect_imgs.size() > 0 && result.detect_result.detect_imgs[0].detect_objs.size() > 0){
            detect_img.detect_objs[0].bbox = result.detect_result.detect_imgs[b].detect_objs[0].bbox;
            detect_img.detect_objs[0].class_id = result.detect_result.detect_imgs[b].detect_objs[0].class_id;
            detect_img.detect_objs[0].score = result.detect_result.detect_imgs[b].detect_objs[0].score;
            detect_img.detect_objs[0].detect_id = result.detect_result.detect_imgs[b].detect_objs[0].detect_id;
        }

        auto output_x_shape = model_ptr->OutputShape(0);
        auto output_y_shape = model_ptr->OutputShape(1);
        int model_out_h = output_y_shape[2];
        int model_out_w = output_x_shape[2];
        
        decodeNCHWInt8(out_data, model_out_w, model_out_h, threshold, b, output_scales, output_zp, detect_img);

        // nms
        // resize to origin img size
        if (detect_img.detect_objs.size() > 0){
            for (auto &obj : detect_img.detect_objs){
                gddeploy::Bbox box = {0, 0, detect_img.img_w, detect_img.img_h};
        
                if (result.detect_result.detect_imgs.size() > 0 && result.detect_result.detect_imgs[0].detect_objs.size() > 0){
                    box = result.detect_result.detect_imgs[0].detect_objs[0].bbox;
                }
                rtmpose::get_point(box.x, box.y, box.w, box.h, model_w, model_h, obj.point);
            }
        }
        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT_POSE);
        result.detect_pose_result.detect_imgs.emplace_back(detect_img);
    }
    result.detect_pose_result.batch_size = frame_info.size();

    return 0;
}
}