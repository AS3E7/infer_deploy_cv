#include "postproc_pose.h"

#include <math.h>

using namespace gddeploy;

static float iou(const Bbox &box1, const Bbox &box2)
{
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.w, box2.x + box2.w);
    float y2 = std::min(box1.y + box1.h, box2.y + box2.h);
    float over_w = std::max(0.0f, x2 - x1);
    float over_h = std::max(0.0f, y2 - y1);
    float over_area = over_w * over_h;
    float iou = over_area / ((box1.w) * (box1.h) + (box2.w) * (box2.h) - over_area);

    return iou;
}

static void get_rect(int img_w, int img_h, int model_w, int model_h, int keypoint_num, DetectPoseObject *obj)
{
    int w, h, x, y;
    float r_w = model_w / (img_w * 1.0);
    float r_h = model_h / (img_h * 1.0);

    Bbox *bbox = &obj->bbox;
    if (r_h > r_w)
    {
        bbox->x = bbox->x / r_w;
        bbox->w = bbox->w / r_w;
        bbox->h = bbox->h / r_w;

        h = r_w * img_h;
        y = 0;
        bbox->y = (bbox->y - y) / r_w;

        for (int k = 0; k < keypoint_num; k++)
        {
            obj->point[k].x /= r_w;
            obj->point[k].y = (obj->point[k].y - y) / r_w;
        }
    }
    else
    {
        bbox->y = bbox->y / r_h;
        bbox->w = bbox->w / r_h;
        bbox->h = bbox->h / r_h;

        w = r_h * img_w;
        x = 0;
        bbox->x = (bbox->x - x) / r_h;

        for (int k = 0; k < keypoint_num; k++)
        {
            obj->point[k].x = (obj->point[k].x - x) / r_h;
            obj->point[k].y /= r_h;
        }
    }

    bbox->x = std::max(0.0f, bbox->x);
    bbox->y = std::max(0.0f, bbox->y);
    bbox->w = std::min((float)img_w, bbox->x + bbox->w) - bbox->x;
    bbox->h = std::min((float)img_h, bbox->y + bbox->h) - bbox->y;
}

static std::vector<DetectPoseObject> nms(std::vector<DetectPoseObject> objInfos, float conf_thresh)
{
    std::sort(objInfos.begin(), objInfos.end(), [](DetectPoseObject lhs, DetectPoseObject rhs)
              { return lhs.score > rhs.score; });
    if (objInfos.size() > 1000)
    {
        objInfos.erase(objInfos.begin() + 1000, objInfos.end());
    }

    std::vector<DetectPoseObject> result;

    while (objInfos.size() > 0)
    {
        result.push_back(objInfos[0]);

        for (auto it = objInfos.begin() + 1; it != objInfos.end();)
        {
            float iou_value = iou(objInfos[0].bbox, it->bbox);
            if (iou_value > conf_thresh)
                it = objInfos.erase(it);
            else
                it++;
        }
        objInfos.erase(objInfos.begin());
    }

    return result;
}

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(const int target_size, std::vector<int> &strides,
                                      std::vector<GridAndStride> &grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid = target_size / stride;
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                grid_strides.emplace_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

int PoseDecodeOutputNHWC(std::vector<gddeploy::BufSurfWrapperPtr> out_data,
                         gddeploy::InferResult &result,
                         gddeploy::any param,
                         std::vector<gddeploy::FrameInfo> frame_info,
                         gddeploy::ModelPtr model_ptr)
{
    if (!out_data[0]->GetHostData(0))
    {
        std::cout << "[gddeploy Samples] [DetectionRunner] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    // 1. 获取Model信息
    auto input_shape = model_ptr->InputShape(0);
    int batch_size = input_shape[0];
    int model_w = input_shape[2];
    int model_h = input_shape[1];
    int model_c = input_shape[3];

    auto output_shape = model_ptr->OutputShape(0);
    int keypoint_num = (output_shape[3] - 6) / 3;
    int mode_out_c = output_shape[3];

    float threshold = any_cast<float>(param);
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(model_w, strides, grid_strides);

    int feature_map_index_grid = 0;
    for (size_t b = 0; b < batch_size; b++)
    {
        int img_w = frame_info[b].width;
        int img_h = frame_info[b].height;

        DetectPoseImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;
        detect_img.img_w = img_w;
        detect_img.img_h = img_h;

        for (int j = 0; j < 3; j++) // output num
        {
            BufSurfWrapperPtr surf_ptr = out_data[j];
            float *src = static_cast<float *>(surf_ptr->GetHostData(0, b));

            // auto s = model->OutputShape(j);
            // printf("output shape:[%ld, %ld, %ld]\n", s[0], s[1], s[2]);

            int feature_map_w = model_w / strides[j];
            int feature_map_h = model_h / strides[j];
            int feature_map_size = feature_map_w * feature_map_h;

            for (int h = 0; h < feature_map_h; h++)
            {
                for (int w = 0; w < feature_map_w; w++)
                {
                    int feature_map_index = h * feature_map_w + w;

                    // int offset = j * feature_num;
                    float conf = *(src + 4 + mode_out_c * feature_map_index);
                    float class_conf = *(src + 5 + mode_out_c * feature_map_index) * conf;
                    if (class_conf > threshold)
                    {
                        DetectPoseObject obj;

                        const int grid0 = grid_strides[feature_map_index_grid + feature_map_index].grid0;
                        const int grid1 = grid_strides[feature_map_index_grid + feature_map_index].grid1;
                        const int stride = grid_strides[feature_map_index_grid + feature_map_index].stride;

                        obj.bbox.w = exp(*(src + 2 + mode_out_c * feature_map_index)) * stride;
                        obj.bbox.h = exp(*(src + 3 + mode_out_c * feature_map_index)) * stride;
                        obj.bbox.x = (*(src + 0 + mode_out_c * feature_map_index) + grid0) * stride - obj.bbox.w * 0.5f;
                        obj.bbox.y = (*(src + 1 + mode_out_c * feature_map_index) + grid1) * stride - obj.bbox.h * 0.5f;
                        obj.score = class_conf;
                        obj.class_id = 0;

                        obj.point.resize(keypoint_num);
                        for (int k = 0; k < keypoint_num; k++)
                        {
                            obj.point[k].x = (*(src + (6 + k * 2 + 0) + mode_out_c * feature_map_index) + grid0) * stride;
                            obj.point[k].y = (*(src + (6 + k * 2 + 1) + mode_out_c * feature_map_index) + grid0) * stride;
                            obj.point[k].number = k;
                            obj.point[k].score = (*(src + (6 + k * 2 + 2) + mode_out_c * feature_map_index) + grid0) * stride;
                        }

                        detect_img.detect_objs.emplace_back(obj);
                    }
                }
            }

            feature_map_index_grid += feature_map_size;
        }
        // nms
        // resize to origin img size
        if (detect_img.detect_objs.size() > 0)
        {
            detect_img.detect_objs = nms(detect_img.detect_objs, 0.5);

            for (auto &obj : detect_img.detect_objs)
            {
                get_rect(img_w, img_h, model_w, model_h, keypoint_num, &obj);
            }
        }

        // InferResult infer_result;
        // infer_result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT_POSE);
        // infer_result.detect_pose_result.detect_imgs.emplace_back(detect_img);
        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT_POSE);
        result.detect_pose_result.detect_imgs.emplace_back(detect_img);
    }

    return 0;
}

int PoseDecodeOutput1NCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data,
                          gddeploy::InferResult &result,
                          gddeploy::any param,
                          std::vector<gddeploy::FrameInfo> frame_info,
                          gddeploy::ModelPtr model_ptr)
{
    if (!out_data[0]->GetHostData(0))
    {
        std::cout << "[gddeploy Samples] [DetectionRunner] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    // 1. 获取Model信息
    auto input_shape = model_ptr->InputShape(0);
    int batch_size = input_shape[0];
    int model_w = input_shape[3];
    int model_h = input_shape[2];
    int model_c = input_shape[1];

    auto output_shape = model_ptr->OutputShape(0);
    int feature_size = output_shape[2];
    int feature_num = output_shape[1];
    int keypoint_num = (feature_size - 6) / 3;

    float threshold = any_cast<float>(param);
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(model_w, strides, grid_strides);

    // int feature_map_index_grid = 0;
    for (size_t b = 0; b < frame_info.size(); b++)
    {
        int img_w = frame_info[b].width;
        int img_h = frame_info[b].height;

        DetectPoseImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;
        detect_img.img_w = img_w;
        detect_img.img_h = img_h;
        BufSurfWrapperPtr surf_ptr = out_data[0];
        float *src = static_cast<float *>(surf_ptr->GetHostData(0, b));
        int conf_num = 0;
        for (int j = 0; j < feature_num; j++)
        {
            int offset = j * feature_size;
            float conf = *(src + offset + 4);
            float class_conf = *(src + offset + 5) * conf;
            if (class_conf > threshold)
            {
                conf_num++;
                const int grid0 = grid_strides[j].grid0;
                const int grid1 = grid_strides[j].grid1;
                const int stride = grid_strides[j].stride;

                float x_center = (*(src + offset + 0) + grid0) * stride;
                float y_center = (*(src + offset + 1) + grid1) * stride;
                float w = exp(*(src + offset + 2)) * stride;
                float h = exp(*(src + offset + 3)) * stride;
                float x0 = x_center - w * 0.5f;
                float y0 = y_center - h * 0.5f;

                DetectPoseObject obj;
                obj.bbox.h = h;
                obj.bbox.w = h;
                obj.bbox.x = x0;
                obj.bbox.y = y0;
                obj.score = class_conf;
                obj.class_id = 0;
                obj.point.resize(keypoint_num);
                for (int k = 0; k < keypoint_num; k++)
                {
                    obj.point[k].x = (*(src + offset + 6 + k * 2 + 0) + grid0) * stride;
                    obj.point[k].y = (*(src + offset + 6 + k * 2 + 1) + grid1) * stride;
                    obj.point[k].number = k;
                    obj.point[k].score = *(src + offset + 6 + keypoint_num * 2 + k);
                }
                detect_img.detect_objs.emplace_back(obj);
            }
        }
        if (detect_img.detect_objs.size() > 0)
        {
            detect_img.detect_objs = nms(detect_img.detect_objs, 0.5);

            for (auto &obj : detect_img.detect_objs)
            {
                get_rect(img_w, img_h, model_w, model_h, keypoint_num, &obj);
            }
        }

        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT_POSE);
        result.detect_pose_result.detect_imgs.emplace_back(detect_img);
    }
    result.detect_result.batch_size = frame_info.size();

    return 0;
}

int PoseDecodeOutputNCHW(std::vector<gddeploy::BufSurfWrapperPtr> out_data,
                         gddeploy::InferResult &result,
                         gddeploy::any param,
                         std::vector<gddeploy::FrameInfo> frame_info,
                         gddeploy::ModelPtr model_ptr)
{
    if (!out_data[0]->GetHostData(0))
    {
        std::cout << "[gddeploy Samples] [DetectionRunner] Postprocess failed, copy data1 to host failed." << std::endl;
        return -1;
    }

    // 1. 获取Model信息
    auto input_shape = model_ptr->InputShape(0);
    int batch_size = input_shape[0];
    int model_w = input_shape[3];
    int model_h = input_shape[2];
    int model_c = input_shape[1];

    auto output_shape = model_ptr->OutputShape(0);
    int keypoint_num = (output_shape[1] - 6) / 3;

    float threshold = any_cast<float>(param);
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(model_w, strides, grid_strides);

    int feature_map_index_grid = 0;
    for (size_t b = 0; b < frame_info.size(); b++)
    {
        int img_w = frame_info[b].width;
        int img_h = frame_info[b].height;

        DetectPoseImg detect_img;
        detect_img.img_id = frame_info[b].frame_id;
        detect_img.img_w = img_w;
        detect_img.img_h = img_h;

        for (int j = 0; j < 3; j++) // output num
        {
            BufSurfWrapperPtr surf_ptr = out_data[j];
            float *src = static_cast<float *>(surf_ptr->GetHostData(0, b));

            // auto s = model->OutputShape(j);
            // printf("output shape:[%ld, %ld, %ld]\n", s[0], s[1], s[2]);

            int feature_map_w = model_w / strides[j];
            int feature_map_h = model_h / strides[j];
            int feature_map_size = feature_map_w * feature_map_h;

            for (int h = 0; h < feature_map_h; h++)
            {
                for (int w = 0; w < feature_map_w; w++)
                {
                    int feature_map_index = h * feature_map_w + w;

                    // int offset = j * feature_num;
                    float conf = *(src + 4 * feature_map_size + feature_map_index);
                    float class_conf = *(src + 5 * feature_map_size + feature_map_index) * conf;
                    if (class_conf > threshold)
                    {
                        DetectPoseObject obj;

                        const int grid0 = grid_strides[feature_map_index_grid + feature_map_index].grid0;
                        const int grid1 = grid_strides[feature_map_index_grid + feature_map_index].grid1;
                        const int stride = grid_strides[feature_map_index_grid + feature_map_index].stride;

                        obj.bbox.w = exp(*(src + 2 * feature_map_size + feature_map_index)) * stride;
                        obj.bbox.h = exp(*(src + 3 * feature_map_size + feature_map_index)) * stride;
                        obj.bbox.x = (*(src + 0 * feature_map_size + feature_map_index) + grid0) * stride - obj.bbox.w * 0.5f;
                        obj.bbox.y = (*(src + 1 * feature_map_size + feature_map_index) + grid1) * stride - obj.bbox.h * 0.5f;

                        obj.score = conf;
                        obj.class_id = 0;

                        obj.point.resize(keypoint_num);
                        for (int k = 0; k < keypoint_num; k++)
                        {
                            obj.point[k].x = (*(src + (6 + k * 2 + 0) * feature_map_size + feature_map_index) + grid0) * stride;
                            obj.point[k].y = (*(src + (6 + k * 2 + 1) * feature_map_size + feature_map_index) + grid1) * stride;
                            obj.point[k].number = k;
                            obj.point[k].score = *(src + (6 + k * 2 + 2) * feature_map_size + feature_map_index);
                        }

                        detect_img.detect_objs.emplace_back(obj);
                    }
                }
            }

            feature_map_index_grid += feature_map_size;
        }
        // nms
        // resize to origin img size
        if (detect_img.detect_objs.size() > 0)
        {
            detect_img.detect_objs = nms(detect_img.detect_objs, 0.5);

            for (auto &obj : detect_img.detect_objs)
            {
                get_rect(img_w, img_h, model_w, model_h, keypoint_num, &obj);
            }
        }

        result.result_type.emplace_back(GDD_RESULT_TYPE_DETECT_POSE);
        result.detect_pose_result.detect_imgs.emplace_back(detect_img);
    }

    return 0;
}
