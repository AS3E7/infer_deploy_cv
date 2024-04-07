#include "mlu_inference.h"
#include "gdd_api.h"
#include "gdd_result_type.h"
#include "node_msg_def.h"
#include "spdlog/spdlog.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <future>
#include <memory>
#include <utility>

namespace gddi {
namespace algo {

struct MluInference::Impl {
    std::unique_ptr<gdd::GddInfer> alg_impl;
    InferCallback infer_callback;
    AlgoType algo_type{AlgoType::kUndefined};

    size_t clip_len{20};
    std::map<int, std::vector<Rect2f>> cache_target_box;                         // track_id - bbox
    std::map<int, std::vector<std::vector<nodes::PoseKeyPoint>>> cache_key_point;// track_id - key_point
};

MluInference::MluInference() : impl_(std::make_unique<MluInference::Impl>()) {}

MluInference::~MluInference() {}

bool MluInference::init(const ModParms &parms) {
    impl_->alg_impl = std::make_unique<gdd::GddInfer>();

    impl_->alg_impl->Init(0, 0, "");
    if (impl_->alg_impl->LoadModel(parms.mod_path, "", parms.mod_thres) != 0) {
        spdlog::error("Failed to load model.");
        return false;
    }

    auto model_type = impl_->alg_impl->GetModelType();
    if (model_type == "detection") {
        impl_->algo_type = AlgoType::kDetection;
    } else if (model_type == "classification") {
        impl_->algo_type = AlgoType::kClassification;
    } else if (model_type == "pose") {
        impl_->algo_type = AlgoType::kPose;
    } else if (model_type == "segmentation") {
        impl_->algo_type = AlgoType::kSegmentation;
    } else if (model_type == "action") {
        impl_->algo_type = AlgoType::kAction;
    } else if (model_type == "ocr_det") {
        impl_->algo_type = AlgoType::kOCR_DET;
    } else if (model_type == "ocr_rec") {
        impl_->algo_type = AlgoType::kOCR_REC;
    } else {
        spdlog::error("Undefined model type.");
        return false;
    }

    AbstractAlgo::init(parms, impl_->algo_type, impl_->alg_impl->GetLabels());

    return true;
}

void MluInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                             const InferType type) {
    infer_server::video::VideoFrame video_frame;
    video_frame.width = info->src_frame->data->width;
    video_frame.height = info->src_frame->data->height;
    video_frame.format = infer_server::video::PixelFmt::NV12;
    video_frame.stride[0] = info->src_frame->data->stride[0];
    video_frame.stride[1] = info->src_frame->data->stride[1];
    video_frame.plane_num = info->src_frame->data->planeNum;
    video_frame.plane[0] = infer_server::Buffer((void *)info->src_frame->data->plane[0].addr,
                                                info->src_frame->data->plane[0].size, nullptr, 0);
    video_frame.plane[1] = infer_server::Buffer((void *)info->src_frame->data->plane[1].addr,
                                                info->src_frame->data->plane[1].size, nullptr, 0);

    auto in_packet = infer_server::Package::Create(1);
    in_packet->data[0]->Set(video_frame);
    in_packet->data[0]->SetUserData(info->infer_frame_idx);

    if (type == InferType::kAsync) {
        impl_->alg_impl->InferAsync(
            in_packet,
            [this](infer_server::Status status, infer_server::PackagePtr packet, infer_server::any user_data) {
                for (auto &batch_data : packet->data) {
                    int index = 0;
                    std::vector<algo::AlgoOutput> vec_output;
                    auto &result = batch_data->GetLref<gdd::InferResult>();
                    auto frame_idx = batch_data->GetUserData<int64_t>();

                    if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT) {
                        auto &detect_img = result.detect_result.detect_imgs[0];
                        vec_output.resize(detect_img.detect_objs.size());
                        for (auto &obj : detect_img.detect_objs) { convert_algo_output(obj, vec_output[index++]); }
                    } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_CLASSIFY) {
                        auto &detect_img = result.classify_result.detect_imgs[0];
                        vec_output.resize(detect_img.detect_objs.size());
                        for (auto &obj : detect_img.detect_objs) { convert_algo_output(obj, vec_output[index++]); }
                    } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT_POSE) {
                        auto &detect_img = result.detect_pose_result.detect_imgs[0];
                        vec_output.resize(detect_img.detect_objs.size());
                        for (auto &obj : detect_img.detect_objs) { convert_algo_output(obj, vec_output[index++]); }
                    }

                    if (infer_callback_) { infer_callback_(frame_idx, impl_->algo_type, vec_output); };
                }
            });
    } else {
        infer_server::PackagePtr out_packet = infer_server::Package::Create(1);
        impl_->alg_impl->InferSync(in_packet, out_packet);

        std::vector<algo::AlgoOutput> vec_output;
        auto result = out_packet->data[0]->GetLref<gdd::InferResult>();
        auto frame_idx = out_packet->data[0]->GetUserData<int64_t>();

        int index = 0;
        if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT) {
            auto &detect_img = result.detect_result.detect_imgs[0];
            vec_output.resize(detect_img.detect_objs.size());
            for (auto &obj : detect_img.detect_objs) { convert_algo_output(obj, vec_output[index++]); }
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kDetection, vec_output); };
        } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_CLASSIFY) {
            auto &detect_img = result.classify_result.detect_imgs[0];
            vec_output.resize(detect_img.detect_objs.size());
            for (auto &obj : detect_img.detect_objs) { convert_algo_output(obj, vec_output[index++]); }
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kClassification, vec_output); };
        } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT_POSE) {
            auto &detect_img = result.detect_pose_result.detect_imgs[0];
            vec_output.resize(detect_img.detect_objs.size());
            for (auto &obj : detect_img.detect_objs) { convert_algo_output(obj, vec_output[index++]); }
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kPose, vec_output); };
        } else {
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kUndefined, vec_output); };
        }
    }
}

AlgoType MluInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                                 std::map<int, std::vector<algo::AlgoOutput>> &outputs) {
    auto last_ext_info = info->ext_info.back();
    if (impl_->algo_type == AlgoType::kAction) {
        // 动作识别
        if (last_ext_info.action_type == ActionType::kCount) {
            for (const auto &[track_id, item] : last_ext_info.action_key_points) {
                auto key_points = complementary_frame(item, impl_->clip_len);
                auto infer_result = structure_infer_result(last_ext_info.map_target_box, key_points);
                auto vec_res = inference_implement(task_name, std::move(infer_result));
                for (const auto &item : vec_res.classify_result.detect_imgs[0].detect_objs) {
                    AlgoOutput output;
                    convert_algo_output(item, output);
                    outputs[last_ext_info.tracked_box[track_id].target_id].emplace_back(output);
                }
            }
        } else if (last_ext_info.action_type == ActionType::kBase) {
            for (const auto &[track_id, item] : last_ext_info.tracked_box) {
                auto &key_point = last_ext_info.map_key_points.at(item.target_id);
                impl_->cache_key_point[track_id].emplace_back(key_point);
                impl_->cache_target_box[track_id].emplace_back(last_ext_info.map_target_box.at(item.target_id).box);
                if (impl_->cache_key_point.at(track_id).size() == impl_->clip_len) {
                    auto infer_result =
                        structure_infer_result(last_ext_info.map_target_box, impl_->cache_key_point[track_id]);
                    auto vec_res = inference_implement(task_name, std::move(infer_result));
                    for (const auto &item : vec_res.classify_result.detect_imgs[0].detect_objs) {
                        AlgoOutput output;
                        convert_algo_output(item, output);
                        outputs[track_id].emplace_back(output);
                    }

                    impl_->cache_key_point.at(track_id).erase(impl_->cache_key_point.at(track_id).begin());
                }
            }
        }
    } else if (!info->ext_info.back().flag_crop) {
        if (!last_ext_info.map_target_box.empty()) {
            infer_server::video::VideoFrame video_frame;
            video_frame.width = info->src_frame->data->width;
            video_frame.height = info->src_frame->data->height;
            video_frame.format = infer_server::video::PixelFmt::NV12;
            video_frame.stride[0] = info->src_frame->data->stride[0];
            video_frame.stride[1] = info->src_frame->data->stride[1];
            video_frame.plane_num = info->src_frame->data->planeNum;
            video_frame.plane[0] = infer_server::Buffer((void *)info->src_frame->data->plane[0].addr,
                                                        info->src_frame->data->plane[0].size, nullptr, 0);
            video_frame.plane[1] = infer_server::Buffer((void *)info->src_frame->data->plane[1].addr,
                                                        info->src_frame->data->plane[1].size, nullptr, 0);

            auto in_packet = infer_server::Package::Create(1);
            in_packet->data[0]->Set(video_frame);

            infer_server::PackagePtr out_packet = infer_server::Package::Create(1);
            impl_->alg_impl->InferSync(in_packet, out_packet);

            auto result = out_packet->data[0]->GetLref<gdd::InferResult>();
            if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT) {
                auto &detect_img = result.detect_result.detect_imgs[0];
                for (auto &item : detect_img.detect_objs) {
                    AlgoOutput output;
                    item.bbox.x += last_ext_info.crop_rects.at(target_id).x;
                    item.bbox.y += last_ext_info.crop_rects.at(target_id).y;
                    convert_algo_output(item, output);
                    outputs[target_id].emplace_back(output);
                }
            } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_CLASSIFY) {
                auto &detect_img = result.classify_result.detect_imgs[0];
                for (auto &item : detect_img.detect_objs) {
                    AlgoOutput output;
                    convert_algo_output(item, output);
                    outputs[target_id].emplace_back(output);
                }
            }
        }
    } else {
        for (const auto &[target_id, image] : last_ext_info.crop_images) {
            infer_server::video::VideoFrame video_frame;
            video_frame.format = infer_server::video::PixelFmt::NV12;
            video_frame.width = image->width;
            video_frame.height = image->height;
            video_frame.plane_num = 2;
            video_frame.stride[0] = image->stride[0];
            video_frame.stride[1] = image->stride[1];
            video_frame.plane[0] = infer_server::Buffer((void *)image->plane[0].addr,
                                                        video_frame.height * video_frame.stride[0], nullptr, 0);
            video_frame.plane[1] = infer_server::Buffer((void *)image->plane[1].addr,
                                                        video_frame.height * video_frame.stride[1] / 2, nullptr, 0);

            auto in_packet = infer_server::Package::Create(1);
            in_packet->data[0]->Set(video_frame);

            infer_server::PackagePtr out_packet = infer_server::Package::Create(1);
            impl_->alg_impl->InferSync(in_packet, out_packet);

            auto result = out_packet->data[0]->GetLref<gdd::InferResult>();
            if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT) {
                auto &detect_img = result.detect_result.detect_imgs[0];
                for (auto &item : detect_img.detect_objs) {
                    AlgoOutput output;
                    item.bbox.x += last_ext_info.crop_rects.at(target_id).x;
                    item.bbox.y += last_ext_info.crop_rects.at(target_id).y;
                    convert_algo_output(item, output);
                    outputs[target_id].emplace_back(output);
                }
            } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_CLASSIFY) {
                auto &detect_img = result.classify_result.detect_imgs[0];
                for (auto &item : detect_img.detect_objs) {
                    AlgoOutput output;
                    convert_algo_output(item, output);
                    outputs[target_id].emplace_back(output);
                }
            }
        }
    }

    return impl_->algo_type;
}

gdd::InferResult MluInference::structure_infer_result(const std::map<int, nodes::BoxInfo> &target_box,
                                                      const std::vector<std::vector<nodes::PoseKeyPoint>> &key_points) {
    gdd::InferResult infer_result;
    infer_result.result_type.emplace_back(gdd::GDD_RESULT_TYPE_DETECT_POSE);

    gdd::DetectPoseImg detect_img;
    for (const auto &item : key_points) {
        gdd::DetectPoseObject det_pose_obj;
        for (const auto &key_point : item) {
            det_pose_obj.point.emplace_back(
                gdd::PoseKeyPoint{key_point.x, key_point.y, key_point.number, key_point.prob});
        }

        auto &box = target_box.at(0).box;
        det_pose_obj.bbox.x = box.x;
        det_pose_obj.bbox.y = box.y;
        det_pose_obj.bbox.w = box.width;
        det_pose_obj.bbox.h = box.height;

        detect_img.detect_objs.emplace_back(det_pose_obj);
    }

    infer_result.detect_pose_result.detect_imgs.emplace_back(detect_img);

    return infer_result;
}

gdd::InferResult MluInference::inference_implement(const std::string &task_name, gdd::InferResult &&infer_result) {
    infer_server::PackagePtr in_packet = infer_server::Package::Create(1);
    in_packet->data[0]->Set(std::move(infer_result));
    in_packet->tag = "no img";

    infer_server::PackagePtr out_packet = infer_server::Package::Create(1);
    impl_->alg_impl->InferSync(in_packet, out_packet);

    return out_packet->data[0]->GetLref<gdd::InferResult>();
}

void MluInference::convert_algo_output(const gdd::DetectObject &obj, algo::AlgoOutput &output) {
    output.class_id = obj.class_id;
    output.prob = obj.score;
    output.box = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};
}

void MluInference::convert_algo_output(const gdd::ClassifyObject &obj, algo::AlgoOutput &output) {
    output.class_id = obj.class_id;
    output.prob = obj.score;
}

void MluInference::convert_algo_output(const gdd::DetectPoseObject &obj, algo::AlgoOutput &output) {
    output.class_id = output.class_id;
    output.prob = obj.score;
    output.box = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};
    for (const auto &point : obj.point) {
        output.vec_key_points.emplace_back(std::make_tuple(point.number, point.x, point.y, point.prob));
    }
}

}// namespace algo
}// namespace gddi