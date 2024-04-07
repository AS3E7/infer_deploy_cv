#include "bm_inference.h"
#include <api/infer_api.h>
#include <core/mem/buf_surface_impl.h>
#include <core/mem/buf_surface_util.h>
#include <core/result_def.h>
#include <future>
#include <memory>

namespace gddi {
namespace algo {

struct BmInference::Impl {
    std::unique_ptr<gddeploy::InferAPI> alg_impl;
    AlgoType algo_type{AlgoType::kUndefined};

    std::map<int, std::vector<Rect2f>> cache_target_box;                         // track_id - bbox
    std::map<int, std::vector<std::vector<nodes::PoseKeyPoint>>> cache_key_point;// track_id - key_point
};

BmInference::BmInference() : impl_(std::make_unique<BmInference::Impl>()) {}

BmInference::~BmInference() {}

bool BmInference::init(const ModParms &params) {
    impl_->alg_impl = std::make_unique<gddeploy::InferAPI>();
    impl_->alg_impl->Init("", params.mod_path, "", gddeploy::ENUM_API_TYPE::ENUM_API_SESSION_API);

    try {
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
        } else if (model_type == "ocr") {
            impl_->algo_type = AlgoType::kOCR;
        } else if (model_type == "ocr_det") {
            impl_->algo_type = AlgoType::kOCR_DET;
        } else if (model_type == "ocr_rec") {
            impl_->algo_type = AlgoType::kOCR_REC;
        } else {
            throw std::runtime_error("Undefined model type: " + model_type);
        }
        AbstractAlgo::init(params, impl_->algo_type, impl_->alg_impl->GetLabels());
    } catch (std::exception &ex) {
        spdlog::error("{}", ex.what());
        return false;
    }

    return true;
}

void BmInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                            const InferType type) {
    gddeploy::PackagePtr in = gddeploy::Package::Create(1);
    in->data[0]->Set(gddi::image_wrapper::convert_bm_image_to_sufsurface(*info->src_frame->data));
    in->data[0]->SetUserData(info->infer_frame_idx);

    if (type == InferType::kAsync) {
        impl_->alg_impl->InferAsync(
            in, [this](gddeploy::Status status, gddeploy::PackagePtr data, gddeploy::any user_data) {
                std::vector<algo::AlgoOutput> vec_output;
                parse_infer_result(data, vec_output);
                if (infer_callback_) {
                    infer_callback_(data->data[0]->GetUserData<int64_t>(), impl_->algo_type, vec_output);
                }
            });
    } else {
        auto packet = gddeploy::Package::Create(1);
        impl_->alg_impl->InferSync(in, packet);

        std::vector<algo::AlgoOutput> vec_output;
        parse_infer_result(packet, vec_output);

        if (infer_callback_) { infer_callback_(info->infer_frame_idx, impl_->algo_type, vec_output); }
    }
}

AlgoType BmInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                                std::map<int, std::vector<algo::AlgoOutput>> &outputs) {

    auto &back_ext_info = info->ext_info.back();

    if (!info->ext_info.back().flag_crop) {
        gddeploy::PackagePtr in = gddeploy::Package::Create(1);
        in->data[0]->Set(gddi::image_wrapper::convert_bm_image_to_sufsurface(*info->src_frame->data));
        in->data[0]->SetUserData(info->infer_frame_idx);

        gddeploy::PackagePtr out = gddeploy::Package::Create(1);
        impl_->alg_impl->InferSync(in, out);

        parse_infer_result(out, outputs[info->infer_frame_idx]);
    } else if (back_ext_info.crop_images.second) {
        int index = 0;
        for (const auto &item : *back_ext_info.crop_images.second) {
            gddeploy::PackagePtr in = gddeploy::Package::Create(1);
            in->data[0]->Set(gddi::image_wrapper::convert_bm_image_to_sufsurface(item));

            gddeploy::PackagePtr out = gddeploy::Package::Create(1);
            impl_->alg_impl->InferSync(in, out);

            auto target_id = back_ext_info.crop_images.first.at(index++);
            parse_infer_result(out, outputs[target_id]);

            for (auto &output : outputs[target_id]) {
                output.box.x += back_ext_info.crop_rects.at(target_id).x;
                output.box.y += back_ext_info.crop_rects.at(target_id).y;
            }
        }
    }

    return impl_->algo_type;
}

void BmInference::parse_infer_result(const gddeploy::PackagePtr &packet, std::vector<algo::AlgoOutput> &outputs) {
    if (packet->data[0]->HasMetaValue()) {
        auto result = packet->data[0]->GetMetaData<gddeploy::InferResult>();
        for (auto result_type : result.result_type) {
            if (result_type == gddeploy::GDD_RESULT_TYPE_DETECT) {
                for (const auto &item : result.detect_result.detect_imgs) {
                    for (const auto &item : result.detect_result.detect_imgs) {
                        for (auto &obj : item.detect_objs) {
                            outputs.emplace_back(
                                algo::AlgoOutput{.class_id = obj.class_id,
                                                 .prob = obj.score,
                                                 .box = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h}});
                        }
                    }
                }
            } else if (result_type == gddeploy::GDD_RESULT_TYPE_CLASSIFY) {
                for (const auto &item : result.classify_result.detect_imgs) {
                    for (auto &obj : item.detect_objs) {
                        outputs.emplace_back(
                            algo::AlgoOutput{.class_id = obj.class_id, .prob = obj.score, .box = {0, 0, 0, 0}});
                    }
                }
            } else if (result_type == gddeploy::GDD_RESULT_TYPE_DETECT_POSE) {
                for (const auto &item : result.detect_pose_result.detect_imgs) {
                    for (auto &obj : item.detect_objs) {
                        algo::AlgoOutput output;
                        output.class_id = obj.class_id;
                        output.prob = obj.score;
                        output.box = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};
                        for (auto &keypoint : obj.point) {
                            output.vec_key_points.emplace_back(
                                std::make_tuple(keypoint.number, keypoint.x, keypoint.y, keypoint.score));
                        }
                        outputs.emplace_back(std::move(output));
                    }
                }
            } else if (result_type == gddeploy::GDD_RESULT_TYPE_SEG) {
                for (const auto &item : result.seg_result.seg_imgs) {
                    outputs.emplace_back(algo::AlgoOutput{.seg_width = item.map_w,
                                                          .seg_height = item.map_h,
                                                          .seg_map = std::move(item.seg_map)});
                    // int count = 0;
                    // for (int i = 0; i < res.map_w * res.map_h; i++) {
                    //     if (output.seg_map[i] > 0) { count++; }
                    // }
                    // spdlog::info("seg map count: {}", count);

                    // cv::Mat seg_mat(res.map_h, res.map_w, CV_8UC1, (void *)output.seg_map.data());
                    // cv::Mat thresh_mat;
                    // cv::threshold(seg_mat, thresh_mat, 0, 255, cv::THRESH_BINARY);
                    // cv::imwrite("./seg_mask.jpg", thresh_mat);
                }
            } else if (result_type == gddeploy::GDD_RESULT_TYPE_OCR_DETECT) {
                for (const auto &item : result.ocr_detect_result.ocr_detect_imgs) {
                    for (auto &obj : item.ocr_objs) {
                        algo::AlgoOutput output;
                        output.class_id = obj.class_id;
                        output.prob = obj.score;
                        output.box = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};
                        for (auto &keypoint : obj.point) {
                            output.vec_key_points.emplace_back(
                                std::make_tuple(keypoint.number, keypoint.x, keypoint.y, keypoint.score));
                        }
                        outputs.emplace_back(std::move(output));
                    }
                }
            } else if (result_type == gddeploy::GDD_RESULT_TYPE_ACTION) {
                // TODO: action
            }
        }
    }
}

}// namespace algo
}// namespace gddi