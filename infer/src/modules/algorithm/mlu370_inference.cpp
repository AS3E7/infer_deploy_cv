#include "mlu370_inference.h"
#include "gdd_api.h"
#include "gdd_result_type.h"
#include "node_msg_def.h"

#include <cstdint>
#include <future>
#include <memory>

namespace gddi {
namespace algo {

struct MluInference::Impl {
    std::unique_ptr<gdd::GddInfer> alg_impl;
    InferCallback infer_callback;
    AlgoType algo_type{AlgoType::kUndefined};
};

MluInference::MluInference() : impl_(std::make_unique<MluInference::Impl>()) {}

MluInference::~MluInference() {}

bool MluInference::init(const ModParms &parms) {
    impl_->alg_impl = std::make_unique<gdd::GddInfer>();

    impl_->alg_impl->Init(0, 0, "");
    if (impl_->alg_impl->LoadModel(parms.mod_path, "") != 0) {
        spdlog::error("Failed to load model.");
        return false;
    }

    AbstractAlgo::init(parms, impl_->algo_type, impl_->alg_impl->GetLabels());

    return true;
}

void MluInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                             const InferType type) {
    infer_server::PreprocInput preproc_input;
    preproc_input.surf = std::make_shared<cnedk::BufSurfaceWrapper>(info->src_frame->data.get(), false);
    preproc_input.has_bbox = false;

    auto in_packet = infer_server::Package::Create(1);
    in_packet->data[0]->Set(preproc_input);
    in_packet->data[0]->SetUserData(info->frame_idx);

    if (type == InferType::kAsync) {
        impl_->alg_impl->InferAsync(
            in_packet,
            [this](infer_server::Status status, infer_server::PackagePtr packet, infer_server::any user_data) {
                for (auto &batch_data : packet->data) {
                    std::vector<algo::AlgoOutput> vec_output;
                    auto &result = batch_data->GetLref<gdd::InferResult>();
                    auto frame_idx = batch_data->GetUserData<int64_t>();

                    int index = 0;
                    if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT) {
                        vec_output.resize(result.detect_result.detect_imgs.size());
                        for (auto &item : result.detect_result.detect_imgs) {
                            for (auto &obj : item.detect_objs) {
                                convert_algo_output(item.img_id, obj, vec_output[index]);
                                ++index;
                            }
                        }
                        if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kDetection, vec_output); };
                    } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_CLASSIFY) {
                        vec_output.resize(result.classify_result.detect_imgs.size());
                        for (auto &item : result.classify_result.detect_imgs) {
                            for (auto &obj : item.detect_objs) {
                                convert_algo_output(item.img_id, obj, vec_output[index]);
                                ++index;
                            }
                        }
                        if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kClassification, vec_output); };
                    } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT_POSE) {
                        vec_output.resize(result.detect_pose_result.detect_imgs.size());
                        for (auto &item : result.detect_pose_result.detect_imgs) {
                            for (auto &obj : item.detect_objs) {
                                convert_algo_output(item.img_id, obj, vec_output[index]);
                                ++index;
                            }
                        }
                        if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kPose, vec_output); };
                    } else {
                        if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kUndefined, vec_output); };
                    }
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
            vec_output.resize(result.detect_result.detect_imgs.size());
            for (auto &item : result.detect_result.detect_imgs) {
                for (auto &obj : item.detect_objs) {
                    convert_algo_output(item.img_id, obj, vec_output[index]);
                    ++index;
                }
            }
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kDetection, vec_output); };
        } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_CLASSIFY) {
            vec_output.resize(result.classify_result.detect_imgs.size());
            for (auto &item : result.classify_result.detect_imgs) {
                for (auto &obj : item.detect_objs) {
                    convert_algo_output(item.img_id, obj, vec_output[index]);
                    ++index;
                }
            }
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kClassification, vec_output); };
        } else if (result.result_type[0] == gdd::GDD_RESULT_TYPE_DETECT_POSE) {
            vec_output.resize(result.detect_pose_result.detect_imgs.size());
            for (auto &item : result.detect_pose_result.detect_imgs) {
                for (auto &obj : item.detect_objs) {
                    convert_algo_output(item.img_id, obj, vec_output[index]);
                    ++index;
                }
            }
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kPose, vec_output); };
        } else {
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kUndefined, vec_output); };
        }
    }
}

AlgoType MluInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                                 std::map<int, algo::AlgoOutput> &outputs) {
    auto last_ext_info = info->ext_info.back();
    for (const auto &[idx, image] : last_ext_info.crop_images) {

        infer_server::PreprocInput preproc_input;
        preproc_input.surf = std::make_shared<cnedk::BufSurfaceWrapper>(image.get(), true);
        preproc_input.has_bbox = false;

        auto in_packet = infer_server::Package::Create(1);
        in_packet->data[0]->Set(preproc_input);

        infer_server::PackagePtr out_packet = infer_server::Package::Create(1);
        impl_->alg_impl->InferSync(in_packet, out_packet);

        int index = 0;
        for (auto &item : out_packet->data[0]->GetLref<std::vector<gdd::DetectObject>>()) {
            item.bbox.x += last_ext_info.crop_rects.at(idx).x;
            item.bbox.y += last_ext_info.crop_rects.at(idx).y;
            convert_algo_output(idx, item, outputs[index++]);
        }
    }

    return impl_->algo_type;
}

void MluInference::convert_algo_output(const int prev_id, const gdd::DetectObject &obj, algo::AlgoOutput &output) {
    output.prev_id = prev_id;
    output.class_id = obj.class_id;
    output.prob = obj.score;
    output.box = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};
}

void MluInference::convert_algo_output(const int prev_id, const gdd::ClassifyObject &obj, algo::AlgoOutput &output) {
    output.prev_id = prev_id;
    output.class_id = obj.class_id;
    output.prob = obj.score;
}

void MluInference::convert_algo_output(const int prev_id, const gdd::DetectPoseObject &obj, algo::AlgoOutput &output) {
    output.prev_id = prev_id;
    output.class_id = obj.class_id;
    output.prob = obj.score;
    output.box = {obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h};
    for (const auto &point : obj.point) {
        output.vec_key_points.emplace_back(std::make_tuple(point.number, point.x, point.y, point.prob));
    }
}

}// namespace algo
}// namespace gddi