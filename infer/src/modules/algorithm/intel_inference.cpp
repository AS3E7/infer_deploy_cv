#include "intel_inference.h"
#include "gddi_api.h"
#include <future>
#include <memory>

namespace gddi {
namespace algo {

struct IntelInference::Impl {
    std::unique_ptr<gdd::GddInfer> alg_impl;
};

IntelInference::IntelInference() : impl_(std::make_unique<IntelInference::Impl>()) {}

IntelInference::~IntelInference() { impl_->alg_impl->InferAsyncWait(); }

bool IntelInference::init(const ModParms &parms) {
    impl_->alg_impl = std::make_unique<gdd::GddInfer>();
    impl_->alg_impl->Init("", 0, 0, "");
    impl_->alg_impl->SetCallback([this](gdd::InferResult *result, std::shared_ptr<void> user_data) {
        auto frame_idx = *std::reinterpret_pointer_cast<int64_t>(user_data);
        if (result->detect_result_.detectObjects_.empty()) {
            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kDetection, {}); }
        } else {
            int index = 0;
            auto &object = result->detect_result_.detectObjects_[0];
            auto vec_output = std::vector<algo::AlgoOutput>(object.detect_objs.size());
            for (auto &item : object.detect_objs) {
                vec_output[index++] = algo::AlgoOutput{.class_id = item.class_id,
                                                       .prob = item.score,
                                                       .box = {item.bbox.x, item.bbox.y, item.bbox.w, item.bbox.h}};
            }

            if (infer_callback_) { infer_callback_(frame_idx, AlgoType::kDetection, vec_output); }
        }
    });

    if (impl_->alg_impl->LoadModel(parms.mod_path, "") != 0) {
        spdlog::error("Failed to load model.");
        return false;
    }

    AbstractAlgo::init(parms, AlgoType::kDetection, impl_->alg_impl->GetLabels());

    return true;
}

void IntelInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                               const InferType type) {
    impl_->alg_impl->InferAsync(info->src_frame->data.get(), std::make_shared<int64_t>(info->infer_frame_idx));
}

AlgoType IntelInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                                   std::map<int, std::vector<algo::AlgoOutput>> &outputs) {
    int index = 0;
    for (const auto &[idx, image] : info->ext_info.back().crop_images) {
        gdd::InferResult result;
        impl_->alg_impl->InferSync(const_cast<cv::Mat *>(&image), &result);

        auto &object = result.detect_result_.detectObjects_[0];
        for (auto &item : object.detect_objs) {
            item.bbox.x += info->ext_info.back().crop_rects[index].x;
            item.bbox.y += info->ext_info.back().crop_rects[index].y;
            outputs[index].emplace_back(algo::AlgoOutput{.class_id = item.class_id,
                                                         .prob = item.score,
                                                         .box = {item.bbox.x, item.bbox.y, item.bbox.w, item.bbox.h}});
        }

        ++index;
    }

    return AlgoType::kDetection;
}

}// namespace algo
}// namespace gddi