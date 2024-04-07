#include "rv_inference.h"
#include "alg.h"
#include "modules/algorithm/abstract_algo.h"
#include "node_msg_def.h"
#include "node_struct_def.h"
#include "res.h"
#include <chrono>
#include <cstdio>
#include <exception>
#include <future>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

namespace gddi {
namespace algo {

struct RvInference::Impl {
    std::unique_ptr<AlgImpl> alg_impl;

    AlgoType algo_type{AlgoType::kUndefined};
    std::vector<std::shared_ptr<nodes::FrameInfo>> vec_frame_info;

    std::map<int, std::vector<Rect2f>> cache_target_box;                         // track_id - bbox
    std::map<int, std::vector<std::vector<nodes::PoseKeyPoint>>> cache_key_point;// track_id - key_point
};

RvInference::RvInference() : impl_(std::make_unique<RvInference::Impl>()) {}

RvInference::~RvInference() {}

bool RvInference::init(const ModParms &parms) {
    impl_->alg_impl = std::make_unique<AlgImpl>();

    try {
        if (impl_->alg_impl->Init(0, "config/alg_config.json", parms.mod_path, 1) != GDDI_SUCCESS) {
            throw std::runtime_error("Failed to load model.");
        }
        switch (impl_->alg_impl->GetModelType()) {
            case Task::DETECT_TASK: impl_->algo_type = AlgoType::kDetection; break;
            case Task::CLASSIFY_TASK: impl_->algo_type = AlgoType::kClassification; break;
            case Task::POSE_TASK: impl_->algo_type = AlgoType::kPose; break;
            case Task::ACTION_TASK: impl_->algo_type = AlgoType::kAction; break;
            case Task::PLATE_TASK: impl_->algo_type = AlgoType::kOCR; break;
            case Task::FACE_RECOGNITION_TASK: impl_->algo_type = AlgoType::kFace; break;
            case Task::SEGMENTATION_TASK: impl_->algo_type = AlgoType::kSegmentation; break;
            case Task::OCR_REC_TASK: impl_->algo_type = AlgoType::kOCR; break;
            default: throw std::runtime_error("Undefined model type.");
        }
        AbstractAlgo::init(parms, impl_->algo_type, impl_->alg_impl->GetLabels());
    } catch (std::exception &ex) {
        spdlog::error("{}", ex.what());
        return false;
    }

    return true;
}

void RvInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                            const InferType type) {

    std::shared_ptr<TransferBuffer> trans_buf;
    if (info->src_frame->data->channels() == 1) {
        trans_buf = std::make_shared<TransferBuffer>(task_name, info->height(), info->width(),
                                                     info->src_frame->data->data, info->infer_frame_idx, true);
    } else {
        int width = ALIGN(info->width(), 4);
        float scale = (float)width / info->width();
        int height = info->height();
        cv::resize(*info->src_frame->data, *info->src_frame->data, cv::Size(width, height));
        trans_buf = std::make_shared<TransferBuffer>(task_name, height, width, info->src_frame->data->data,
                                                     info->infer_frame_idx, false);
    }
    impl_->alg_impl->PreProcess(trans_buf);
    impl_->alg_impl->Process(trans_buf);

    AlgOutput alg_res;
    impl_->alg_impl->PostProcess(trans_buf, alg_res);

    std::vector<algo::AlgoOutput> vec_output;

    int index = 0;
    if (alg_res.data_) {
        if (impl_->algo_type == AlgoType::kDetection) {
            auto vec_res = *(std::vector<DetectRes> *)alg_res.data_.get();
            vec_output.resize(vec_res.size());
            for (auto &item : *(std::vector<DetectRes> *)alg_res.data_.get()) {
                convert_algo_output(item, vec_output[index++]);
            }
        } else if (impl_->algo_type == AlgoType::kClassification) {
            vec_output.resize(1);
            convert_algo_output(*(ClassifyRes *)alg_res.data_.get(), vec_output[0]);
        } else if (impl_->algo_type == AlgoType::kPose) {
            auto vec_res = *(std::vector<DetectPoseRes> *)alg_res.data_.get();
            vec_output.resize(vec_res.size());
            for (auto &item : *(std::vector<DetectPoseRes> *)alg_res.data_.get()) {
                convert_algo_output(item, vec_output[index++]);
            }
        }
    }

    if (infer_callback_) { infer_callback_(info->infer_frame_idx, impl_->algo_type, vec_output); }
}

AlgoType RvInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                                std::map<int, std::vector<algo::AlgoOutput>> &outputs) {
    auto &last_ext_info = info->ext_info.back();
    // 图片推理
    int index = 0;
    for (auto &[idx, image] : last_ext_info.crop_images) {
        int width = ALIGN(image.cols, 4);
        float scale = (float)width / image.cols;
        int height = image.rows;

        cv::Mat dst_image;
        cv::resize(image, dst_image, cv::Size(width, height));

        if (!info->ext_info.back().mask_points.empty()) {
            cv::fillPoly(dst_image, info->ext_info.back().mask_points, cv::Scalar(114, 114, 114));
        }

        std::shared_ptr<TransferBuffer> trans_buf;
        if (impl_->algo_type == AlgoType::kFace) {
            std::vector<float> points;
            for (const auto &item : last_ext_info.map_key_points.at(idx)) {
                points.emplace_back(item.x - last_ext_info.crop_rects.at(idx).x);
                points.emplace_back(item.y - last_ext_info.crop_rects.at(idx).y);
            }
            trans_buf = std::make_shared<TransferBuffer>(task_name, height, width, dst_image.data, idx, false, points);
        } else {
            trans_buf = std::make_shared<TransferBuffer>(task_name, height, width, dst_image.data, idx);
        }

        impl_->alg_impl->PreProcess(trans_buf);
        impl_->alg_impl->Process(trans_buf);

        AlgOutput alg_res;
        impl_->alg_impl->PostProcess(trans_buf, alg_res);

        if (alg_res.data_) {
            if (alg_res.task_ == Task::DETECT_TASK) {
                for (auto &item : *(std::vector<DetectRes> *)alg_res.data_.get()) {
                    item.bbox_[0] += last_ext_info.crop_rects.at(idx).x;
                    item.bbox_[1] += last_ext_info.crop_rects.at(idx).y;
                    AlgoOutput output;
                    convert_algo_output(item, output);
                    outputs[idx].emplace_back(output);
                }
            } else if (alg_res.task_ == Task::CLASSIFY_TASK) {
                outputs[idx].resize(1);
                convert_algo_output(*(ClassifyRes *)alg_res.data_.get(), outputs[idx][0]);
            } else if (alg_res.task_ == Task::PLATE_TASK) {
                algo::AlgoOutput output;
                output.prob = 1;
                gddi::nodes::OcrInfo ocr_info;
                auto crop_rect = last_ext_info.crop_rects.at(idx);
                ocr_info.points.emplace_back(nodes::PoseKeyPoint{0, crop_rect.x, crop_rect.y, 1});
                ocr_info.points.emplace_back(nodes::PoseKeyPoint{1, crop_rect.x + crop_rect.width, crop_rect.y, 1});
                ocr_info.points.emplace_back(
                    nodes::PoseKeyPoint{2, crop_rect.x + crop_rect.width, crop_rect.y + crop_rect.height, 1});
                ocr_info.points.emplace_back(nodes::PoseKeyPoint{3, crop_rect.x, crop_rect.y + crop_rect.height, 1});
                ocr_info.labels.emplace_back(nodes::LableInfo{0, 1, ((OcrRes *)alg_res.data_.get())->ret_});
                output.vec_ocr_info.emplace_back(std::move(ocr_info));
                outputs[idx].emplace_back(std::move(output));
            } else if (alg_res.task_ == Task::FACE_RECOGNITION_TASK) {
                outputs[idx].resize(1);
                convert_algo_output(*(FaceRecognitionRes *)alg_res.data_.get(), outputs[idx][0]);
            } else if (alg_res.task_ == Task::SEGMENTATION_TASK) {
                outputs[idx].resize(1);
                outputs[idx][0].class_id = 0;
                outputs[idx][0].prob = 1;
                convert_algo_output(*(SegmentationRes *)alg_res.data_.get(), outputs[idx][0]);
            } else if (alg_res.task_ == Task::OCR_REC_TASK) {
                algo::AlgoOutput output;
                output.prob = 1;
                gddi::nodes::OcrInfo ocr_info;
                auto crop_rect = last_ext_info.crop_rects.at(idx);
                ocr_info.points.emplace_back(nodes::PoseKeyPoint{0, crop_rect.x, crop_rect.y, 1});
                ocr_info.points.emplace_back(nodes::PoseKeyPoint{1, crop_rect.x + crop_rect.width, crop_rect.y, 1});
                ocr_info.points.emplace_back(
                    nodes::PoseKeyPoint{2, crop_rect.x + crop_rect.width, crop_rect.y + crop_rect.height, 1});
                ocr_info.points.emplace_back(nodes::PoseKeyPoint{3, crop_rect.x, crop_rect.y + crop_rect.height, 1});
                ocr_info.labels.emplace_back(nodes::LableInfo{0, 1, ((OcrRes *)alg_res.data_.get())->ret_});
                output.vec_ocr_info.emplace_back(std::move(ocr_info));
                outputs[idx].emplace_back(std::move(output));
            }
        }
    }

    return impl_->algo_type;
}

void RvInference::convert_algo_output(const DetectRes &res, algo::AlgoOutput &output) {
    output.class_id = res.class_id_;
    output.prob = res.prob_;
    output.box = {res.bbox_[0], res.bbox_[1], res.bbox_[2], res.bbox_[3]};
}

void RvInference::convert_algo_output(const ClassifyRes &res, algo::AlgoOutput &output) {
    output.class_id = res.class_id_;
    output.prob = res.prob_;
}

void RvInference::convert_algo_output(const DetectPoseRes &res, algo::AlgoOutput &output) {
    output.class_id = res.detect_res_.class_id_;
    output.prob = res.detect_res_.prob_;
    output.box = {res.detect_res_.bbox_[0], res.detect_res_.bbox_[1], res.detect_res_.bbox_[2],
                  res.detect_res_.bbox_[3]};
    for (auto &keypoint : res.points_) {
        output.vec_key_points.emplace_back(std::make_tuple(keypoint.number_, keypoint.x_, keypoint.y_, keypoint.prob_));
    }
}

void RvInference::convert_algo_output(const FaceRecognitionRes &res, algo::AlgoOutput &output) {
    output.feature = std::move(res.ret_);
}

void RvInference::convert_algo_output(const SegmentationRes &res, algo::AlgoOutput &output) {
    output.seg_width = res.width_;
    output.seg_height = res.height_;
    output.seg_map = res.res_;
}

}// namespace algo
}// namespace gddi