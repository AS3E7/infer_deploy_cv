#include "jet_inference.h"
#include "res.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <future>
#include <memory>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

namespace gddi {
namespace algo {

struct JetInference::Impl {
    std::unique_ptr<AlgImpl> alg_impl;

    int banch_{1};
    AlgoType algo_type{AlgoType::kUndefined};
    std::vector<std::shared_ptr<nodes::FrameInfo>> vec_frame_info;

    std::map<int, std::vector<Rect2f>> cache_target_box;                         // track_id - bbox
    std::map<int, std::vector<std::vector<nodes::PoseKeyPoint>>> cache_key_point;// track_id - key_point

    cv::Mat feature_mat;                  // 人脸特征库
    std::vector<std::string> category_ids;// 人脸标签列表
};

JetInference::JetInference() : impl_(std::make_unique<JetInference::Impl>()) {}

JetInference::~JetInference() {}

bool JetInference::init(const ModParms &parms) {
    impl_->alg_impl = std::make_unique<AlgImpl>();

    try {
        if (impl_->alg_impl->Init(0, "config/alg_config.json", parms.mod_path, impl_->banch_) != GDDI_SUCCESS) {
            throw std::runtime_error("Failed to load model.");
        }
        switch (impl_->alg_impl->GetModelType()) {
            case Task::DETECT_TASK: impl_->algo_type = AlgoType::kDetection; break;
            case Task::CLASSIFY_TASK: impl_->algo_type = AlgoType::kClassification; break;
            case Task::SEGMENTATION_TASK: impl_->algo_type = AlgoType::kSegmentation; break;
            case Task::POSE_TASK: impl_->algo_type = AlgoType::kPose; break;
            case Task::ACTION_TASK: impl_->algo_type = AlgoType::kAction; break;
            case Task::FACE_RECOGNITION_TASK: impl_->algo_type = AlgoType::kFace; break;
            default: throw std::runtime_error("Undefined model type.");
        }
        AbstractAlgo::init(parms, impl_->algo_type, impl_->alg_impl->GetLabels());

        // loading face feature library
        if (impl_->algo_type == AlgoType::kFace) { init_face_future_dataset("resources/face_bank.json"); }
    } catch (std::exception &ex) {
        spdlog::error("{}", ex.what());
        return false;
    }

    return true;
}

void JetInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                             const InferType type) {
    impl_->vec_frame_info.emplace_back(info);
    if (impl_->vec_frame_info.size() < impl_->banch_) { return; }

    std::map<int64_t, cv::Mat> vec_images;
    for (const auto &frame_info : impl_->vec_frame_info) {
        vec_images[frame_info->frame_idx] = *frame_info->src_frame->data;
    }
    impl_->vec_frame_info.clear();

    // auto start = std::chrono::steady_clock::now();
    auto vec_res = inference_implement(task_name, vec_images);
    // spdlog::info("Inference time: {} ms",
    //              std::chrono::duration_cast<std::chrono::milliseconds>(
    //                  std::chrono::steady_clock::now() - start)
    //                  .count());

    int index = 0;
    for (const auto &res : vec_res) {
        std::vector<algo::AlgoOutput> vec_output;
        if (impl_->algo_type == AlgoType::kDetection) {
            auto vec_res = *(std::vector<DetectRes> *)res.data_.get();
            vec_output.resize(vec_res.size());
            for (auto &item : vec_res) { convert_algo_output(item, vec_output[index++]); }
        } else if (impl_->algo_type == AlgoType::kClassification) {
            vec_output.resize(1);
            convert_algo_output(*(ClassifyRes *)res.data_.get(), vec_output[0]);
        } else if (impl_->algo_type == AlgoType::kPose) {
            auto vec_res = *(std::vector<DetectPoseRes> *)res.data_.get();
            vec_output.resize(vec_res.size());
            for (auto &item : vec_res) { convert_algo_output(item, vec_output[index++]); }
        } else if (impl_->algo_type == AlgoType::kSegmentation) {
            vec_output.resize(1);
            convert_algo_output(*(SegmentationRes *)res.data_.get(), vec_output[0]);
        }

        if (infer_callback_) { infer_callback_(info->frame_idx, impl_->algo_type, vec_output); }
    }
}

AlgoType JetInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                                 std::map<int, algo::AlgoOutput> &outputs) {
    auto last_ext_info = info->ext_info.back();
    if (impl_->algo_type == AlgoType::kAction) {
        // 动作识别
        if (last_ext_info.action_type == ActionType::kCount) {
            for (const auto &[track_id, item] : last_ext_info.action_key_points) {
                auto vec_res =
                    inference_implement(task_name, info->width(), info->height(), last_ext_info.map_target_box, item);
                for (const auto &res : vec_res) {
                    convert_algo_output(*(ActionClassifyRes *)res.data_.get(),
                                        outputs[last_ext_info.tracked_box[track_id].target_id]);
                }
            }
        } else if (last_ext_info.action_type == ActionType::kBase) {
            for (const auto &[track_id, item] : last_ext_info.tracked_box) {
                auto &key_point = last_ext_info.map_key_points.at(item.target_id);
                impl_->cache_key_point[track_id].emplace_back(key_point);
                impl_->cache_target_box[track_id].emplace_back(last_ext_info.map_target_box.at(item.target_id).box);
                if (impl_->alg_impl->GetActionClipLen() == impl_->cache_key_point.at(track_id).size()) {
                    TrackerPoseRes tracker_pose_res;
                    int cache_size = impl_->cache_key_point.at(track_id).size();
                    for (int i = 0; i < cache_size; i++) {
                        DetectPoseRes detect_pose_res;
                        for (const auto &key_point : impl_->cache_key_point[track_id][i]) {
                            PoseKeyPoint pos_key_point;
                            pos_key_point.number_ = key_point.number;
                            pos_key_point.x_ = key_point.x;
                            pos_key_point.y_ = key_point.y;
                            pos_key_point.prob_ = key_point.prob;
                            detect_pose_res.points_.emplace_back(pos_key_point);
                            auto &rect = impl_->cache_target_box[track_id][i];
                            detect_pose_res.detect_res_.bbox_[0] = rect.x;
                            detect_pose_res.detect_res_.bbox_[1] = rect.y;
                            detect_pose_res.detect_res_.bbox_[2] = rect.width;
                            detect_pose_res.detect_res_.bbox_[3] = rect.height;
                        }
                        tracker_pose_res.detect_pose_res_.emplace_back(detect_pose_res);
                        tracker_pose_res.frame_ids_.push_back(i);
                    }

                    auto vec_res = inference_implement(task_name, info->width(), info->height(), tracker_pose_res);
                    for (const auto &res : vec_res) {
                        convert_algo_output(*(ActionClassifyRes *)res.data_.get(), outputs[track_id]);
                    }

                    impl_->cache_key_point.at(track_id).erase(impl_->cache_key_point.at(track_id).begin());
                }
            }
        }
    } else if (impl_->algo_type == AlgoType::kFace) {
        for (const auto &[target_id, key_points] : last_ext_info.map_key_points) {
            std::vector<std::shared_ptr<TransferBuffer>> trans_bufs;

            std::vector<float> points;
            for (auto &item : key_points) {
                points.emplace_back(item.x);
                points.emplace_back(item.y);
            }

            trans_bufs.emplace_back(std::make_shared<TransferBuffer>(std::string("0"), info->height(), info->width(),
                                                                     info->src_frame->data->data, 0, false, points));

            std::vector<AlgOutput> alg_res;
            impl_->alg_impl->PreProcessBatch(trans_bufs);
            impl_->alg_impl->ProcessBatch(trans_bufs);
            impl_->alg_impl->PostProcessBatch(trans_bufs, alg_res);

            for (const auto &item : alg_res) {
                convert_algo_output(*(FaceRecognitionRes *)item.data_.get(), outputs[target_id]);
            }
        }
    } else {
        // 图片推理
        auto iter = last_ext_info.crop_images.begin();
        while (iter != last_ext_info.crop_images.end()) {
            std::map<int64_t, cv::Mat> vec_images;
            for (size_t i = 0; i < impl_->banch_ && iter != last_ext_info.crop_images.end(); i++) {
                vec_images.insert(*iter++);
            }

            auto vec_res = inference_implement(task_name, vec_images);

            auto image_iter = vec_images.begin();
            for (const auto &res : vec_res) {
                if (res.task_ == Task::DETECT_TASK) {
                    for (auto &item : *(std::vector<DetectRes> *)res.data_.get()) {
                        item.bbox_[0] += last_ext_info.crop_rects.at(image_iter->first).x;
                        item.bbox_[1] += last_ext_info.crop_rects.at(image_iter->first).y;
                        convert_algo_output(item, outputs[image_iter->first]);
                    }
                } else if (res.task_ == Task::CLASSIFY_TASK) {
                    convert_algo_output(*(ClassifyRes *)res.data_.get(), outputs[image_iter->first]);
                }
            }
        }
    }

    return impl_->algo_type;
}

void JetInference::init_face_future_dataset(const std::string &path) {
    FILE *file_handle = fopen(path.c_str(), "rb");
    if (!file_handle) { throw std::runtime_error("Failed to loading face feature library!"); }

    fseek(file_handle, 0, SEEK_END);
    size_t file_size = ftell(file_handle);
    fseek(file_handle, 0, SEEK_SET);

    auto content = std::vector<char>(file_size);
    fread(content.data(), sizeof(char), file_size, file_handle);
    fclose(file_handle);

    auto json_obj = nlohmann::json::parse(content);
    impl_->feature_mat = cv::Mat(json_obj.size(), 512, CV_32FC1);
    int index = 0;
    for (const auto &item : json_obj) {
        impl_->category_ids.emplace_back(item["category_id"].get<std::string>());
        memcpy(impl_->feature_mat.data + index * 512 * sizeof(float),
               item["embedding"].get<std::vector<float>>().data(), 512 * sizeof(float));
        ++index;
    }
}

std::vector<AlgOutput> JetInference::inference_implement(const std::string &task_name,
                                                         const std::map<int64_t, cv::Mat> &images) {
    std::vector<std::shared_ptr<TransferBuffer>> trans_buf;
    for (const auto &[idx, image] : images) {
        trans_buf.emplace_back(
            std::make_shared<TransferBuffer>(task_name, image.rows, image.cols, image.data, idx, false));
    }

    impl_->alg_impl->PreProcessBatch(trans_buf);
    impl_->alg_impl->ProcessBatch(trans_buf);

    std::vector<AlgOutput> res;
    impl_->alg_impl->PostProcessBatch(trans_buf, res);

    return res;
}

std::vector<AlgOutput>
JetInference::inference_implement(const std::string &task_name, const int width, const int height,
                                  const std::map<int, nodes::BoxInfo> &target_box,
                                  const std::vector<std::vector<nodes::PoseKeyPoint>> &key_points) {
    std::vector<std::shared_ptr<TransferBuffer>> input_buffers;

    std::vector<std::vector<nodes::PoseKeyPoint>> input_key_points;
    int clip_len = impl_->alg_impl->GetActionClipLen();
    if (key_points.size() < clip_len) {
        // 补帧
        do {
            input_key_points.insert(input_key_points.end(), key_points.begin(), key_points.end());
        } while (input_key_points.size() < clip_len);
        input_key_points.resize(20);
    } else if (key_points.size() > clip_len) {
        // 抽帧
        float key_point_size = key_points.size();
        float bsize = key_point_size / clip_len;
        for (float i = 0; i < clip_len; i++) {
            // 随机数
            int rand_num = std::rand() / ((RAND_MAX + 1u) / int(bsize));
            input_key_points.emplace_back(key_points[int(i * bsize + 0.5) + rand_num]);
        }
    } else {
        input_key_points = key_points;
    }

    assert(input_key_points.size() == clip_len);

    int index = 0;
    TrackerPoseRes tracker_pose_res;
    for (const auto &item : input_key_points) {
        DetectPoseRes detect_pose_res;
        for (const auto &key_point : item) {
            PoseKeyPoint pos_key_point;
            pos_key_point.number_ = key_point.number;
            pos_key_point.x_ = key_point.x;
            pos_key_point.y_ = key_point.y;
            pos_key_point.prob_ = key_point.prob;
            detect_pose_res.points_.emplace_back(pos_key_point);
            auto &box = target_box.at(0).box;
            detect_pose_res.detect_res_.bbox_[0] = box.x;
            detect_pose_res.detect_res_.bbox_[1] = box.y;
            detect_pose_res.detect_res_.bbox_[2] = box.width;
            detect_pose_res.detect_res_.bbox_[3] = box.height;
        }
        tracker_pose_res.detect_pose_res_.emplace_back(detect_pose_res);
        tracker_pose_res.frame_ids_.push_back(index++);
    }

    if (input_buffers.size() % impl_->banch_ != 0) {
        input_buffers.emplace_back(
            std::make_shared<TransferBuffer>(std::string("padding"), height, width, -1, tracker_pose_res));
    }

    tracker_pose_res.tracker_id_ = 1;// TODO: 需要跟踪 ID
    input_buffers.emplace_back(std::make_shared<TransferBuffer>(task_name, height, width, 1, tracker_pose_res));

    impl_->alg_impl->PreProcessBatch(input_buffers);
    impl_->alg_impl->ProcessBatch(input_buffers);

    std::vector<AlgOutput> res;
    impl_->alg_impl->PostProcessBatch(input_buffers, res);

    return res;
}

std::vector<AlgOutput> JetInference::inference_implement(const std::string &task_name, const int width,
                                                         const int height, TrackerPoseRes &tracker_pose_res) {
    std::vector<std::shared_ptr<TransferBuffer>> input_buffers;

    tracker_pose_res.tracker_id_ = 1;// TODO: 需要跟踪 ID
    input_buffers.emplace_back(std::make_shared<TransferBuffer>(task_name, height, width, 1, tracker_pose_res));

    impl_->alg_impl->PreProcessBatch(input_buffers);
    impl_->alg_impl->ProcessBatch(input_buffers);

    std::vector<AlgOutput> res;
    impl_->alg_impl->PostProcessBatch(input_buffers, res);

    return res;
}

void JetInference::convert_algo_output(const DetectRes &res, AlgoOutput &output) {
    output.class_id = res.class_id_;
    output.prob = res.prob_;
    output.box = {res.bbox_[0], res.bbox_[1], res.bbox_[2], res.bbox_[3]};
}

void JetInference::convert_algo_output(const ClassifyRes &res, AlgoOutput &output) {
    output.class_id = res.class_id_;
    output.prob = res.prob_;
}

void JetInference::convert_algo_output(const DetectPoseRes &res, AlgoOutput &output) {
    output.class_id = res.detect_res_.class_id_;
    output.prob = res.detect_res_.prob_;
    output.box = {res.detect_res_.bbox_[0], res.detect_res_.bbox_[1], res.detect_res_.bbox_[2],
                  res.detect_res_.bbox_[3]};
    for (auto &keypoint : res.points_) {
        output.vec_key_points.emplace_back(std::make_tuple(keypoint.number_, keypoint.x_, keypoint.y_, keypoint.prob_));
    }
}

void JetInference::convert_algo_output(const SegmentationRes &res, AlgoOutput &output) {
    output.width = res.width_;
    output.height = res.height_;
    output.seg_map = res.res_;
}

void JetInference::convert_algo_output(const FaceRecognitionRes &res, algo::AlgoOutput &output) {
    cv::Mat feature(1, res.ret_.size(), CV_32FC1, const_cast<float *>(res.ret_.data()));
    if (impl_->feature_mat.cols != feature.cols) {
        spdlog::error("find image in database error! compare_database_mat.cols %d output.cols %d\n",
                      impl_->feature_mat.cols, feature.cols);
        return;
    }

    float max_score = -FLT_MAX;
    int index = -1;
    cv::Mat result = impl_->feature_mat * feature.t();
    for (int i = 0; i < result.rows; i++) {
        if (result.at<float>(i, 0) > max_score) {
            max_score = result.at<float>(i, 0);
            index = i;
        }
    }

    output.class_id = index;
    output.label = impl_->category_ids[index];
    output.feature = res.ret_;
    output.prob = max_score;
}

}// namespace algo
}// namespace gddi