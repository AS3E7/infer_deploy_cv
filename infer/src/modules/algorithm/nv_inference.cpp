#include "nv_inference.h"
#include "alg.h"
#include "modules/algorithm/abstract_algo.h"
#include "node_msg_def.h"
#include "node_struct_def.h"
#include "res.h"
#include <boost/filesystem.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

namespace gddi {
namespace algo {

const size_t kFeatureSize = 512;
static std::mutex device_mtx;
static std::map<int, int> device_counter;

struct NvInference::Impl {
    std::unique_ptr<AlgImpl> alg_impl;

    AlgoType algo_type{AlgoType::kUndefined};
    std::vector<std::shared_ptr<nodes::FrameInfo>> vec_frame_info;

    std::map<int, std::vector<Rect2f>> cache_target_box;                         // track_id - bbox
    std::map<int, std::vector<std::vector<nodes::PoseKeyPoint>>> cache_key_point;// track_id - key_point

    cv::Mat feature_mat;                  // 人脸特征库
    std::vector<std::string> category_ids;// 人脸标签列表

    uint32_t device_id{0};
    uint32_t batch_size{1};
};

NvInference::NvInference() : impl_(std::make_unique<NvInference::Impl>()) {
    // std::lock_guard<std::mutex> glk(device_mtx);
    // int device_count;
    // cudaGetDeviceCount(&device_count);

    // std::pair<int, int> cur_device{0, 512};
    // for (int i = 0; i < device_count; i++) {
    //     if (device_counter.count(i) == 0) { device_counter[i] = 0; }
    //     if (device_counter.at(i) < cur_device.second) {
    //         cur_device.first = i;
    //         cur_device.second = device_counter.at(i);
    //     }
    // }
    // impl_->device_id = cur_device.first;
    impl_->device_id = 0;
    // ++device_counter.at(impl_->device_id);
}

NvInference::~NvInference() {
    // std::lock_guard<std::mutex> glk(device_mtx);
    // --device_counter.at(impl_->device_id);
}

bool NvInference::init(const ModParms &parms) {
    impl_->batch_size = parms.batch_size;
    impl_->alg_impl = std::make_unique<AlgImpl>();

    try {
        if (impl_->alg_impl->Init(impl_->device_id, "config/alg_config.json", parms.mod_path, parms.batch_size)
            != GDDI_SUCCESS) {
            throw std::runtime_error("Failed to load model.");
        }
        switch (impl_->alg_impl->GetModelType()) {
            case Task::DETECT_TASK: impl_->algo_type = AlgoType::kDetection; break;
            case Task::CLASSIFY_TASK: impl_->algo_type = AlgoType::kClassification; break;
            case Task::SEGMENTATION_TASK: impl_->algo_type = AlgoType::kSegmentation; break;
            case Task::POSE_TASK: impl_->algo_type = AlgoType::kPose; break;
            case Task::ACTION_TASK: impl_->algo_type = AlgoType::kAction; break;
            case Task::FACE_RECOGNITION_TASK: impl_->algo_type = AlgoType::kFace; break;
            case Task::OCR_REC_TASK: impl_->algo_type = AlgoType::kOCR_REC; break;
            default: throw std::runtime_error("Undefined model type.");
        }
        AbstractAlgo::init(parms, impl_->algo_type, impl_->alg_impl->GetLabels());
        if (impl_->algo_type == AlgoType::kFace) { init_feature_library(parms.lib_paths); }
    } catch (std::exception &ex) {
        spdlog::error("{}", ex.what());
        return false;
    }

    return true;
}

void NvInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                            const InferType type) {
    impl_->vec_frame_info.emplace_back(info);
    if (impl_->vec_frame_info.size() < impl_->batch_size) { return; }

    std::map<int64_t, cv::cuda::GpuMat> vec_images;
    for (const auto &frame_info : impl_->vec_frame_info) {
        vec_images[frame_info->infer_frame_idx] = *frame_info->src_frame->data;
    }
    impl_->vec_frame_info.clear();

    // auto start = std::chrono::steady_clock::now();
    auto vec_res = inference_implement(task_name, vec_images);
    // spdlog::info("Inference time: {} ms",
    //              std::chrono::duration_cast<std::chrono::milliseconds>(
    //                  std::chrono::steady_clock::now() - start)
    //                  .count());

    auto iter = vec_images.begin();
    for (const auto &res : vec_res) {
        std::vector<algo::AlgoOutput> vec_output;
        if (impl_->algo_type == AlgoType::kDetection) {
            auto vec_res = *(std::vector<DetectRes> *)res.data_.get();
            int index = 0;
            vec_output.resize(vec_res.size());
            for (auto &item : vec_res) { convert_algo_output(item, vec_output[index++]); }
        } else if (impl_->algo_type == AlgoType::kClassification) {
            vec_output.resize(1);
            convert_algo_output(*(ClassifyRes *)res.data_.get(), vec_output[0]);
        } else if (impl_->algo_type == AlgoType::kPose) {
            auto vec_res = *(std::vector<DetectPoseRes> *)res.data_.get();
            int index = 0;
            vec_output.resize(vec_res.size());
            for (auto &item : vec_res) { convert_algo_output(item, vec_output[index++]); }
        } else if (impl_->algo_type == AlgoType::kSegmentation) {
            vec_output.resize(1);
            convert_algo_output(*(SegmentationRes *)res.data_.get(), vec_output[0]);
        }

        if (infer_callback_) { infer_callback_(iter->first, impl_->algo_type, vec_output); }

        ++iter;
    }
}

AlgoType NvInference::inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                                std::map<int, std::vector<algo::AlgoOutput>> &outputs) {
    const auto last_ext_info = info->ext_info.back();
    if (impl_->algo_type == AlgoType::kFace) {
        for (const auto &[target_id, key_points] : last_ext_info.map_key_points) {
            std::vector<std::shared_ptr<TransferBuffer>> trans_bufs;

            std::vector<float> points;
            for (auto &item : key_points) {
                points.emplace_back(item.x);
                points.emplace_back(item.y);
            }

            cv::Mat face_mat;
            info->src_frame->data->download(face_mat);
            trans_bufs.emplace_back(std::make_shared<TransferBuffer>(std::string("0"), info->height(), info->width(),
                                                                     face_mat.data, 0, false, points));

            std::vector<AlgOutput> alg_res;
            impl_->alg_impl->PreProcessBatch(trans_bufs);
            impl_->alg_impl->ProcessBatch(trans_bufs);
            impl_->alg_impl->PostProcessBatch(trans_bufs, alg_res);

            for (const auto &item : alg_res) {
                AlgoOutput output;
                convert_algo_output(*(FaceRecognitionRes *)item.data_.get(), output);
                outputs[target_id].emplace_back(output);
            }
        }
    } else if (impl_->algo_type == AlgoType::kAction) {
        // 动作识别
        if (last_ext_info.action_type == ActionType::kCount) {
            for (const auto &[track_id, item] : last_ext_info.action_key_points) {
                auto vec_res =
                    inference_implement(task_name, info->width(), info->height(), last_ext_info.map_target_box, item);
                for (const auto &res : vec_res) {
                    AlgoOutput output;
                    convert_algo_output(*(ActionClassifyRes *)res.data_.get(), output);
                    outputs[last_ext_info.tracked_box.at(track_id).target_id].emplace_back(output);
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
                        AlgoOutput output;
                        convert_algo_output(*(ActionClassifyRes *)res.data_.get(), output);
                        outputs[last_ext_info.tracked_box.at(track_id).target_id].emplace_back(output);
                    }

                    impl_->cache_key_point.at(track_id).erase(impl_->cache_key_point.at(track_id).begin());
                }
            }
        }
    } else if (!info->ext_info.back().flag_crop) {
        if (!last_ext_info.map_target_box.empty()) {
            std::map<int64_t, cv::cuda::GpuMat> vec_images;
            for (size_t i = 0; i < impl_->batch_size; ++i) { vec_images[i] = *info->src_frame->data; }

            auto vec_res = inference_implement(task_name, vec_images);

            auto image_iter = vec_images.begin();
            for (const auto &res : vec_res) {
                if (res.task_ == Task::DETECT_TASK) {
                    for (auto &item : *(std::vector<DetectRes> *)res.data_.get()) {
                        AlgoOutput output;
                        convert_algo_output(item, output);
                        outputs[image_iter->first].emplace_back(output);
                    }
                } else if (res.task_ == Task::CLASSIFY_TASK) {
                    AlgoOutput output;
                    convert_algo_output(*(ClassifyRes *)res.data_.get(), output);
                    outputs[image_iter->first].emplace_back(output);
                }
            }
        }
    } else {
        // 图片推理
        auto iter = last_ext_info.crop_images.begin();
        while (iter != last_ext_info.crop_images.end()) {
            std::map<int64_t, cv::cuda::GpuMat> vec_images;
            for (size_t i = 0; i < impl_->batch_size && iter != last_ext_info.crop_images.end(); i++) {
                vec_images.insert(*iter++);
            }

            auto vec_res = inference_implement(task_name, vec_images);

            auto image_iter = vec_images.begin();
            for (const auto &res : vec_res) {
                if (res.task_ == Task::DETECT_TASK) {
                    for (auto &item : *(std::vector<DetectRes> *)res.data_.get()) {
                        AlgoOutput output;
                        item.bbox_[0] += last_ext_info.crop_rects.at(image_iter->first).x;
                        item.bbox_[1] += last_ext_info.crop_rects.at(image_iter->first).y;
                        convert_algo_output(item, output);
                        outputs[image_iter->first].emplace_back(output);
                    }
                } else if (res.task_ == Task::CLASSIFY_TASK) {
                    AlgoOutput output;
                    convert_algo_output(*(ClassifyRes *)res.data_.get(), output);
                    outputs[image_iter->first].emplace_back(output);
                } else if (res.task_ == Task::POSE_TASK) {
                    for (auto &item : *(std::vector<DetectPoseRes> *)res.data_.get()) {
                        AlgoOutput output;
                        item.detect_res_.bbox_[0] += last_ext_info.crop_rects.at(image_iter->first).x;
                        item.detect_res_.bbox_[1] += last_ext_info.crop_rects.at(image_iter->first).y;
                        for (auto &key_point : item.points_) {
                            key_point.x_ += last_ext_info.crop_rects.at(image_iter->first).x;
                            key_point.y_ += last_ext_info.crop_rects.at(image_iter->first).y;
                        }
                        convert_algo_output(item, output);
                        outputs[image_iter->first].emplace_back(output);
                    }
                } else if (res.task_ == Task::OCR_REC_TASK) {
                    auto ocr_res = (OcrRes *)res.data_.get();
                    auto ocr_rect = last_ext_info.crop_rects.at(image_iter->first);

                    nodes::OcrInfo ocr_info;
                    ocr_info.points.emplace_back(
                        nodes::PoseKeyPoint{.number = 0, .x = ocr_rect.x, .y = ocr_rect.y, .prob = 1});
                    ocr_info.points.emplace_back(
                        nodes::PoseKeyPoint{.number = 1, .x = ocr_rect.x + ocr_rect.width, .y = ocr_rect.y, .prob = 1});
                    ocr_info.points.emplace_back(nodes::PoseKeyPoint{.number = 2,
                                                                     .x = ocr_rect.x + ocr_rect.width,
                                                                     .y = ocr_rect.y + ocr_rect.height,
                                                                     .prob = 1});
                    ocr_info.points.emplace_back(nodes::PoseKeyPoint{.number = 3,
                                                                     .x = ocr_rect.x,
                                                                     .y = ocr_rect.y + ocr_rect.height,
                                                                     .prob = 1});
                    ocr_info.labels.emplace_back(nodes::LableInfo{.class_id = 0, .prob = 1, .str = ocr_res->ret_});

                    AlgoOutput output;
                    output.vec_ocr_info.emplace_back(ocr_info);
                    output.prob = 1;
                    outputs[image_iter->first].emplace_back(output);
                }
            }
        }
    }

    return impl_->algo_type;
}

void NvInference::init_feature_library(const std::vector<std::string> &paths) {
    if (paths.empty()) { return; }

    std::vector<nodes::FeatureInfo_v1> all_feature_info;

    std::ifstream ifs("/home/data/" + paths[0].substr(0, 53) + "/features.bin", std::ios::binary);
    if (ifs.is_open()) {
        ifs.seekg(0, std::ios::end);
        size_t file_size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        uint32_t verison;
        ifs.read(reinterpret_cast<char *>(&verison), sizeof(verison));

        if (verison == 1) {
            size_t feature_size = (file_size - sizeof(uint32_t)) / sizeof(nodes::FeatureInfo_v1);
            all_feature_info.resize(feature_size);
            ifs.read(reinterpret_cast<char *>(all_feature_info.data()), feature_size * sizeof(nodes::FeatureInfo_v1));
        }

        ifs.close();
    }

    std::vector<nodes::FeatureInfo_v1> selected_feature_info;
    for (const auto &item : all_feature_info) {
        for (const auto &path : paths) {
            // 获取文件路径，不包含文件名
            if (strstr(item.path, path.c_str()) != nullptr) {
                selected_feature_info.emplace_back(item);
                spdlog::info("path: {}, item: {}", path, item.path);
            }
        }
    }

    impl_->feature_mat = cv::Mat(selected_feature_info.size(), kFeatureSize, CV_32FC1);
    int index = 0;
    for (const auto &item : selected_feature_info) {
        memcpy(impl_->feature_mat.data + index * kFeatureSize * sizeof(float), item.feature,
               kFeatureSize * sizeof(float));
        impl_->category_ids.emplace_back(boost::filesystem::path(item.path).filename().replace_extension("").string());
        ++index;
    }
}

std::vector<AlgOutput> NvInference::inference_implement(const std::string &task_name,
                                                        const std::map<int64_t, cv::cuda::GpuMat> &images) {
    std::vector<std::shared_ptr<TransferBuffer>> trans_buf;
    for (const auto &[idx, image] : images) {
        trans_buf.emplace_back(
            std::make_shared<TransferBuffer>(task_name, image.rows, image.cols, image.data, idx, true));
    }

    impl_->alg_impl->PreProcessBatch(trans_buf);
    impl_->alg_impl->ProcessBatch(trans_buf);

    std::vector<AlgOutput> res;
    impl_->alg_impl->PostProcessBatch(trans_buf, res);

    return res;
}

std::vector<AlgOutput>
NvInference::inference_implement(const std::string &task_name, const int width, const int height,
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

    tracker_pose_res.tracker_id_ = 1;// TODO: 需要跟踪 ID
    input_buffers.emplace_back(std::make_shared<TransferBuffer>(task_name, 0, 0, 0, tracker_pose_res));

    while (input_buffers.size() < impl_->batch_size) {
        input_buffers.emplace_back(std::make_shared<TransferBuffer>(std::string("padding"), 0, 0, 0, tracker_pose_res));
    }

    // FILE *file = fopen("keypoints.txt", "a");
    // for (const auto &item : input_key_points) {
    //     for (const auto &point : item) {
    //         char buffer[128];
    //         int len = sprintf(buffer, "x: %.2f, y: %.2f, prob: %.2f\n", point.x, point.y, point.prob);
    //         fwrite(buffer, sizeof(char), len, file);
    //     }
    // }
    // fclose(file);

    impl_->alg_impl->PreProcessBatch(input_buffers);
    impl_->alg_impl->ProcessBatch(input_buffers);

    std::vector<AlgOutput> res;
    impl_->alg_impl->PostProcessBatch(input_buffers, res);

    return res;
}

std::vector<AlgOutput> NvInference::inference_implement(const std::string &task_name, const int width, const int height,
                                                        TrackerPoseRes &tracker_pose_res) {
    std::vector<std::shared_ptr<TransferBuffer>> input_buffers;

    tracker_pose_res.tracker_id_ = 1;// TODO: 需要跟踪 ID
    input_buffers.emplace_back(std::make_shared<TransferBuffer>(task_name, height, width, 1, tracker_pose_res));

    impl_->alg_impl->PreProcessBatch(input_buffers);
    impl_->alg_impl->ProcessBatch(input_buffers);

    std::vector<AlgOutput> res;
    impl_->alg_impl->PostProcessBatch(input_buffers, res);

    return res;
}

void NvInference::convert_algo_output(const DetectRes &res, AlgoOutput &output) {
    output.class_id = res.class_id_;
    output.prob = res.prob_;
    output.box = {res.bbox_[0], res.bbox_[1], res.bbox_[2], res.bbox_[3]};
}

void NvInference::convert_algo_output(const ClassifyRes &res, AlgoOutput &output) {
    output.class_id = res.class_id_;
    output.prob = res.prob_;
}

void NvInference::convert_algo_output(const DetectPoseRes &res, AlgoOutput &output) {
    output.class_id = res.detect_res_.class_id_;
    output.prob = res.detect_res_.prob_;
    output.box = {res.detect_res_.bbox_[0], res.detect_res_.bbox_[1], res.detect_res_.bbox_[2],
                  res.detect_res_.bbox_[3]};
    for (auto &keypoint : res.points_) {
        output.vec_key_points.emplace_back(std::make_tuple(keypoint.number_, keypoint.x_, keypoint.y_, keypoint.prob_));
    }
}

void NvInference::convert_algo_output(const SegmentationRes &res, AlgoOutput &output) {
    output.seg_width = res.width_;
    output.seg_height = res.height_;
    output.seg_map = res.res_;
}

void NvInference::convert_algo_output(const FaceRecognitionRes &res, algo::AlgoOutput &output) {
    cv::Mat feature(1, res.ret_.size(), CV_32FC1, const_cast<float *>(res.ret_.data()));
    if (feature.cols != kFeatureSize) {
        spdlog::error("feature size is not equal to {}", kFeatureSize);
        return;
    }

    int index = -1;
    float max_score = -1;
    if (impl_->feature_mat.rows > 0) {
        cv::Mat result = impl_->feature_mat * feature.t();
        for (int i = 0; i < result.rows; i++) {
            if (result.at<float>(i, 0) > max_score) {
                max_score = result.at<float>(i, 0);
                index = i;
            }
        }
        output.label = impl_->category_ids[index];
    }

    output.class_id = index;
    output.feature = res.ret_;
    output.prob = max_score;
}

}// namespace algo
}// namespace gddi