#ifndef __JET_INFERENCE_H__
#define __JET_INFERENCE_H__

#include "abstract_algo.h"
#include "alg.h"
#include "nodes/node_msg_def.h"
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

namespace gddi {
namespace algo {
class JetInference : public AbstractAlgo {
public:
    JetInference();
    ~JetInference();

    bool init(const ModParms &parms) override;

    /**
     * @brief 一阶段推理接口
     * 
     * @param task_name 
     * @param info 
     * @param type 
     */
    void inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                   const InferType type) override;

    /**
     * @brief 多阶段推理接口
     * 
     * @param task_name 
     * @param info 
     * @param outputs 
     * @return AlgoType 
     */
    AlgoType inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                       std::map<int, algo::AlgoOutput> &outputs) override;

protected:
    void init_face_future_dataset(const std::string &path);
    std::vector<AlgOutput> inference_implement(const std::string &task_name, const std::map<int64_t, cv::Mat> &images);
    std::vector<AlgOutput> inference_implement(const std::string &task_name, const int width, const int height,
                                               const std::map<int, nodes::BoxInfo> &target_box,
                                               const std::vector<std::vector<nodes::PoseKeyPoint>> &key_points);
    std::vector<AlgOutput> inference_implement(const std::string &task_name, const int width, const int height,
                                               TrackerPoseRes &tracker_pose_res);
    void convert_algo_output(const DetectRes &res, algo::AlgoOutput &output);
    void convert_algo_output(const ClassifyRes &res, algo::AlgoOutput &output);
    void convert_algo_output(const DetectPoseRes &res, algo::AlgoOutput &output);
    void convert_algo_output(const SegmentationRes &res, algo::AlgoOutput &output);
    void convert_algo_output(const FaceRecognitionRes &res, algo::AlgoOutput &output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}// namespace algo
}// namespace gddi

#endif// __NV_INFERENCE_H__