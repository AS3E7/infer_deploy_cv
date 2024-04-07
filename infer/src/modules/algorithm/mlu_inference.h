#ifndef __MLU_DETECTION_H__
#define __MLU_DETECTION_H__

#include <memory>
#include <vector>

#include "abstract_algo.h"
#include "gdd_result_type.h"
#include "nodes/node_msg_def.h"

namespace gddi {
namespace algo {

class MluInference : public AbstractAlgo {
public:
    MluInference();
    ~MluInference();

    bool init(const ModParms &parms) override;
    void inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                   const InferType type) override;
    AlgoType inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                       std::map<int, std::vector<algo::AlgoOutput>> &outputs) override;

protected:
    gdd::InferResult structure_infer_result(const std::map<int, nodes::BoxInfo> &target_box,
                                            const std::vector<std::vector<nodes::PoseKeyPoint>> &key_points);

    gdd::InferResult inference_implement(const std::string &task_name, gdd::InferResult &&infer_result);
    void convert_algo_output(const gdd::DetectObject &obj, algo::AlgoOutput &output);
    void convert_algo_output(const gdd::ClassifyObject &obj, algo::AlgoOutput &output);
    void convert_algo_output(const gdd::DetectPoseObject &obj, algo::AlgoOutput &output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}// namespace algo
}// namespace gddi
#endif// __MLU_DETECTION_H__