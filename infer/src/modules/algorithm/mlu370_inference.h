#ifndef __MLU370_DETECTION_H__
#define __MLU370_DETECTION_H__

#include <memory>

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
                       std::map<int, algo::AlgoOutput> &outputs) override;

protected:
    void convert_algo_output(const int prev_id, const gdd::DetectObject &obj, algo::AlgoOutput &output);
    void convert_algo_output(const int prev_id, const gdd::ClassifyObject &obj, algo::AlgoOutput &output);
    void convert_algo_output(const int prev_id, const gdd::DetectPoseObject &obj, algo::AlgoOutput &output);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}// namespace algo
}// namespace gddi
#endif// __MLU_DETECTION_H__