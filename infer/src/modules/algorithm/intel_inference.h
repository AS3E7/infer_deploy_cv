#ifndef __INTEL_INFERENCE_H__
#define __INTEL_INFERENCE_H__

#include "abstract_algo.h"
#include "nodes/node_msg_def.h"
#include <memory>

namespace gddi {
namespace algo {
class IntelInference : public AbstractAlgo {
public:
    IntelInference();
    ~IntelInference();

    bool init(const ModParms &parms) override;

    void inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                   const InferType type) override;
    AlgoType inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                       std::map<int, std::vector<algo::AlgoOutput>> &outputs) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}// namespace algo
}// namespace gddi

#endif// __INTEL_INFERENCE_H__