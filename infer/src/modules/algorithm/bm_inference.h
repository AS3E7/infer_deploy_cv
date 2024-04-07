#ifndef __BM_INFERENCE_H__
#define __BM_INFERENCE_H__

#include <memory>

#include "abstract_algo.h"
#include "nodes/node_msg_def.h"
#include <core/result_def.h>

namespace gddi {
namespace algo {

class BmInference : public AbstractAlgo {
public:
    BmInference();
    ~BmInference();

    bool init(const ModParms &parms) override;

    void inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                   const InferType type) override;
    AlgoType inference(const std::string &task_name, const std::shared_ptr<nodes::FrameInfo> &info,
                       std::map<int, std::vector<algo::AlgoOutput>> &outputs) override;

protected:
    void parse_infer_result(const gddeploy::PackagePtr &packet, std::vector<algo::AlgoOutput> &outputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}// namespace algo
}// namespace gddi
#endif// __BM_INFERENCE_H__