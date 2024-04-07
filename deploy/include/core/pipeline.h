#pragma once

#include <string>
#include <vector>

#include "core/alg.h"

namespace gddeploy
{
/*
 *
 */

class Pipeline{
public:

    static Pipeline* Instance() noexcept{
        if (pInstance_ == nullptr){
            pInstance_ = new Pipeline();
        }
        return pInstance_;
    }

    // 根据算法和config创建pipeline
    int CreatePipeline(std::string config, ModelPtr model, std::vector<ProcessorPtr> &processors);

    // 给pipeline添加新的processor
    int AddProcessor(std::vector<Processor> &base, std::vector<ProcessorPtr> &new_processors);

private:
    static Pipeline *pInstance_;

};


}