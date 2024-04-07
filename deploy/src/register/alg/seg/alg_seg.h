#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgSeg : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class SegAlgCreator : public AlgCreator{
public: 
    SegAlgCreator():AlgCreator("seg_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgSeg>() ; 
    }

private: 
    std::string postproc_creator_name_ = "seg_creator";
};

}