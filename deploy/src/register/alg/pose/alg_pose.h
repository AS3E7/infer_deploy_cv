#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgPose : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class PoseAlgCreator : public AlgCreator{
public: 
    PoseAlgCreator():AlgCreator("pose_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgPose>() ; 
    }

private: 
    std::string postproc_creator_name_ = "pose_creator";
};

}