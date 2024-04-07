#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgDetect : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class DetectAlgCreator : public AlgCreator{
public: 
    DetectAlgCreator():AlgCreator("detect_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgDetect>() ; 
    }

private: 
    std::string postproc_creator_name_ = "detect_creator";
};

}