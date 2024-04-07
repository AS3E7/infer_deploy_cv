#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgClassify : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class ClassifyAlgCreator : public AlgCreator{
public: 
    ClassifyAlgCreator():AlgCreator("classify_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgClassify>() ; 
    }

private: 
    std::string postproc_creator_name_ = "classify_creator";
};

}