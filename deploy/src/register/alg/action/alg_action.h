#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgAction : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class ActionAlgCreator : public AlgCreator{
public: 
    ActionAlgCreator():AlgCreator("action_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgAction>() ; 
    }

private: 
    std::string postproc_creator_name_ = "action_creator";
};

}