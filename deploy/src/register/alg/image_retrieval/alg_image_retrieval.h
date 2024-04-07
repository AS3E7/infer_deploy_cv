#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgImageRetrieval : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class ImageRetrievalAlgCreator : public AlgCreator{
public: 
    ImageRetrievalAlgCreator():AlgCreator("image_retrieval_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgImageRetrieval>() ; 
    }

private: 
    std::string postproc_creator_name_ = "image_retrieval_creator";
};

}