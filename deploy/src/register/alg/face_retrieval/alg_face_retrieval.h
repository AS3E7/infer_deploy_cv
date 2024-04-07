#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgFaceRetrieval : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class FaceRetrievalAlgCreator : public AlgCreator{
public: 
    FaceRetrievalAlgCreator():AlgCreator("face_retrieval_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgFaceRetrieval>() ; 
    }

private: 
    std::string postproc_creator_name_ = "face_retrieval_creator";
};

}