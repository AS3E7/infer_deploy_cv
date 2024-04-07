#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgOcrRec : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class OcrRecAlgCreator : public AlgCreator{
public: 
    OcrRecAlgCreator():AlgCreator("ocr_retrieval_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgOcrRec>() ; 
    }

private: 
    std::string postproc_creator_name_ = "ocr_retrieval_creator";
};

}