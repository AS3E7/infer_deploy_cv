#pragma once

#include "core/alg.h"

namespace gddeploy {
class AlgOcrDetect : public Alg{
public:
    int Init(std::string config, ModelPtr model) override; 
    int CreateProcessor(std::vector<ProcessorPtr> &processors) override;  

private:
    ModelPtr model_;
};

class OcrDetectAlgCreator : public AlgCreator{
public: 
    OcrDetectAlgCreator():AlgCreator("ocr_det_creator"){}
    
    AlgPtr Create() override{
        return std::make_shared<AlgOcrDetect>() ; 
    }

private: 
    std::string postproc_creator_name_ = "ocr_det_creator";
};

}