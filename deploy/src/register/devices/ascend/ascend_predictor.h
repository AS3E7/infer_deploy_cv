#pragma once

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy {

class AscendPredictorPrivate;

class AscendPredictor: public Predictor{
public: 
    AscendPredictor() : Predictor("ascend_infer"){}
    // AscendPredictor(){}
    // ~AscendPredictor(){}

    Status Process(PackagePtr pack) noexcept override;

    Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

        std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<AscendPredictor>();
      p->CopyParamsFrom(*this);
      if (p->Init(nullptr, model_) != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!AscendPredictor Fork failed\n");
        return nullptr;
      }
      return p;
    }

private: 
    ModelPtr model_;
    std::shared_ptr<AscendPredictorPrivate> priv_;
};


class AscendPredictorCreator : public PredictorCreator{
public: 
    AscendPredictorCreator():PredictorCreator("ascend_creator"){}
    
    std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override{ 
        return std::make_shared<AscendPredictor>() ; 
    }

private: 
    std::string postproc_creator_name_ = "AscendPredictorCreator";
};
} // namespace gddeploy