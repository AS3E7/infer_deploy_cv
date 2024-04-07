#pragma once

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy {

class IntelPredictorPrivate;

class IntelPredictor: public Predictor{
public: 
    IntelPredictor() : Predictor("inte_infer"){}
    // IntelPredictor(){}
    // ~IntelPredictor(){}

    Status Process(PackagePtr pack) noexcept override;

    Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<IntelPredictor>();
      p->CopyParamsFrom(*this);
      if (p->Init(nullptr, model_) != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!IntelPredictor Fork failed\n");
        return nullptr;
      }
      return p;
    }

private: 
    ModelPtr model_;
    std::shared_ptr<IntelPredictorPrivate> priv_;
};


class IntelPredictorCreator : public PredictorCreator{
public: 
    IntelPredictorCreator():PredictorCreator("intel_creator"){}
    
    std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override{ 
        return std::make_shared<IntelPredictor>() ; 
    }

private: 
    std::string postproc_creator_name_ = "IntelPredictorCreator";
};
}