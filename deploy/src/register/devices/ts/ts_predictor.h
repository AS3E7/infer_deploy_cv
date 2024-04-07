#pragma once

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy {

class TsPredictorPrivate;

class TsPredictor: public Predictor{
public: 
    // TsPredictor(){}
    // ~TsPredictor(){}
    TsPredictor() : Predictor("ts_infer"){}

    Status Process(PackagePtr pack) noexcept override;

    Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<TsPredictor>();
      p->CopyParamsFrom(*this);
      if (p->Init(nullptr, model_) != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!TsPredictor Fork failed\n");
        return nullptr;
      }
      return p;
    }

private: 
    ModelPtr model_;
    std::shared_ptr<TsPredictorPrivate> priv_;
};


class TsPredictorCreator : public PredictorCreator{
public: 
    TsPredictorCreator():PredictorCreator("ts_creator"){}
    
    std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override{ 
        return std::make_shared<TsPredictor>() ; 
    }

private: 
    std::string postproc_creator_name_ = "TsPredictorCreator";
};
}