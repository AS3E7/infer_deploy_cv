#pragma once

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy {

class RkPredictorPrivate;

class RkPredictor: public Predictor{
public: 
    RkPredictor(int core_id) : Predictor("rk_infer"){
        dev_id_ = core_id;
    }
    // RkPredictor(){}
    // ~RkPredictor(){}

    Status Process(PackagePtr pack) noexcept override;

    Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<RkPredictor>(dev_id_+1);
      p->CopyParamsFrom(*this);
      if (p->Init(nullptr, model_) != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!RkPredictor Fork failed\n");
        return nullptr;
      }
      return p;
    }

private: 
    int dev_id_ = 0;
    ModelPtr model_;
    std::shared_ptr<RkPredictorPrivate> priv_;
};


class RkPredictorCreator : public PredictorCreator{
public: 
    RkPredictorCreator():PredictorCreator("rk_creator"){}
    
    std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override{ 
        return std::make_shared<RkPredictor>(core_num_) ; 
    }

private: 
    int core_num_ = 0;
    std::string postproc_creator_name_ = "RkPredictorCreator";
};
}