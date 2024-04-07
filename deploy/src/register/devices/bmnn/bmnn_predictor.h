#ifndef ONNXRUNTIME_PREDICTOR_H
#define ONNXRUNTIME_PREDICTOR_H

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy {

class BmnnPredictorPrivate;

class BmnnPredictor: public Predictor{
public: 
    BmnnPredictor() : Predictor("bmnn_infer"){}
    // BmnnPredictor(){}
    // ~BmnnPredictor(){}

    Status Process(PackagePtr pack) noexcept override;

    Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

    std::shared_ptr<Processor> Fork() override
    {
      auto p = std::make_shared<BmnnPredictor>();
      p->CopyParamsFrom(*this);
      if (p->Init(nullptr, model_) != Status::SUCCESS){
        printf("!!!!!!!!!!!!!!!BmnnPredictor Fork failed\n");
        return nullptr;
      }
      return p;
    }

private: 
    ModelPtr model_;
    std::shared_ptr<BmnnPredictorPrivate> priv_;
};


class BmnnPredictorCreator : public PredictorCreator{
public: 
    BmnnPredictorCreator():PredictorCreator("ort_creator"){}
    
    std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override{ 
        return std::make_shared<BmnnPredictor>() ; 
    }

private: 
    std::string postproc_creator_name_ = "BmnnPredictorCreator";
};
}

#endif