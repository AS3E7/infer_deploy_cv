#pragma once
#if 1

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy {

class Cambricon2xxPredictorPrivate;

class Cambricon2xxPredictor: public Predictor{
public: 
    // Cambricon2xxPredictor(){}
    // ~Cambricon2xxPredictor(){}

    Status Process(PackagePtr pack) noexcept override;

    Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

private: 
    ModelPtr model_;
    std::shared_ptr<Cambricon2xxPredictorPrivate> priv_;
};


class CambriconPredictorCreator : public PredictorCreator{
public: 
    CambriconPredictorCreator():PredictorCreator("cambricon_creator"){}
    
    std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override{ 
        return std::make_shared<Cambricon2xxPredictor>() ; 
    }

private: 
    std::string postproc_creator_name_ = "CambriconPredictorCreator";
};
}

#endif