#ifndef ONNXRUNTIME_PREDICTOR_H
#define ONNXRUNTIME_PREDICTOR_H

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy {

class OrtPredictorPrivate;

class OrtPredictor: public Predictor{
public: 
    // OrtPredictor(){}
    // ~OrtPredictor(){}

    Status Process(PackagePtr pack) noexcept override;

    Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

private: 
    ModelPtr model_;
    std::shared_ptr<OrtPredictorPrivate> priv_;
};


class OrtPredictorCreator : public PredictorCreator{
public: 
    OrtPredictorCreator():PredictorCreator("ort_creator"){}
    
    std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override{ 
        return std::make_shared<OrtPredictor>() ; 
    }

private: 
    std::string postproc_creator_name_ = "OrtPredictorCreator";
};
}

#endif