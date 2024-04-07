#ifndef NV_PREDICTOR_H
#define NV_PREDICTOR_H
#pragma once

#include <iostream>
#include "core/infer_server.h"
#include "core/predictor.h"

namespace gddeploy
{

    class NvPredictorPrivate;

    class NvPredictor : public Predictor
    {
    public:
        // OrtPredictor(){}
        // ~OrtPredictor(){}

        Status Process(PackagePtr pack) noexcept override;

        Status Init(PredictorConfigPtr config, ModelPtr model) noexcept override;

    private:
        ModelPtr model_;
        std::shared_ptr<NvPredictorPrivate> priv_;
    };

    class NvPredictorCreator : public PredictorCreator
    {
    public:
        NvPredictorCreator() : PredictorCreator("nv_creator") {}

        std::shared_ptr<Predictor> Create(PredictorConfigPtr config, ModelPtr model) override
        {
            return std::make_shared<NvPredictor>();
        }

    private:
        std::string postproc_creator_name_ = "NvPredictorCreator";
    };
}

#endif