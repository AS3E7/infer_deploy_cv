#include "alg_seg.h"
#include "core/preprocess.h"
#include "core/predictor.h"
#include "core/postprocess.h"
#include <memory>
#include <utility>
#include <vector>

using namespace gddeploy;

int AlgSeg::Init(std::string config, ModelPtr model)
{
    model_ = model;
    return 0;
}

int AlgSeg::CreateProcessor(std::vector<ProcessorPtr> &processors)
{
    // std::vector<ProcessorPtr> processors;
    auto model_info_priv = model_->GetModelInfoPriv();
    auto manu = model_info_priv->GetProductType();
    auto chip_type = model_info_priv->GetChipType();

    // 1. 创建前处理模块
    // 先找能否找到硬件处理单元，如果找不到就改为CPU前处理
    PreProcManager* preprocmgr = PreProcManager::Instance();
    PreProcCreator* preproc_creator = preprocmgr->GetPreProcCreator(manu, chip_type);
    if (preproc_creator == nullptr){
        preproc_creator = preprocmgr->GetPreProcCreator("any", "cpu");
    }

    // if (preproc_creator != nullptr){
    PreProcPtr preproc =  preproc_creator->Create();
    preproc->Init(model_, "");

    processors.emplace_back(std::move(preproc));

    // 2. 推理
    PredictorManager* predictormgr = PredictorManager::Instance();
    auto predictor_creator = predictormgr->GetPredictorCreator(manu, chip_type);
    if (predictor_creator == nullptr){
        predictor_creator = predictormgr->GetPredictorCreator("ort", "cpu");
    }
    auto predictor = predictor_creator->Create(nullptr, model_);
    predictor->Init(nullptr, model_);

    processors.emplace_back(predictor);

    // 3. 后处理
    PostProcManager* prepostmgr = PostProcManager::Instance();
    auto postproc_creator = prepostmgr->GetPostProcCreator(manu, chip_type);
    if (postproc_creator == nullptr){
        postproc_creator = prepostmgr->GetPostProcCreator("any", "cpu");
    }
    auto postproc = postproc_creator->Create();
    postproc->Init(model_, "");

    ProcessorPtr test = postproc;

    processors.emplace_back(postproc);

    return 0;
}