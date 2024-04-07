#include "core/predictor.h"
#include "common/logger.h"
#include <sstream>

using namespace gddeploy;

// std::unordered_map<std::string, std::shared_ptr<Predictor>> PredictorManager::predictor_cache_;
// std::mutex PredictorManager::predictor_cache_mutex_;
// std::unordered_map<std::string, std::unordered_map<std::string, PredictorCreator*>> PredictorManager::predictor_creator_;
PredictorManager *PredictorManager::pInstance_ = nullptr;

PredictorConfig::PredictorConfig(std::string config)
{
}

void PredictorConfig::PrintConfig()
{
}

static inline std::string GetModelKey(const std::string &config_ptr, const std::string &model_key = "") noexcept
{
    std::ostringstream ss;
    ss << config_ptr << "_" << model_key;
    return ss.str();
}

// std::shared_ptr<Predictor> PredictorManager::GetPredictor(PredictorConfigPtr config, ModelPtr model) noexcept
// {
//     std::unique_lock<std::mutex> lk(predictor_cache_mutex_);

//     std::string model_key = GetModelKey(config->GetConfigStr(), model->GetKey());

//     if (predictor_cache_.find(model_key) == predictor_cache_.cend()) {
//         // Todo: 这里补充如果找不到，判断模型类型，可以使用onnx等推理

//         ModelInfoPrivatePtr model_info = model->GetModelInfoPriv();
//         auto product = model_info->GetProductType();
//         auto chip = model_info->GetChipType();

//         PredictorCreator* creator = predictor_creator_[product][chip];
//         std::shared_ptr<Predictor> preditor = creator->Create(config, model);

//         predictor_cache_[model_key] = preditor;
//         return preditor;
//     }
//     return predictor_cache_.at(model_key);
// }

// void PredictorManager::ClearCache() noexcept {
//     std::lock_guard<std::mutex> lk(predictor_cache_mutex_);
//     predictor_cache_.clear();
// }

// int PredictorManager::CacheSize() noexcept { return predictor_cache_.size(); }

#if defined(WITH_ORT)
#include "onnx/ort_predictor.h"
#endif

#ifdef WITH_BM1684
#include "bmnn/bmnn_predictor.h"
#endif

#ifdef WITH_NVIDIA
#include "nvidia/nv_predictor.h"
#endif

#ifdef WITH_TS
#include "ts/ts_predictor.h"
#endif

#ifdef WITH_MLU220
#include "cambricon/cambricon_predictor.h"
#endif

#ifdef WITH_MLU270
#include "cambricon/cambricon_predictor.h"
#endif

#ifdef WITH_INTEL
#include "intel/intel_predictor.h"
#endif

#ifdef WITH_ASCEND
#include "ascend/ascend_predictor.h"
#endif

#ifdef WITH_RK3588 or WITH_RK3568
#include "rk/rk_predictor.h"
#endif

int gddeploy::register_predictor_module()
{
#if defined(WITH_ORT)
    PredictorManager *preprocmgr = PredictorManager::Instance();
    GDDEPLOY_INFO("[Register] register predictor module");

    OrtPredictorCreator *opencv_preproc = new OrtPredictorCreator();
    preprocmgr->register_predictor("ort", "cpu", opencv_preproc);
#endif

#ifdef WITH_BM1684
    GDDEPLOY_INFO("[Register] register preproc bmnn module");
    PredictorManager *preprocmgr = PredictorManager::Instance();
    BmnnPredictorCreator *bmnn_predictor = new BmnnPredictorCreator();
    preprocmgr->register_predictor("SOPHGO", "SE5", bmnn_predictor);
#endif

#ifdef WITH_NVIDIA
    GDDEPLOY_INFO("[Register] register preproc nvidia module");
    PredictorManager *preprocmgr = PredictorManager::Instance();
    NvPredictorCreator *nv_predictor = new NvPredictorCreator();
    preprocmgr->register_predictor("Nvidia", "any", nv_predictor);
#endif

#ifdef WITH_TS
    GDDEPLOY_INFO("[Register] register preproc ts module");
    PredictorManager* preprocmgr = PredictorManager::Instance();
    TsPredictorCreator *ts_predictor = new TsPredictorCreator();
    preprocmgr->register_predictor("Tsingmicro", "TX5368A", ts_predictor);
#endif

#ifdef WITH_MLU220
    GDDEPLOY_INFO("[Register] register preproc cambricon module");
    PredictorManager* preprocmgr = PredictorManager::Instance();
    CambriconPredictorCreator *cambricon_predictor = new CambriconPredictorCreator();
    preprocmgr->register_predictor("Cambricon", "MLU220", cambricon_predictor);
#endif

#ifdef WITH_MLU270
    GDDEPLOY_INFO("[Register] register preproc cambricon module");
    PredictorManager* preprocmgr = PredictorManager::Instance();
    CambriconPredictorCreator *cambricon_predictor = new CambriconPredictorCreator();
    preprocmgr->register_predictor("Cambricon", "MLU270", cambricon_predictor);
#endif

#ifdef WITH_INTEL
    GDDEPLOY_INFO("[Register] register predictor intel module");
    PredictorManager* preprocmgr = PredictorManager::Instance();
    IntelPredictorCreator *intel_predictor = new IntelPredictorCreator();
    preprocmgr->register_predictor("Intel", "any", intel_predictor);
#endif

#ifdef WITH_ASCEND
    GDDEPLOY_INFO("[Register] register predictor ascend module");
    PredictorManager* preprocmgr = PredictorManager::Instance();
    AscendPredictorCreator *ascend_predictor = new AscendPredictorCreator();
    preprocmgr->register_predictor("Ascend", "any", ascend_predictor);
#endif

#ifdef WITH_RK3588
    GDDEPLOY_INFO("[Register] register predictor rk3588 module");
    PredictorManager* preprocmgr = PredictorManager::Instance();
    RkPredictorCreator *rk_predictor = new RkPredictorCreator();
    preprocmgr->register_predictor("Rockchip", "3588", rk_predictor);
#endif

    return 0;
}