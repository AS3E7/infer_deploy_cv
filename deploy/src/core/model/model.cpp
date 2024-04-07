#include "core/model.h"
#include "common/logger.h"

using namespace gddeploy;

ModelManager *ModelManager::pInstance_ = nullptr;

#if defined(WITH_ORT)
#include "onnx/ort_model.h"
#endif

#ifdef WITH_BM1684
#include "bmnn/bmnn_model.h"
#endif


#ifdef WITH_NVIDIA
#include "nvidia/nv_model.h"
#endif

#ifdef WITH_TS
#include "ts/ts_model.h"
#endif

#ifdef WITH_MLU220
#include "cambricon/cambricon_model.h"
#endif

#ifdef WITH_MLU270
#include "cambricon/cambricon_model.h"
#endif

#ifdef WITH_INTEL
#include "intel/intel_model.h"
#endif

#ifdef WITH_ASCEND
#include "ascend/ascend_model.h"
#endif

#ifdef WITH_RK3588
#include "rk/rk_model.h"
#endif

int gddeploy::register_model_module()
{

    ModelManager *modelmgr = ModelManager::Instance();
    GDDEPLOY_INFO("[Register] register model module");

#if defined(WITH_ORT)
    GDDEPLOY_INFO("[Register] register model module");
    OrtModelCreator *ort_model_creator = new OrtModelCreator();
    modelmgr->RegisterModel("ort", "cpu", ort_model_creator);
#endif

#ifdef WITH_BM1684
    GDDEPLOY_INFO("[Register] register bmnn model module");
    BmnnModelCreator *bmnn_model_creator = new BmnnModelCreator();
    modelmgr->RegisterModel("SOPHGO", "SE5", bmnn_model_creator);
#endif

#ifdef WITH_NVIDIA
    GDDEPLOY_INFO("[Register] register nvidia model module");
    NvModelCreator *nvidia_model_creator = new NvModelCreator();
    modelmgr->RegisterModel("NVIDIA", "any", nvidia_model_creator);
#endif

#ifdef WITH_TS
    GDDEPLOY_INFO("[Register] register ts model module");
    TsModelCreator *ts_model_creator = new TsModelCreator();
    modelmgr->RegisterModel("Tsingmicro", "TX5368A", ts_model_creator);
#endif

#ifdef WITH_MLU220
    GDDEPLOY_INFO("[Register] register mlu220 model module");
    CambriconModelCreator *cambricon_model_creator = new CambriconModelCreator();
    modelmgr->RegisterModel("Cambricon", "MLU220", cambricon_model_creator);
#endif

#ifdef WITH_MLU270
    GDDEPLOY_INFO("[Register] register mlu220 model module");
    CambriconModelCreator *cambricon_model_creator = new CambriconModelCreator();
    modelmgr->RegisterModel("Cambricon", "MLU270", cambricon_model_creator);
#endif

#ifdef WITH_INTEL
    GDDEPLOY_INFO("[Register] register intel model module");
    IntelModelCreator *intel_model_creator = new IntelModelCreator();
    modelmgr->RegisterModel("Intel", "any", intel_model_creator);
#endif

#ifdef WITH_ASCEND
    GDDEPLOY_INFO("[Register] register ascend model module");
    AscendModelCreator *ascend_model_creator = new AscendModelCreator();
    modelmgr->RegisterModel("Ascend", "any", ascend_model_creator);
#endif

#ifdef WITH_RK3588
    GDDEPLOY_INFO("[Register] register rk model module");
    RkModelCreator *rk_model_creator = new RkModelCreator();
    modelmgr->RegisterModel("Rockchip", "3588", rk_model_creator);
#endif

    return 0;
}