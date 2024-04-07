#include "common/json.hpp"
#include "core/preprocess.h"
#include "common/logger.h"
#include <iostream>

using namespace nlohmann;
using namespace gddeploy;

PreProcManager *PreProcManager::pInstance_ = nullptr;
// std::unordered_map<std::string, std::unordered_map<std::string, PreProcCreator*>> PreProcManager::preproc_creator_;
// std::string PreProcManager::name_;
// std::unordered_map<std::string, std::unordered_map<std::string, std::unordered_map<std::string, PreProcOpCreator*>>> PreProcOpManager::preproc_op_creator_;


PreProcConfig::PreProcConfig(std::string config)
{
    auto config_json = json::parse(config);

    // for (auto& x : config_json.items())
    // {
    //     std::cout << "key: " << x.key() << ", value: " << x.value() << '\n';
    //     ops_[x.key()] = x.value();
    // }
}

void PreProcConfig::PrintConfig()
{
    std::cout << "preprocess config: " << std::endl;
    for (auto & iter : ops_)
    {
        std::cout << iter.first << ", " << iter.second << std::endl;
    }
}



Status PreProc::Process(PackagePtr data) noexcept
{
    // for (auto &op : ops_){
    //     op.Process();
    // }

    return Status::SUCCESS;
}

#include "cpu/cpu_preproc.h"

#ifdef WITH_BM1684
#include "bmnn/bmnn_preproc.h"
#endif

#ifdef WITH_NVIDIA
#include "nvidia/nv_preproc_npp.h"
#endif

#ifdef WITH_TS
#include "ts/ts_preproc.h"
#endif

#ifdef WITH_ASCEND
#include "ascend/ascend_preproc.h"
#endif

// #ifdef WITH_RK3588 or WITH_RK3568
// #include "rk/rk_preproc.h"
// #endif

int gddeploy::register_preproc_module()
{
    PreProcManager* preprocmgr = PreProcManager::Instance();

    GDDEPLOY_INFO("[Register] register preproc module");

    CPUPreProcCreator *opencv_preproc = new CPUPreProcCreator();
    preprocmgr->register_preproc("any", "cpu", opencv_preproc);

#ifdef WITH_BM1684
    GDDEPLOY_INFO("[Register] register preproc bmnn module");
    BmnnPreProcCreator *bmnn_preproc = new BmnnPreProcCreator();
    preprocmgr->register_preproc("SOPHGO", "SE5", bmnn_preproc);
#endif

#ifdef WITH_NVIDIA
    GDDEPLOY_INFO("[Register] register preproc nv module");
    NvPreProcNPPCreator *nv_preproc = new NvPreProcNPPCreator();
    preprocmgr->register_preproc("Nvidia", "any", nv_preproc);
#endif

#ifdef WITH_TS
    GDDEPLOY_INFO("[Register] register preproc ts module");
    TsPreProcCreator *ts_preproc = new TsPreProcCreator();
    preprocmgr->register_preproc("Tsingmicro", "TX5368A", ts_preproc);
#endif

#ifdef WITH_ASCEND
    GDDEPLOY_INFO("[Register] register preproc ascend module");
    AscendPreProcCreator *ascend_preproc = new AscendPreProcCreator();
    preprocmgr->register_preproc("HUAWEI", "Ascend310", ascend_preproc);
#endif

// #ifdef WITH_RK3588 or WITH_RK3568
//     GDDEPLOY_INFO("[Register] register preproc rk module");
//     RkPreProcCreator *rk_preproc = new RkPreProcCreator();
//     preprocmgr->register_preproc("Rockchip", "3588", rk_preproc);
// #endif

    return 0;
}