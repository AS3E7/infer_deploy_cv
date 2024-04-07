#include "common/json.hpp"
#include "core/postprocess.h"
#include "common/logger.h"
#include <iostream>

using namespace nlohmann;
using namespace gddeploy;

// std::unordered_map<std::string, std::unordered_map<std::string, PostProcCreator*>> PostProcManager::postproc_creator_;
PostProcManager *PostProcManager::pInstance_ = nullptr;

PostProcConfig::PostProcConfig(std::string config)
{
    auto config_json = json::parse(config);

    for (auto& x : config_json.items())
    {
        std::cout << "key: " << x.key() << ", value: " << x.value() << '\n';
        ops_[x.key()] = x.value();
    }
}

void PostProcConfig::PrintConfig()
{
    std::cout << "postproc config: " << std::endl;
    for (auto & iter : ops_)
    {
        std::cout << iter.first << ", " << iter.second << std::endl;
    }
}


#include "cpu/cpu_postproc.h"

int gddeploy::register_postproc_module()
{
    PostProcManager* postprocmgr = PostProcManager::Instance();
    GDDEPLOY_INFO("[Register] register postproc module");

    CpuPostprocCreator *cpu_postproc = new CpuPostprocCreator();
    postprocmgr->register_postproc("any", "cpu", cpu_postproc);

    return 0;
}
