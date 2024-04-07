#include "api/global_config.h"
#include "core/register.h"

#include<iostream>
#include<mutex>
#include<thread>

#include "version.h"

namespace gddeploy{

int gddeploy_init(std::string config)
{
    // static auto once_call = [&]()->int {
    //     return register_all_module();
    // }();

    static std::once_flag init_flag;
    std::call_once(init_flag, [&]()->int {
        return register_all_module();
    });    

    return 0;
}

std::string gddeploy_version()
{
    return PROJECT_VERSION;
}

}