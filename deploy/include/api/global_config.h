#pragma once

#include <string>

namespace gddeploy
{

int gddeploy_init(std::string config);

// 0: dynamic, 1: static
#define SESSION_DESC_STRATEGY 0
#define SESSION_DESC_BATCH_TIMEOUT 100
#define SESSION_DESC_ENGINE_NUM 3
#define SESSION_DESC_PRIORITY 0
#define SESSION_DESC_SHOW_PERF 0

std::string gddeploy_version();

}