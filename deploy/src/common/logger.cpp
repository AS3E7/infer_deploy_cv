
#include "common/logger.h"

#include <cstdlib>
#include <iostream>

namespace gddeploy{

static void LoadEnvLevels() {
  auto p = std::getenv("SPDLOG_LEVEL");
  if (p) {
    const std::string str(p);
    if (str == "trace") {
      spdlog::set_level(spdlog::level::trace);
    } else if (str == "debug") {
      spdlog::set_level(spdlog::level::debug);
    } else if (str == "info") {
      spdlog::set_level(spdlog::level::info);
    } else if (str == "warn") {
  spdlog::set_level(spdlog::level::warn);
    } else if (str == "err") {
      spdlog::set_level(spdlog::level::err);
    } else if (str == "critical") {
      spdlog::set_level(spdlog::level::critical);
    } else if (str == "off") {
      spdlog::set_level(spdlog::level::off);
    }
  } else {
    spdlog::set_level(spdlog::level::err);
  }
  // spdlog::set_level(spdlog::level::err);
}

std::shared_ptr<spdlog::logger> CreateDefaultLogger() {
  LoadEnvLevels();
  
  constexpr const auto logger_name = "gddeploy";
  return spdlog::stdout_color_mt(logger_name);
}

std::shared_ptr<spdlog::logger> &gLogger() {
  static auto ptr = new std::shared_ptr<spdlog::logger>{CreateDefaultLogger()};
  return *ptr;
}

spdlog::logger *GetLogger() { 
    return gLogger().get(); 
}

}