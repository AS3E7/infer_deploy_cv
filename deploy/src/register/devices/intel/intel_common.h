#pragma once

#include <string>
#include <vector>
#include <memory>
#include "openvino/openvino.hpp"

typedef struct {
    int cpu_request_num;
    int gpu_request_num;
    std::shared_ptr<ov::CompiledModel> compiled_model_cpu_;
    std::shared_ptr<ov::CompiledModel> compiled_model_gpu_;
} MultiCompiledModel;