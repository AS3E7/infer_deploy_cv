#pragma once

#include "core/device.h"

namespace gddeploy {

class CPUDevice : public Device {
public:
    CPUDevice() noexcept : Device("cpu_device") {}

    std::string GetDeviceSN() override;

private:
};


} // namespace gddeploy