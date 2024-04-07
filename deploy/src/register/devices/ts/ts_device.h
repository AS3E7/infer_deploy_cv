#pragma once

#include "core/device.h"

namespace gddeploy {

class TsDevice : public Device {
public:
    TsDevice();

    std::string GetDeviceSN() override;

private:
};

} // namespace gddeploy