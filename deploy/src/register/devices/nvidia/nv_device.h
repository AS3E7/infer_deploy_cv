#pragma once

#include "core/device.h"

namespace gddeploy {

class NvDevice : public Device {
public:
    NvDevice() noexcept :Device("Nv_device") {
        std::vector<DeviceIp> device_ips;
        DeviceIp pre_ip;
        pre_ip.ip_type = "preproc";
        pre_ip.ip_name = "cuda";
        pre_ip.ip_num = 1;
        device_ips.push_back(pre_ip);

        DeviceIp infer_ip;
        infer_ip.ip_type = "infer";
        infer_ip.ip_name = "cuda";
        infer_ip.ip_num = 1;
        device_ips.push_back(infer_ip);

        DeviceIp post_ip;
        post_ip.ip_type = "postproc";
        post_ip.ip_name = "cpu";
        post_ip.ip_num = 1;
        device_ips.push_back(post_ip);

        SetDeviceIps(device_ips);
    }
    

    std::string GetDeviceSN() override;

private:
};

} // namespace gddeploy