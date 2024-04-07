#pragma once

#include "core/device.h"

namespace gddeploy {

class IntelDevice : public Device {
public:
    IntelDevice() noexcept :Device("intel_device") {
        std::vector<DeviceIp> device_ips;
        DeviceIp pre_ip;
        pre_ip.ip_type = "preproc";
        pre_ip.ip_name = "cpu";
        pre_ip.ip_num = 2;
        device_ips.push_back(pre_ip);

        DeviceIp infer_ip1;
        infer_ip1.ip_type = "infer";
        infer_ip1.ip_name = "gpu";
        infer_ip1.ip_num = 4;
        device_ips.push_back(infer_ip1);

        DeviceIp infer_ip2;
        infer_ip2.ip_type = "infer";
        infer_ip2.ip_name = "cpu";
        infer_ip2.ip_num = 2;
        device_ips.push_back(infer_ip2);

        DeviceIp post_ip;
        post_ip.ip_type = "postproc";
        post_ip.ip_name = "cpu";
        post_ip.ip_num = 3;
        device_ips.push_back(post_ip);

        SetDeviceIps(device_ips);
    }
    

    std::string GetDeviceSN() override;

private:
};

} // namespace gddeploy