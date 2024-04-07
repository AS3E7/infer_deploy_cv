#include "nv_device.h"
#include "common/logger.h"
#include <nvml.h>

using namespace gddeploy;

std::string NvDevice::GetDeviceSN()
{
#ifdef AARCH64
    std::ifstream file("/sys/firmware/devicetree/base/serial-number");
    std::string serial_number((std::istream_iterator<char>(file)), std::istream_iterator<char>());
    serial_number.erase(std::remove_if(serial_number.begin(), serial_number.end(),
                                       [](char c)
                                       { return !std::isprint(c); }),
                        serial_number.end());
    // std::cout << "serial-number " << serial_number<<std::endl;
    return serial_number;
#else
    uint32_t device_count = 0;
    nvmlInit_v2();
    nvmlDeviceGetCount_v2(&device_count);

    if (device_count > 0) {
        auto handle = std::make_unique<nvmlDevice_t>();
        if (nvmlDeviceGetHandleByIndex_v2(0, handle.get()) == NVML_SUCCESS) {
            char buffer[NVML_DEVICE_UUID_V2_BUFFER_SIZE];
            if (nvmlDeviceGetUUID(*handle, buffer, NVML_DEVICE_UUID_V2_BUFFER_SIZE)
                == NVML_SUCCESS) {
                // printf("GPU UUID: %s)\n", buffer);
                return std::string(buffer);
            }
        }
    }
#endif
}