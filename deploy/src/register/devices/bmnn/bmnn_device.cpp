#include "bmnn_device.h"
#include "bmlib_runtime.h"
#include "common/logger.h"

using namespace gddeploy;

std::string BmnnDevice::GetDeviceSN()
{
    // char product_sn[64] = {0};
    // // 新建handle
    // bm_handle_t handle;
    // bm_dev_request(&handle, 0);

    // bm_status_t ret = bm_get_sn(handle, product_sn);
    // if (ret != BM_SUCCESS){
    //     GDDEPLOY_ERROR("Bmnn get product sn fail:{}", ret);
    //     return "";
    // }

    // bm_dev_free(handle);

    // return std::string(product_sn);

    std::string serial_number;
    serial_number.resize(256);
    FILE *dfp = popen("lsblk --nodeps -no serial /dev/mmcblk0 ", "r");
    if (dfp == nullptr) { std::runtime_error("Failed to get serial number"); }
    serial_number.resize(fread((char *)serial_number.data(), 1, serial_number.size(), dfp) - 1);
    serial_number.erase(std::remove_if(serial_number.begin(), serial_number.end(), isspace),
                        serial_number.end());

    if (serial_number.size() <= 3) { std::runtime_error("Failed to get serial number"); }

    // printf("%s\n", serial_number.c_str());

    return serial_number;
}