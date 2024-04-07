#include "modcrypt.h"
#include <string>

#include "core/device.h"
#include "api/global_config.h"

int main(int argc, char *argv[])
{
    std::string path;
    if (argc < 2)
    {
        printf("Usage: %s [gxt_file_path]\n", argv[0]);
        path = "./";
    } else {
        path = std::string(argv[1]);
    }

    gddeploy::gddeploy_init("");
    std::string sn = gddeploy::DeviceManager::Instance()->GetDevice()->GetDeviceSN();
    std::string device_name = gddeploy::DeviceManager::Instance()->GetDevice()->GetDeviceName();

#ifdef WITH_NVIDIA
    //兼容之前sdk的序列号
    std::string uuid = gddi::get_device_uuid(sn, "9mflg7sl2pqved84");
#else
    std::string uuid = gddi::get_device_uuid(sn, "inference-engine");
#endif
    printf("SN:%s, UUID :%s \n", sn.c_str(), uuid.c_str());

    printf("Gtx file will be save in: %s/%s.gxt\n", path.c_str(), uuid.c_str());
    
    gddi::export_gtx_file(uuid, device_name, path);

    return 0;
}