/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/
#include "core/infer_server.h"
#include "core/device.h"
#include "common/logger.h"
#include "modcrypt.h"

namespace gddeploy {

DeviceManager *DeviceManager::pInstance_ = nullptr;

std::string Device::GetDeviceUUID(){
    std::string sn = gddeploy::DeviceManager::Instance()->GetDevice()->GetDeviceSN();
#ifdef WITH_NVIDIA
    //兼容之前sdk的序列号
    std::string uuid = gddi::get_device_uuid(sn, "9mflg7sl2pqved84");
#else
    std::string uuid = gddi::get_device_uuid(sn, "inference-engine");
#endif
    return uuid;
}

bool SetCurrentDevice(int device_id) noexcept {
    return true;
}

uint32_t TotalDeviceCount() noexcept {
  return 0;
}

bool CheckDevice(int device_id) noexcept {
  return true;
}
}  // namespace gddeploy

using namespace gddeploy;

#include "cpu/cpu_device.h"

#ifdef WITH_BM1684
#include "bmnn/bmnn_device.h"
#endif

#ifdef WITH_TS
#include "ts/ts_device.h"
#endif

// #ifdef WITH_NVIDIA
// #include "nvidia/nv_device_npp.h"
// #endif

#ifdef WITH_ASCEND
#include "ascend/ascend_device.h"
#endif

#ifdef WITH_RK3588
#include "rk/rk_device.h"
#endif
#ifdef WITH_NVIDIA
#include "nvidia/nv_device.h"
#endif

#ifdef WITH_INTEL
#include "intel/intel_device.h"
#endif

int gddeploy::register_device_module()
{
    DeviceManager* devicemgr = DeviceManager::Instance();

    GDDEPLOY_INFO("[Register] register device module");

#ifdef WITH_BM1684
    GDDEPLOY_INFO("[Register] register device bmnn module");
    BmnnDevice *bmnn_device = new BmnnDevice();
    devicemgr->RegisterDevice("SOPHGO", "SE5", bmnn_device);
#endif

// #ifdef WITH_NVIDIA
//     GDDEPLOY_INFO("[Register] register device nv module");
//     NvDeviceNPPCreator *nv_device = new NvDeviceNPPCreator();
//     devicemgr->RegisterDevice("Nvidia", "any", nv_device);
// #endif

#ifdef WITH_TS
    GDDEPLOY_INFO("[Register] register device ts module");
    TsDevice *ts_device = new TsDevice();
    devicemgr->RegisterDevice("Tsingmicro", "TX5368A", ts_device);
#endif

#ifdef WITH_ASCEND
    GDDEPLOY_INFO("[Register] register device ascend module");
    AscendDevice *ascend_device = new AscendDevice();
    devicemgr->RegisterDevice("HUAWEI", "Ascend310", ascend_device);
#endif

#ifdef WITH_RK3588
    GDDEPLOY_INFO("[Register] register device rk module");
    RkDevice *rk_device = new RkDevice();
    devicemgr->RegisterDevice("Rockchip", "3588", rk_device);
#endif
#ifdef WITH_NVIDIA
    GDDEPLOY_INFO("[Register] register device nvidia module");
    NvDevice *nv_device = new NvDevice();
    devicemgr->RegisterDevice("Nvidia", "any", nv_device);
#endif

#ifdef WITH_INTEL
    GDDEPLOY_INFO("[Register] register device intel module");
    IntelDevice *intel_device = new IntelDevice();
    devicemgr->RegisterDevice("Intel", "any", intel_device);
#endif

    CPUDevice *opencv_device = new CPUDevice();
    devicemgr->RegisterDevice("any", "cpu", opencv_device);

    return 0;
}