#include "ts_device.h"

#include "common/logger.h"

using namespace gddeploy;

#include "mpi_sys.h"
#include "mpi_vb.h"
#include "mpi_vdec.h"
#include "ts_common.h"
#include "ts_buffer.h"
#include "ts_comm_sys.h"
#include "ts_comm_vb.h"
#include "ts_comm_vdec.h"
#include "ts_defines.h"
#include "mpi_sys.h"
#include "mpi_vb.h"
#include "mpi_vdec.h"
#include "ts_math.h"
#include "semaphore.h"
#include "ts_type.h"
#include "mpi_vgs.h"

static int tsmpp_init(ts_u32 rtsp_chn_num)
{
	VB_CONFIG_S stVbConfig;
	TS_S32 s32Ret = 0;

	SIZE_S stDispSize;
	stDispSize.u32Width  = 1920;
	stDispSize.u32Height = 1080;

    memset(&stVbConfig, 0, sizeof(VB_CONFIG_S));

	stVbConfig.u32MaxPoolCnt = 1;
	stVbConfig.astCommPool[0].u32BlkCnt  = 2*rtsp_chn_num;
	stVbConfig.astCommPool[0].u64BlkSize = 65536;//COMMON_GetPicBufferSize(stDispSize.u32Width, stDispSize.u32Height,
											//		PIXEL_FORMAT_YVU_SEMIPLANAR_420, DATA_BITWIDTH_8, COMPRESS_MODE_SEG, 0);

	// stVbConfig.astCommPool[1].u32BlkCnt  = 50*rtsp_chn_num;
	// stVbConfig.astCommPool[1].u64BlkSize = COMMON_GetPicBufferSize(stRTSP_URL_MutiCH.stAlgoParam.s32Width, stRTSP_URL_MutiCH.stAlgoParam.s32Height,
	// stVbConfig.astCommPool[1].u64BlkSize = 	(640*640) <<2;
			
	TS_MPI_SYS_Exit();
	TS_MPI_VB_Exit();

	VB_CONFIG_S *pstVbConfig = &stVbConfig;
	if (NULL == pstVbConfig) {
		printf("input parameter is null, it is invaild!\n");
		return -1;
	}

	s32Ret = TS_MPI_VB_SetConfig(pstVbConfig);
	if (0 != s32Ret) {
		printf("TS_MPI_VB_SetConf failed!\n");
		return -1;
	}

	s32Ret = TS_MPI_VB_Init();

	if (0 != s32Ret) {
		printf("TS_MPI_VB_Init failed!\n");
		return -1;
	}

	s32Ret = TS_MPI_SYS_Init();
	if (0 != s32Ret) {
		printf("TS_MPI_SYS_Init failed!\n");
		TS_MPI_VB_Exit();
		return -1;
	}

    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!##################\n");

	return 0;
}
namespace gddeploy {
TsDevice::TsDevice() : Device("Ts_device") {
	
	std::vector<DeviceIp> device_ips;
	DeviceIp pre_ip;
	pre_ip.ip_type = "preproc";
	pre_ip.ip_name = "gpu";
	pre_ip.ip_num = 1;
	device_ips.push_back(pre_ip);
	
	DeviceIp infer_ip;
	infer_ip.ip_type = "infer";
	infer_ip.ip_name = "rne";
	infer_ip.ip_num = 1;
	device_ips.push_back(infer_ip);
	
	DeviceIp post_ip;
	post_ip.ip_type = "postproc";
	post_ip.ip_name = "cpu";
	post_ip.ip_num = 1;
	device_ips.push_back(post_ip);

	SetDeviceIps(device_ips);
    static std::once_flag init_flag;
	
    std::call_once(init_flag, [&]()->int {
        return tsmpp_init(1);
    });    
}

std::string TsDevice::GetDeviceSN()
{
    char product_sn[64] = {0};
    return std::string("123456");
}
}