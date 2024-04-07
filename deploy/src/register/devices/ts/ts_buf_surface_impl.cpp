#include "ts_buf_surface_impl.h"

#include <cstdint>
#include <cstdlib>  // for malloc/free
#include <cstring>  // for memset
#include <string>
#include <iostream>

#include "common/logger.h"
#include "core/mem/buf_surface.h"
#include "core/mem/buf_surface_util.h"

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

using namespace gddeploy;

int MemAllocatorTs::Create(BufSurfaceCreateParams *params) {
    create_params_ = *params;
    uint32_t alignment = 4;
    if (create_params_.batch_size == 0) {
        create_params_.batch_size = 1;
    }
    if (params->force_align_1) {
        alignment = 1;
    }

    memset(&plane_params_, 0, sizeof(BufSurfacePlaneParams));
    block_size_ = params->size;

    GetColorFormatInfo(params->color_format, params->width, params->height, alignment, alignment, &plane_params_);
    if (block_size_) {
        if (create_params_.color_format == GDDEPLOY_BUF_COLOR_FORMAT_INVALID) {
            create_params_.color_format = GDDEPLOY_BUF_COLOR_FORMAT_RGB;
        }
        block_size_ = (block_size_ + alignment - 1) / alignment * alignment;
    } else {
        for (uint32_t i = 0; i < plane_params_.num_planes; i++) {
            block_size_ += plane_params_.psize[i];
        }
    }

    created_ = true;
    return 0;
}

int MemAllocatorTs::Destroy() {
    created_ = false;
    
    return 0;
}

float getBitSizeByFmt(PIXEL_FORMAT_E pixel_fmt) {

	float fVal = 0;

	switch (pixel_fmt)
	{
	case PIXEL_FORMAT_ARGB_8888:
    case PIXEL_FORMAT_ABGR_8888:
		fVal = 4;
		break;
    case PIXEL_FORMAT_BGR_888:
    case PIXEL_FORMAT_RGB_888:
        fVal = 3;
        break;
	case PIXEL_FORMAT_YVU_SEMIPLANAR_420:
	case PIXEL_FORMAT_YUV_SEMIPLANAR_420:
	case PIXEL_FORMAT_NV_12:
	case PIXEL_FORMAT_NV_21:
		fVal = 1.5;
		break;
	default:
		break;
	}

	return fVal;
}

PIXEL_FORMAT_E convertFormat(BufSurfaceColorFormat format)
{
    if (format == GDDEPLOY_BUF_COLOR_FORMAT_NV12){
        return PIXEL_FORMAT_NV_12;
    } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_NV21){
        return PIXEL_FORMAT_NV_21;
    } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_BGR){
        return PIXEL_FORMAT_BGR_888;
    } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_RGB){
        return PIXEL_FORMAT_BGR_888;
    } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_ARGB){
        return PIXEL_FORMAT_ARGB_8888;
    } else if (format == GDDEPLOY_BUF_COLOR_FORMAT_ABGR){
        return PIXEL_FORMAT_ABGR_8888;
    } else {
        return PIXEL_FORMAT_ABGR_8888;
    }
}

// 申请vb内存大小
TS_S32 SAMPLE_VGS_GetFrameVb(VIDEO_FRAME_INFO_S *pstFrameInfo)
{
    TS_U64 u64PhyAddr = 0;
	TS_S32 ret = 0;
    VB_POOL_CONFIG_S stVbPoolCfg;
    TS_U32 vbPool = 0;
    TS_S32 u32HeadStride = ALIGN_UP(pstFrameInfo->stVFrame.u32Width, DEFAULT_ALIGN);
    TS_S32 nlinesize = pstFrameInfo->stVFrame.u32Width * pstFrameInfo->stVFrame.u32Height;
    TS_S32 nsize = nlinesize * getBitSizeByFmt(pstFrameInfo->stVFrame.enPixelFormat);
    // printf("###to call TS_MPI_VB_CreatePool start: u32VBSize=%d\n", nsize);
    stVbPoolCfg.u64BlkSize = nsize;
    stVbPoolCfg.u32BlkCnt = 1;
    stVbPoolCfg.enRemapMode = VB_REMAP_MODE_CACHED;
    pstFrameInfo->u32PoolId = TS_MPI_VB_CreatePool(&stVbPoolCfg);
    vbPool = pstFrameInfo->u32PoolId;
    // printf("!!!!!!!!!!!!!!!!!!!!!!##########################pool id:%d, h:%d, w:%d\n", pstFrameInfo->u32PoolId, pstFrameInfo->stVFrame.u32Width, pstFrameInfo->stVFrame.u32Height);

    // printf("###to call TS_MPI_VB_GetBlock start: vbPool=%d\n", vbPool);
    //VB_INVALID_POOLID
    VB_BLK VbHandle = TS_MPI_VB_GetBlock(vbPool, nsize, TS_NULL);
    if (VB_INVALID_HANDLE == VbHandle) {
        GDDEPLOY_ERROR("TS_MPI_VB_GetBlock failed!\n");
        return TS_FAILURE;
    }

    // printf("###to call TS_MPI_VB_Handle2PhysAddr start VbHandle:%d\n", VbHandle);
    u64PhyAddr = TS_MPI_VB_Handle2PhysAddr(VbHandle);
    if (0 == u64PhyAddr) {
        GDDEPLOY_ERROR("TS_MPI_VB_Handle2PhysAddr failed!.\n");
        TS_MPI_VB_ReleaseBlock(VbHandle);
        TS_MPI_VB_DestroyPool(vbPool);
        GDDEPLOY_ERROR("TS_MPI_VB_DestroyPool\n");
        return TS_FAILURE;
    }

    // printf("###to call TS_MPI_SYS_Mmap start u64PhyAddr=%ld\n", u64PhyAddr);
    // printf("###to call TS_MPI_SYS_Mmap start u32VBSize=%d\n", nsize);
    ret = TS_MPI_VB_MmapPool(vbPool);
    // printf("###call TS_MPI_VB_MmapPool start: ret=%d\n", ret);
    
    ret = TS_MPI_VB_GetBlockVirAddr(vbPool, u64PhyAddr, (void**)&(pstFrameInfo->stVFrame.u64VirAddr[0]));
    // printf("###call TS_MPI_VB_GetBlockVirAddr start: ret=%d\n", ret);
    // printf("###call TS_MPI_VB_GetBlockVirAddr: pstVgsVbInfo->pu8VirAddr=0x%ld\n", pstFrameInfo->stVFrame.u64VirAddr[0]);

    if (TS_NULL == pstFrameInfo->stVFrame.u64VirAddr[0]) {
        GDDEPLOY_ERROR("TS_MPI_VB_GetBlockVirAddr failed!.\n");
        return TS_FAILURE;
    }

    pstFrameInfo->enModId = TS_ID_VGS;
    pstFrameInfo->u32PoolId = vbPool;//TS_MPI_VB_Handle2PoolId(pstVgsVbInfo->VbHandle);
    pstFrameInfo->stVFrame.enField        = VIDEO_FIELD_FRAME;
    pstFrameInfo->stVFrame.enVideoFormat  = VIDEO_FORMAT_LINEAR;
    pstFrameInfo->stVFrame.enCompressMode = COMPRESS_MODE_NONE;
    pstFrameInfo->stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;
    pstFrameInfo->stVFrame.enColorGamut   = COLOR_GAMUT_BT601;

    pstFrameInfo->stVFrame.u32HeaderStride[0]  = u32HeadStride;
    pstFrameInfo->stVFrame.u32HeaderStride[1]  = u32HeadStride;
    pstFrameInfo->stVFrame.u32HeaderStride[2]  = u32HeadStride;
    pstFrameInfo->stVFrame.u64HeaderPhyAddr[0] = u64PhyAddr;
    pstFrameInfo->stVFrame.u64HeaderPhyAddr[1] = pstFrameInfo->stVFrame.u64HeaderPhyAddr[0] + nlinesize;
    pstFrameInfo->stVFrame.u64HeaderPhyAddr[2] = pstFrameInfo->stVFrame.u64HeaderPhyAddr[1];
    pstFrameInfo->stVFrame.u64HeaderVirAddr[0] = pstFrameInfo->stVFrame.u64VirAddr[0];
    pstFrameInfo->stVFrame.u64HeaderVirAddr[1] = pstFrameInfo->stVFrame.u64HeaderVirAddr[0] + nlinesize;
    pstFrameInfo->stVFrame.u64HeaderVirAddr[2] = pstFrameInfo->stVFrame.u64HeaderVirAddr[1];

    pstFrameInfo->stVFrame.u32Stride[0]  = u32HeadStride;
    pstFrameInfo->stVFrame.u32Stride[1]  = u32HeadStride;
    pstFrameInfo->stVFrame.u32Stride[2]  = u32HeadStride;
    pstFrameInfo->stVFrame.u64PhyAddr[0] = pstFrameInfo->stVFrame.u64HeaderPhyAddr[0];
    pstFrameInfo->stVFrame.u64PhyAddr[1] = pstFrameInfo->stVFrame.u64PhyAddr[0] + nlinesize;
    pstFrameInfo->stVFrame.u64PhyAddr[2] = pstFrameInfo->stVFrame.u64PhyAddr[1];
    pstFrameInfo->stVFrame.u64VirAddr[0] = pstFrameInfo->stVFrame.u64HeaderVirAddr[0] ;
    pstFrameInfo->stVFrame.u64VirAddr[1] = pstFrameInfo->stVFrame.u64VirAddr[0] + nlinesize;
    pstFrameInfo->stVFrame.u64VirAddr[2] = pstFrameInfo->stVFrame.u64VirAddr[1];
    pstFrameInfo->stVFrame.u64VirAddr[2] = pstFrameInfo->stVFrame.u64VirAddr[1];
    pstFrameInfo->stVFrame.s32Size = nsize;	
	
    return TS_SUCCESS;
}

TS_S32 SAMPLE_VGS_ReleaseFrameVb(VIDEO_FRAME_INFO_S *pstFrameInfo)
{
	TS_S32 ret = TS_FAILURE;

	if (pstFrameInfo->u32PoolId != 0 ) {
        // printf("!!!!!!!!!!!!!!!!!!!!!!##########################release pool id:%d\n", pstFrameInfo->u32PoolId);
        ret = TS_MPI_VB_MunmapPool(pstFrameInfo->u32PoolId);
        // GDDEPLOY_INFO("###TS_MPI_VB_MunmapPool start: ret=%d\n", ret);

        ret = TS_MPI_VB_DestroyPool(pstFrameInfo->u32PoolId);
        GDDEPLOY_INFO("###TS_MPI_VB_DestroyPool start: ret=%d\n", ret);
        pstFrameInfo->u32PoolId = 0;
	}

	return ret;
}

int MemAllocatorTs::Alloc(BufSurface *surf) {
    memset(surf, 0, sizeof(BufSurface));
    surf->mem_type = create_params_.mem_type;
    surf->opaque = nullptr;  // will be filled by MemPool
    surf->batch_size = create_params_.batch_size;
    surf->device_id = create_params_.device_id;
    surf->is_contiguous = 1;
    surf->surface_list = reinterpret_cast<BufSurfaceParams *>(malloc(sizeof(BufSurfaceParams) * surf->batch_size));
    memset(surf->surface_list, 0, sizeof(BufSurfaceParams) * surf->batch_size);

    VIDEO_FRAME_INFO_S *pstFrameInfo = new VIDEO_FRAME_INFO_S;
    pstFrameInfo->stVFrame.u32Width = create_params_.width;
    pstFrameInfo->stVFrame.u32Height = ALIGN_UP(create_params_.height, 32); //1088
    pstFrameInfo->stVFrame.enPixelFormat = convertFormat(create_params_.color_format);
    if (TS_FAILURE == SAMPLE_VGS_GetFrameVb(pstFrameInfo)){
        GDDEPLOY_ERROR("###SAMPLE_VGS_GetFrameVb fail\n");
    }

    for (int i = 0; i < surf->batch_size; i++) {
        surf->surface_list[i].color_format = create_params_.color_format;
        surf->surface_list[i].data_ptr = (void *)(pstFrameInfo);
        surf->surface_list[i].width = create_params_.width;
        surf->surface_list[i].height = create_params_.height;
        surf->surface_list[i].pitch = plane_params_.pitch[0];
        surf->surface_list[i].data_size = block_size_;
        // surf->surface_list[i].plane_params = plane_params;
    }
    return 0;
}

int MemAllocatorTs::Free(BufSurface *surf) {
    void *addr = surf->surface_list[0].data_ptr;
    SAMPLE_VGS_ReleaseFrameVb((VIDEO_FRAME_INFO_S *)addr);

    delete addr;
    addr = nullptr;
    ::free(reinterpret_cast<void *>(surf->surface_list));
    surf->surface_list = nullptr;
    return 0;
}

int MemAllocatorTs::Copy(BufSurface *src_surf, BufSurface *dst_surf)
{
    for (uint32_t i = 0; i < src_surf->batch_size; i++)
    {
        memcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_size);
    }
    return 0;
}

int MemAllocatorTs::Memset(BufSurface *surf, int value)
{
    for (uint32_t i = 0; i < surf->batch_size; i++)
    {
        memset(surf->surface_list[i].data_ptr, value, surf->surface_list[i].data_size);
    }
    return 0;
}