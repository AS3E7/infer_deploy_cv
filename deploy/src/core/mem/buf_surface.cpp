#include "core/mem/buf_surface.h"
#include <algorithm>
#include <cstring>  // for memset
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <iostream>

#include "core/mem/buf_surface_impl.h"
#include "core/mem/buf_surface_util.h"
#include "common/logger.h"

namespace gddeploy {

class BufSurfaceService {
public:
    static BufSurfaceService &Instance() {
        static std::once_flag s_flag;
        std::call_once(s_flag, [&] { instance_.reset(new BufSurfaceService); });
        return *instance_;
    }
    ~BufSurfaceService() = default;
    int BufPoolCreate(void **pool, BufSurfaceCreateParams *params, uint32_t block_num) {
        if (pool && params && block_num) {
            MemPool *mempool = new MemPool();
            if (!mempool) {
            GDDEPLOY_ERROR("[BufSurfaceService] BufPoolCreate(): new memory pool failed");
                return -1;
            }
            *pool = reinterpret_cast<void *>(mempool);
            if (mempool->Create(params, block_num) == 0) {
                return 0;
            }
            delete mempool;
            GDDEPLOY_ERROR("[BufSurfaceService] BufPoolCreate(): Create memory pool failed");
            return -1;
        }
        return -1;
    }
    int BufPoolDestroy(void *pool) {
        if (pool) {
            MemPool *mempool = reinterpret_cast<MemPool *>(pool);
            if (mempool) {
                int ret = mempool->Destroy();
                if (ret != 0) {
                    GDDEPLOY_ERROR("[BufSurfaceService] BufPoolDestroy(): Destroy memory pool failed, ret = {}", ret);
                    return ret;
                }
                delete mempool;
            }
            return 0;
        }
        GDDEPLOY_ERROR("[BufSurfaceService] BufPoolDestroy(): Pool is not existed");
        return -1;
    }
    int CreateFromPool(BufSurface **surf, void *pool) {
        if (surf && pool) {
            BufSurface surface;
            MemPool *mempool = reinterpret_cast<MemPool *>(pool);
            if (mempool->Alloc(&surface) < 0) {
                GDDEPLOY_ERROR("[BufSurfaceService] CreateFromPool(): Create BufSurface from pool failed");
                return -1;
            }
            *surf = AllocSurface();
            if (!(*surf)) {
                mempool->Free(&surface);
                GDDEPLOY_ERROR("[BufSurfaceService] CreateFromPool(): Alloc BufSurface failed");
                return -1;
            }
            *(*surf) = surface;
            return 0;
        }
        GDDEPLOY_ERROR("[BufSurfaceService] CreateFromPool(): surf or pool is nullptr");
        return -1;
    }
    int Create(BufSurface **surf, BufSurfaceCreateParams *params) {
        if (surf && params) {
            if (CheckParams(params) < 0) {
                GDDEPLOY_ERROR("[BufSurfaceService] Create(): Parameters are invalid");
                return -1;
            }
            BufSurface surface;
            if (CreateSurface(params, &surface) < 0) {
                GDDEPLOY_ERROR("[BufSurfaceService] Create(): Create BufSurface failed");
                return -1;
            }
            *surf = AllocSurface();
            if (!(*surf)) {
                DestroySurface(&surface);
                GDDEPLOY_ERROR("[BufSurfaceService] Create(): Alloc BufSurface failed");
                return -1;
            }
            *(*surf) = surface;
            return 0;
        }
        GDDEPLOY_ERROR("[BufSurfaceService] Create(): surf or params is nullptr");
        return -1;
    }

    int Destroy(BufSurface *surf) {
        if (!surf) {
            GDDEPLOY_ERROR("[BufSurfaceService] Destroy(): surf is nullptr");
            return -1;
        }

        if (surf->opaque) {
            MemPool *mempool = reinterpret_cast<MemPool *>(surf->opaque);
            int ret = mempool->Free(surf);
            FreeSurface(surf);
            if (ret) {
                GDDEPLOY_ERROR("[BufSurfaceService] Destroy(): Free BufSurface back to memory pool failed");
            }
            return ret;
        }

        // created by BufSurfaceCreate()
        int ret = DestroySurface(surf);
        FreeSurface(surf);
        if (ret) {
            GDDEPLOY_ERROR("[BufSurfaceService] Destroy(): Destroy BufSurface failed");
        }
        return ret;
    }

    int SyncForCpu(BufSurface *surf, int index, int plane) {
        // if (surf->mem_type == GDDEPLOY_BUF_MEM_UNIFIED_CACHED || surf->mem_type == GDDEPLOY_BUF_MEM_VB_CACHED) {
        //     if (index == -1) {
        //     for (uint32_t i = 0; i < surf->batch_size; i++) {
        //         cnrtMcacheOperation(surf->surface_list[i].data_ptr, surf->surface_list[i].mapped_data_ptr,
        //                             surf->surface_list[i].data_size, CNRT_INVALID_CACHE);
        //     }
        //     } else {
        //     if (index < 0 || index >= static_cast<int>(surf->batch_size)) {
        //         GDDEPLOY_ERROR("[BufSurfaceService] SyncForCpu(): batch index is invalid");
        //         return -1;
        //     }
        //     cnrtMcacheOperation(surf->surface_list[index].data_ptr, surf->surface_list[index].mapped_data_ptr,
        //                         surf->surface_list[index].data_size, CNRT_INVALID_CACHE);
        //     }
        //     return 0;
        // }
        return -1;
    }

  int SyncForDevice(BufSurface *surf, int index, int plane) {
    // if (surf->mem_type == GDDEPLOY_BUF_MEM_UNIFIED_CACHED || surf->mem_type == GDDEPLOY_BUF_MEM_VB_CACHED) {
    //   if (index == -1) {
    //     for (uint32_t i = 0; i < surf->batch_size; i++) {
    //       cnrtMcacheOperation(surf->surface_list[i].data_ptr, surf->surface_list[i].mapped_data_ptr,
    //                           surf->surface_list[i].data_size, CNRT_FLUSH_CACHE);
    //     }
    //   } else {
    //     if (index < 0 || index >= static_cast<int>(surf->batch_size)) {
    //       GDDEPLOY_ERROR("[BufSurfaceService] SyncForDevice(): batch index is invalid");
    //       return -1;
    //     }
    //     cnrtMcacheOperation(surf->surface_list[index].data_ptr, surf->surface_list[index].mapped_data_ptr,
    //                         surf->surface_list[index].data_size, CNRT_FLUSH_CACHE);
    //   }
    //   return 0;
    // }
    return -1;
  }

  int Memset(BufSurface *surf, int index, int plane, uint8_t value) {
    if (!surf) {
      GDDEPLOY_ERROR("[BufSurfaceService] Memset(): BufSurface is nullptr");
      return -1;
    }
    if (index < -1 || index >= static_cast<int>(surf->batch_size)) {
      GDDEPLOY_ERROR("[BufSurfaceService] Memset(): batch index is invalid");
      return -1;
    }
    if (plane < -1 || plane >= static_cast<int>(surf->surface_list[0].plane_params.num_planes)) {
      GDDEPLOY_ERROR("[BufSurfaceService] Memset(): plane index is invalid");
      return -1;
    }
    if (surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM || surf->mem_type == GDDEPLOY_BUF_MEM_PINNED) {
      for (uint32_t i = 0; i < surf->batch_size; i++) {
        if (index >=0 && i != static_cast<uint32_t>(index)) continue;
        for (uint32_t j = 0; j < surf->surface_list[0].plane_params.num_planes; j++) {
          if (plane >= 0 && j != static_cast<uint32_t>(plane)) continue;
          unsigned char *dst8 = static_cast<unsigned char *>(surf->surface_list[i].data_ptr);
          dst8 += surf->surface_list[i].plane_params.offset[j];
          uint32_t size = surf->surface_list[i].plane_params.psize[j];
          memset(dst8, value, size);
        }
      }
      return 0;
    }
    // device memory
    for (uint32_t i = 0; i < surf->batch_size; i++) {
      if (index >=0 && i != static_cast<uint32_t>(index)) continue;
      for (uint32_t j = 0; j < surf->surface_list[0].plane_params.num_planes; j++) {
        if (plane >= 0 && j != static_cast<uint32_t>(plane)) continue;
        unsigned char *dst8 = static_cast<unsigned char *>(surf->surface_list[i].data_ptr);
        dst8 += surf->surface_list[i].plane_params.offset[j];
        uint32_t size = surf->surface_list[i].plane_params.psize[j];
        // CNRT_SAFECALL(cnrtMemset(dst8, value, size), "[BufSurfaceService] Memset(): failed", -1);
      }
    }
    return 0;
  }

  int Copy(BufSurface *src_surf, BufSurface *dst_surf) {
    IMemAllcator *allocator = CreateMemAllocator(dst_surf->mem_type);
    if (allocator == nullptr){
        GDDEPLOY_ERROR("CreateMemAllocator: Unsupported memory type: {}", dst_surf->mem_type);
        return -1;
    }

    allocator->Copy(src_surf, dst_surf);

    // if (!src_surf || !dst_surf) {
    //   GDDEPLOY_ERROR("[BufSurfaceService] Copy(): src or dst BufSurface is nullptr");
    //   return -1;
    // }
    // // check parameters, must be the same size
    // if (src_surf->batch_size != dst_surf->batch_size) {
    //   GDDEPLOY_ERROR("[BufSurfaceService] Copy(): src and dst BufSurface has different batch size");
    //   return -1;
    // }

    // dst_surf->pts = src_surf->pts;
    // bool src_host = (src_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM || src_surf->mem_type == GDDEPLOY_BUF_MEM_PINNED);

    // bool dst_host = (dst_surf->mem_type == GDDEPLOY_BUF_MEM_SYSTEM || dst_surf->mem_type == GDDEPLOY_BUF_MEM_PINNED);

    // if ((!dst_host && !src_host) && (src_surf->device_id != dst_surf->device_id)) {
    //   GDDEPLOY_ERROR("[BufSurfaceService] Copy(): src and dst BufSurface is on different device");
    //   return -1;
    // }

    // for (size_t i = 0; i < src_surf->batch_size; ++i) {
    //   if (src_surf->surface_list[i].data_size != dst_surf->surface_list[i].data_size) {
    //     uint8_t* src_data_ptr = reinterpret_cast<uint8_t*>(src_surf->surface_list[i].data_ptr);
    //     uint8_t* dst_data_ptr = reinterpret_cast<uint8_t*>(dst_surf->surface_list[i].data_ptr);
    //     uint8_t* src = src_data_ptr;
    //     uint8_t* dst = dst_data_ptr;

    //     for (uint32_t plane_idx = 0 ; plane_idx < src_surf->surface_list[i].plane_params.num_planes; plane_idx++) {
    //       uint32_t src_plane_offset = src_surf->surface_list[i].plane_params.offset[plane_idx];
    //       uint32_t dst_plane_offset = dst_surf->surface_list[i].plane_params.offset[plane_idx];
    //       if (plane_idx && (!src_plane_offset ||!dst_plane_offset)) {
    //         GDDEPLOY_ERROR("[BufSurfaceService] Copy(): src or dst BufSurface plane parameter offset is wrong");
    //         return -1;
    //       }
    //       uint32_t copy_size = src_surf->surface_list[i].plane_params.width[plane_idx] *
    //                            src_surf->surface_list[i].plane_params.bytes_per_pix[plane_idx];
    //       uint32_t src_step = src_surf->surface_list[i].plane_params.pitch[plane_idx];
    //       uint32_t dst_step = dst_surf->surface_list[i].plane_params.pitch[plane_idx];

    //       if (!copy_size || !src_step || !dst_step) {
    //         GDDEPLOY_ERROR("[BufSurfaceService] Copy(): src or dst BufSurface plane parameter width, pitch"
    //                    << " or bytes_per_pix is wrong");
    //         return -1;
    //       }
    //       for (uint32_t h_idx = 0; h_idx < src_surf->surface_list[i].plane_params.height[i]; h_idx++) {
    //         if (dst_host && src_host) {
    //           CNRT_SAFECALL(cnrtMemcpy(dst, src, copy_size, cnrtMemcpyHostToHost),
    //                         "[BufSurfaceService] Copy(): failed", -1);
    //         } else if (dst_host && !src_host) {
    //           CNRT_SAFECALL(cnrtMemcpy(dst, src, copy_size, cnrtMemcpyDevToHost),
    //                         "[BufSurfaceService] Copy(): failed", -1);
    //         } else if (!dst_host && src_host) {
    //           CNRT_SAFECALL(cnrtMemcpy(dst, src, copy_size, cnrtMemcpyHostToDev),
    //                         "[BufSurfaceService] Copy(): failed", -1);
    //         } else {
    //           CNRT_SAFECALL(cnrtMemcpy(dst, src, copy_size, cnrtMemcpyDevToDev),
    //                         "[BufSurfaceService] Copy(): failed", -1);
    //         }
    //         src += src_step;
    //         dst += dst_step;
    //       }
    //       src = src_data_ptr + src_plane_offset;
    //       dst = dst_data_ptr + dst_plane_offset;
    //     }
    //   } else {
    //     if (dst_host && src_host) {
    //       CNRT_SAFECALL(cnrtMemcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr,
    //                                dst_surf->surface_list[i].data_size, cnrtMemcpyHostToHost),
    //                     "[BufSurfaceService] Copy(): failed", -1);
    //     } else if (dst_host && !src_host) {
    //       CNRT_SAFECALL(cnrtMemcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr,
    //                                dst_surf->surface_list[i].data_size, cnrtMemcpyDevToHost),
    //                     "[BufSurfaceService] Copy(): failed", -1);
    //     } else if (!dst_host && src_host) {
    //       CNRT_SAFECALL(cnrtMemcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr,
    //                                dst_surf->surface_list[i].data_size, cnrtMemcpyHostToDev),
    //                     "[BufSurfaceService] Copy(): failed", -1);
    //     } else {
    //       CNRT_SAFECALL(cnrtMemcpy(dst_surf->surface_list[i].data_ptr, src_surf->surface_list[i].data_ptr,
    //                                dst_surf->surface_list[i].data_size, cnrtMemcpyDevToDev),
    //                     "[BufSurfaceService] Copy(): failed", -1);
    //     }
    //   }
    // }

    // //
    // bool dstCached =
    //     (dst_surf->mem_type == GDDEPLOY_BUF_MEM_UNIFIED_CACHED || dst_surf->mem_type == GDDEPLOY_BUF_MEM_VB_CACHED);
    // if (dstCached) {
    //   SyncForCpu(dst_surf, -1, -1);
    // }
    return 0;
  }

private:
  BufSurfaceService(const BufSurfaceService &) = delete;
  BufSurfaceService(BufSurfaceService &&) = delete;
  BufSurfaceService &operator=(const BufSurfaceService &) = delete;
  BufSurfaceService &operator=(BufSurfaceService &&) = delete;
  BufSurfaceService() = default;

private:
  std::mutex mutex_;
  bool initialized_ = false;
  std::queue<BufSurface *> surfpool_;
  BufSurface *start_ = nullptr, *end_ = nullptr;
  static const int k_surfs_num_ = 0;//256 * 1024;

private:
  void CreateSurfsPool() {
    if (initialized_) return;
    start_ = reinterpret_cast<BufSurface *>(malloc(sizeof(BufSurface) * k_surfs_num_));
    if (!start_) {
      GDDEPLOY_ERROR("[BufSurfaceService] CreateSurfsPool(): Create BufSurface pointers failed");
      return;
    }
    end_ = &start_[k_surfs_num_ - 1];
    for (int i = 0; i < k_surfs_num_; i++) surfpool_.push(&start_[i]);
    initialized_ = true;
  }

  BufSurface *AllocSurface() {
    std::unique_lock<std::mutex> lk(mutex_);
    if (!initialized_) CreateSurfsPool();
    if (!surfpool_.empty()) {
      BufSurface *res = surfpool_.front();
      surfpool_.pop();
      return res;
    }
    return reinterpret_cast<BufSurface *>(malloc(sizeof(BufSurface)));
  }

  void FreeSurface(BufSurface *surf) {
    std::unique_lock<std::mutex> lk(mutex_);
    if (surf >= start_ && surf <= end_) {
      surfpool_.push(surf);
      return;
    }
    ::free(surf);
  }

 private:
  static std::unique_ptr<BufSurfaceService> instance_;
};

std::unique_ptr<BufSurfaceService> BufSurfaceService::instance_;

}  

int BufPoolCreate(void **pool, BufSurfaceCreateParams *params, uint32_t block_num) {
  return gddeploy::BufSurfaceService::Instance().BufPoolCreate(pool, params, block_num);
}

int BufPoolDestroy(void *pool) { return gddeploy::BufSurfaceService::Instance().BufPoolDestroy(pool); }

int BufSurfaceCreateFromPool(BufSurface **surf, void *pool) {
  return gddeploy::BufSurfaceService::Instance().CreateFromPool(surf, pool);
}

int BufSurfaceCreate(BufSurface **surf, BufSurfaceCreateParams *params) {
  return gddeploy::BufSurfaceService::Instance().Create(surf, params);
}

int BufSurfaceDestroy(BufSurface *surf) { return gddeploy::BufSurfaceService::Instance().Destroy(surf); }

int BufSurfaceSyncForCpu(BufSurface *surf, int index, int plane) {
  return gddeploy::BufSurfaceService::Instance().SyncForCpu(surf, index, plane);
}

int BufSurfaceSyncForDevice(BufSurface *surf, int index, int plane) {
  return gddeploy::BufSurfaceService::Instance().SyncForDevice(surf, index, plane);
}

int BufSurfaceMemSet(BufSurface *surf, int index, int plane, uint8_t value) {
  return gddeploy::BufSurfaceService::Instance().Memset(surf, index, plane, value);
}

int BufSurfaceCopy(BufSurface *src_surf, BufSurface *dst_surf) {
  return gddeploy::BufSurfaceService::Instance().Copy(src_surf, dst_surf);
}
