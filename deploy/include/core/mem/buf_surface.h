#pragma once

#include <stdint.h>
#include <stdbool.h>

/** Defines the default padding length for reserved fields of structures. */
#define GDDEPLOY_PADDING_LENGTH  4

/** Defines the maximum number of planes. */
#define GDDEPLOY_BUF_MAX_PLANES   3

/**
 * Specifies color formats for \ref BufSurface.
 */
typedef enum {
  /** Specifies an invalid color format. */
  GDDEPLOY_BUF_COLOR_FORMAT_INVALID,
  /** Specifies 8 bit GRAY scale - single plane */
  GDDEPLOY_BUF_COLOR_FORMAT_GRAY8,
  /** Specifies BT.601 colorspace - YUV420 multi-planar. */
  GDDEPLOY_BUF_COLOR_FORMAT_YUV420,
  /** Specifies BT.601 colorspace - YUV420 multi-planar. */
  GDDEPLOY_BUF_COLOR_FORMAT_YUVJ420P,
  /** Specifies BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  GDDEPLOY_BUF_COLOR_FORMAT_NV12,
  /** Specifies BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. */
  GDDEPLOY_BUF_COLOR_FORMAT_NV21,
  /** Specifies ARGB-8-8-8-8 single plane. */
  GDDEPLOY_BUF_COLOR_FORMAT_ARGB,
  /** Specifies ABGR-8-8-8-8 single plane. */
  GDDEPLOY_BUF_COLOR_FORMAT_ABGR,
  /** Specifies RGB-8-8-8 single plane. */
  GDDEPLOY_BUF_COLOR_FORMAT_RGB,
  /** Specifies BGR-8-8-8 single plane. */
  GDDEPLOY_BUF_COLOR_FORMAT_BGR,
  /** Specifies RGB-8-8-8 single plane. */
  GDDEPLOY_BUF_COLOR_FORMAT_RGB_PLANNER,
  /** Specifies BGR-8-8-8 single plane. */
  GDDEPLOY_BUF_COLOR_FORMAT_BGR_PLANNER,
  GDDEPLOY_BUF_COLOR_FORMAT_BGRA,
  GDDEPLOY_BUF_COLOR_FORMAT_RGBA,


  /*TODO(gaoyujia): add more color format
  */
  /** Specifies ARGB-1-5-5-5 single plane. */
  GDDEPLOY_BUF_COLOR_FORMAT_ARGB1555,

  /** for inference*/
  GDDEPLOY_BUF_COLOR_FORMAT_TENSOR,
  GDDEPLOY_BUF_COLOR_FORMAT_LAST,
} BufSurfaceColorFormat;

/**
 * Specifies memory types for \ref BufSurface.
 */
typedef enum {
  /** Specifies the default memory type, i.e. \ref GDDEPLOY_BUF_MEM_DEVICE
   for MLUxxx, \ref GDDEPLOY_BUF_MEM_UNIFIED for CExxxx. Use \ref GDDEPLOY_BUF_MEM_DEFAULT
   to allocate whichever type of memory is appropriate for the platform. */
  GDDEPLOY_BUF_MEM_DEFAULT,
  /** Specifies MLU Device memory type. valid only for MLUxxx*/
  GDDEPLOY_BUF_MEM_DEVICE,
  /** Specifies Host memory type. valid only for MLUxxx */
  GDDEPLOY_BUF_MEM_PINNED,
  /** Specifies Unified memory type. Valid only for CExxxx. */
  GDDEPLOY_BUF_MEM_UNIFIED,
  GDDEPLOY_BUF_MEM_UNIFIED_CACHED,
  /** Specifies VB memory type. Valid only for CExxxx. */
  GDDEPLOY_BUF_MEM_BMNN,
  /** Specifies memory allocated by malloc(). */
  GDDEPLOY_BUF_MEM_SYSTEM,
  /** Specifies cambricon mlu memory allocated by malloc(). */
  GDDEPLOY_BUF_MEM_CAMBRICON,
  /** Specifies nvidia cuda memory allocated by malloc(). */
  GDDEPLOY_BUF_MEM_NVIDIA,
  /** Specifies tsingmicro memory allocated by malloc(). */
  GDDEPLOY_BUF_MEM_TS,
  /** Specifies tsingmicro memory allocated by malloc(). */
  GDDEPLOY_BUF_MEM_ASCEND_DVPP,
  GDDEPLOY_BUF_MEM_ASCEND_RT,
  /** Specifies rockchip memory allocated by malloc(). */
  GDDEPLOY_BUF_MEM_RK_RGA,
} BufSurfaceMemType;

/**
 * Holds the planewise parameters of a buffer.
 */
typedef struct BufSurfacePlaneParams {
  /** Holds the number of planes. */
  uint32_t num_planes;
  /** Holds the widths of planes. */
  uint32_t width[GDDEPLOY_BUF_MAX_PLANES];
  /** Holds the heights of planes. */
  uint32_t height[GDDEPLOY_BUF_MAX_PLANES];
  /** Holds the pitches of planes in bytes. */
  uint32_t pitch[GDDEPLOY_BUF_MAX_PLANES];
  /** Holds the offsets of planes in bytes. */
  uint32_t offset[GDDEPLOY_BUF_MAX_PLANES];
  /** Holds the sizes of planes in bytes. */
  uint32_t psize[GDDEPLOY_BUF_MAX_PLANES];
  /** Holds the number of bytes occupied by a pixel in each plane. */
  uint32_t bytes_per_pix[GDDEPLOY_BUF_MAX_PLANES];
  /** Holds the data ptr of plane. */
  void * data_ptr[GDDEPLOY_BUF_MAX_PLANES];

  void * _reserved[GDDEPLOY_PADDING_LENGTH * GDDEPLOY_BUF_MAX_PLANES];
} BufSurfacePlaneParams;

/**
 * Holds parameters required to allocate an \ref BufSurface.
 */
typedef struct BufSurfaceCreateParams {
  /** Holds the type of memory to be allocated. Not valid for GDDEPLOY_BUF_MEM_VB* */
  BufSurfaceMemType mem_type;

  /** Holds the Device ID. */
  uint32_t device_id;
  /** Holds the width of the buffer. */
  uint32_t width;
  /** Holds the height of the buffer. */
  uint32_t height;
  /** Holds the number of bytes occupied by a pixel in each plane. */
  uint32_t bytes_per_pix;
  /** Holds the color format of the buffer. */
  BufSurfaceColorFormat color_format;

  /** Holds the amount of memory to be allocated. Optional; if set, all other
   parameters (width, height, etc.) are ignored. */
  uint32_t size;

  /** Holds the batch size. */
  uint32_t batch_size;

  /** Holds the alignment mode, if set,  1 bytes alignment will be applied;
   Not valid for GDDEPLOY_BUF_MEM_VB and GDDEPLOY_BUF_MEM_VB_CACHED.
  */
  bool force_align_1;

  void *_reserved[GDDEPLOY_PADDING_LENGTH];
} BufSurfaceCreateParams;

/**
 * Holds information about a single buffer in a batch.
 */
typedef struct BufSurfaceParams {
  /** Holds the width of the buffer. */
  uint32_t width;
  /** Holds the height of the buffer. */
  uint32_t height;
  /** Holds the pitch of the buffer. */
  uint32_t pitch;
  /** Holds the color format of the buffer. */
  BufSurfaceColorFormat color_format;

  /** Holds the amount of allocated memory. */
  uint32_t data_size;

  /** Holds a pointer to allocated memory. */
  void * data_ptr;

  /** Holds a pointer to a CPU mapped buffer.
  Valid only for GDDEPLOY_BUF_MEM_UNIFIED* and GDDEPLOY_BUF_MEM_VB* */
  void * mapped_data_ptr;

  /** Holds planewise information (width, height, pitch, offset, etc.). */
  BufSurfacePlaneParams plane_params;

  // index of package
  int index_of_package;

  void * _reserved[GDDEPLOY_PADDING_LENGTH];
} BufSurfaceParams;

/**
 * Holds information about batched buffers.
 */
typedef struct BufSurface {
  /** Holds type of memory for buffers in the batch. */
  BufSurfaceMemType mem_type;

  /** Holds a Device ID. */
  uint32_t device_id;
  /** Holds the batch size. */
  uint32_t batch_size;
  /** Holds the number valid and filled buffers. Initialized to zero when
   an instance of the structure is created. */
  uint32_t num_filled;

  /** Holds an "is contiguous" flag. If set, memory allocated for the batch
  is contiguous. Not valid for GDDEPLOY_BUF_MEM_VB on CE3226 */
  bool is_contiguous;

  /** Holds a pointer to an array of batched buffers. */
  BufSurfaceParams *surface_list;

  /** Holds a pointer to the buffer pool context */
  void *opaque;

  /** Holds the timestamp for video image, valid only for batch_size == 1 */
  uint64_t pts;

  void * _reserved[GDDEPLOY_PADDING_LENGTH];
} BufSurface;

/**
 * @brief  Creates a Buffer Pool.
 *
 * Call BufPoolDestroy() to free resources allocated by this function.
 *
 * @param[out] pool         An indirect pointer to the buffer pool.
 * @param[in]  params       A pointer to an \ref BufSurfaceCreateParams
 *                           structure.
 * @param[in]  block_num    The block number.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufPoolCreate(void **pool, BufSurfaceCreateParams *params, uint32_t block_num);

/**
 * @brief  Frees the buffer pool previously allocated by BufPoolCreate().
 *
 * @param[in] surf  A pointer to an \ref buffer pool to be freed.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufPoolDestroy(void *pool);

/**
 * @brief  Allocates a single buffer.
 *
 * Allocates memory for a buffer and returns a pointer to an
 * allocated \ref BufSurface. The \a params structure must have
 * the allocation parameters of a single buffer.
 *
 * Call BufSurfaceDestroy() to free resources allocated by this function.
 *
 * @param[out] surf         An indirect pointer to the allocated buffer.
 * @param[in]  pool         A pointer to a buffer pool.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufSurfaceCreateFromPool(BufSurface **surf, void *pool);

/**
 * @brief  Allocates a batch of buffers.
 *
 * Allocates memory for \a batch_size buffers and returns a pointer to an
 * allocated \ref BufSurface. The \a params structure must have
 * the allocation parameters. If \a params.size
 * is set, a buffer of that size is allocated, and all other
 * parameters (width, height, color format, etc.) are ignored.
 *
 * Call BufSurfaceDestroy() to free resources allocated by this function.
 *
 * @param[out] surf         An indirect pointer to the allocated batched
 *                           buffers.
 * @param[in]  params       A pointer to an \ref BufSurfaceCreateParams
 *                           structure.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufSurfaceCreate(BufSurface **surf, BufSurfaceCreateParams *params);

/**
 * @brief  Frees a single buffer allocated by BufSurfaceCreate()
 *         or batched buffers previously allocated by BufSurfaceCreate().
 *
 * @param[in] surf  A pointer to an \ref BufSurface to be freed.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufSurfaceDestroy(BufSurface *surf);

/**
 * @brief  Syncs the hardware memory cache for the CPU.
 *
 * Valid only for memory types \ref GDDEPLOY_BUF_MEM_UNIFIED_CACHED and
 * \ref GDDEPLOY_BUF_MEM_VB_CACHED.
 *
 * @param[in] surf      A pointer to an \ref NvBufSurface structure.
 * @param[in] index     Index of the buffer in the batch. -1 refers to
 *                      all buffers in the batch.
 * @param[in] plane     Index of a plane in the buffer. -1 refers to all planes
 *                      in the buffer.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufSurfaceSyncForCpu(BufSurface *surf, int index, int plane);

/**
 * @brief  Syncs the hardware memory cache for the device.
 *
 * Valid only for memory types \ref GDDEPLOY_BUF_MEM_UNIFIED and
 * \ref GDDEPLOY_BUF_MEM_VB.
 *
 * @param[in] surf      A pointer to an \ref BufSurface structure.
 * @param[in] index     Index of a buffer in the batch. -1 refers to all buffers
 *                      in the batch.
 * @param[in] plane     Index of a plane in the buffer. -1 refers to all planes
 *                      in the buffer.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufSurfaceSyncForDevice(BufSurface *surf, int index, int plane);

/**
 * @brief  Copies the content of source batched buffer(s) to destination
 * batched buffer(s).
 *
 * The source and destination \ref BufSurface objects must have same
 * buffer and batch size.
 *
 * @param[in] src_surf   A pointer to the source BufSurface structure.
 * @param[out] dst_surf   A pointer to the destination BufSurface structure.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufSurfaceCopy(BufSurface *src_surf, BufSurface *dst_surf);

/**
 * @brief  Fills each byte of the buffer(s) in an \ref BufSurface with a
 * provided value.
 *
 * You can also use this function to reset the buffer(s) in the batch.
 *
 * @param[in] surf  A pointer to the BufSurface structure.
 * @param[in] index Index of a buffer in the batch. -1 refers to all buffers
 *                  in the batch.
 * @param[in] plane Index of a plane in the buffer. -1 refers to all planes
 *                  in the buffer.
 * @param[in] value The value to be used as fill.
 *
 * @return Returns 0 if this function has run successfully. Otherwise returns -1.
 */
int BufSurfaceMemSet(BufSurface *surf, int index, int plane, uint8_t value);
