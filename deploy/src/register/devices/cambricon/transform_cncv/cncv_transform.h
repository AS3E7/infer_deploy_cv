#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "core/mem/buf_surface.h"
#include "cnrt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GDDEPLOY_TRANSFORM_MAX_CHNS   4
/**
 * Specifies compute devices used by Transform.
 */
typedef enum {
  /** Specifies VGU as a compute device for CExxxx or MLU for MLUxxx. */
  GDDEPLOY_TRANSFORM_COMPUTE_DEFAULT,
  /** Specifies that the MLU is the compute device. */
  GDDEPLOY_TRANSFORM_COMPUTE_MLU,
  /** Specifies that the VGU as a compute device. Only supported on CExxxx. */
  GDDEPLOY_TRANSFORM_COMPUTE_VGU,
  /** Specifies the number of compute modes. */
  GDDEPLOY_TRANSFORM_COMPUTE_NUM
} TransformComputeMode;

/**
 * Specifies transform types.
 */
typedef enum {
  /** Specifies a transform to crop the source rectangle. */
  GDDEPLOY_TRANSFORM_CROP_SRC   = 1,
  /** Specifies a transform to crop the destination rectangle. */
  GDDEPLOY_TRANSFORM_CROP_DST   = 1 << 1,
  /** Specifies a transform to set the filter type. */
  GDDEPLOY_TRANSFORM_FILTER     = 1 << 2,
  /** Specifies a transform to normalize output. */
  GDDEPLOY_TRANSFORM_MEAN_STD  = 1 << 3
} TransformFlag;

/**
 * Holds the coordinates of a rectangle.
 */
typedef struct TransformRect {
  /** Holds the rectangle top. */
  uint32_t top;
  /** Holds the rectangle left side. */
  uint32_t left;
  /** Holds the rectangle width. */
  uint32_t width;
  /** Holds the rectangle height. */
  uint32_t height;
} TransformRect;

/**
 * Specifies data type.
 */
typedef enum {
  /** Specifies the data type to uint8. */
  GDDEPLOY_TRANSFORM_UINT8,
  /** Specifies the data type to float32. */
  GDDEPLOY_TRANSFORM_FLOAT32,
  /** Specifies the data type to float16. */
  GDDEPLOY_TRANSFORM_FLOAT16,
  /** Specifies the data type to int16. */
  GDDEPLOY_TRANSFORM_INT16,
  /** Specifies the data type to int32. */
  GDDEPLOY_TRANSFORM_INT32,
  /** Specifies the number of data types. */
  GDDEPLOY_TRANSFORM_NUM
} TransformDataType;

/**
 * Specifies color format.
 */
typedef enum {
  /** Specifies ABGR-8-8-8-8 single plane. */
  GDDEPLOY_TRANSFORM_COLOR_FORMAT_ARGB,
  /** Specifies ABGR-8-8-8-8 single plane. */
  GDDEPLOY_TRANSFORM_COLOR_FORMAT_ABGR,
  /** Specifies BGRA-8-8-8-8 single plane. */
  GDDEPLOY_TRANSFORM_COLOR_FORMAT_BGRA,
  /** Specifies RGBA-8-8-8-8 single plane. */
  GDDEPLOY_TRANSFORM_COLOR_FORMAT_RGBA,
  /** Specifies RGB-8-8-8 single plane. */
  GDDEPLOY_TRANSFORM_COLOR_FORMAT_RGB,
  /** Specifies BGR-8-8-8 single plane. */
  GDDEPLOY_TRANSFORM_COLOR_FORMAT_BGR,
  /** Specifies the number of color formats. */
  GDDEPLOY_TRANSFORM_COLOR_FORMAT_NUM,
} TransformColorFormat;

/**
 * Holds the shape information.
 */
typedef struct TransformShape {
  /** Holds the dimension n. Normally represents batch size */
  uint32_t n;
  /** Holds the dimension c. Normally represents channel */
  uint32_t c;
  /** Holds the dimension h. Normally represents height */
  uint32_t h;
  /** Holds the dimension h. Normally represents width */
  uint32_t w;
} TransformShape;

/**
 * Holds the descriptions of a tensor.
 */
typedef struct TransformTensorDesc {
  /** Holds the shape of the tensor */
  TransformShape shape;
  /** Holds the data type of the tensor */
  TransformDataType data_type;
  /** Holds the color format of the tensor */
  TransformColorFormat color_format;
} TransformTensorDesc;

/**
 * Holds the parameters of a MeanStd tranformation.
 */
typedef struct TransformMeanStdParams {
  /** Holds a pointer of mean values */
  float mean[GDDEPLOY_TRANSFORM_MAX_CHNS];
  /** Holds a pointer of std values */
  float std[GDDEPLOY_TRANSFORM_MAX_CHNS];
} TransformMeanStdParams;

/**
 * Holds configuration parameters for a transform/composite session.
 */
typedef struct TransformConfigParams {
  /** Holds the mode of operation:  VGU (CE3226) or MLU (M370/CE3226)
   If VGU is configured, device_id is ignored. */
  TransformComputeMode compute_mode;

  /** Holds the Device ID to be used for processing. */
  int32_t device_id;

  /** User configure stream to be used. If NULL, the default stream is used.
   Ignored if MLU is not used. */
  cnrtQueue_t cnrt_queue;
} TransformConfigParams;

/**
 * Holds transform parameters for a transform call.
 */
typedef struct TransformParams {
  /** Holds a flag that indicates which transform parameters are valid. */
  uint32_t transform_flag;
  /** Hold a pointer of normalize value*/
  TransformMeanStdParams *mean_std_params;
  /** Hold a pointer of tensor desc of src */
  TransformTensorDesc *src_desc;

  /** Not used. Hold a pointer of tensor desc of dst */
  TransformTensorDesc *dst_desc;

  /** Holds a pointer to a list of source rectangle coordinates for
   a crop operation. */
  TransformRect *src_rect;
  /** Holds a pointer to list of destination rectangle coordinates for
   a crop operation. */
  TransformRect *dst_rect;
} TransformParams;


/**
 * @brief  Sets user-defined session parameters.
 *
 * If user-defined session parameters are set, they override the
 * Transform() function's default session.
 *
 * @param[in] config_params     A pointer to a structure that is populated
 *                              with the session parameters to be used.
 *
 * @return Returns 0 if this function run successfully, otherwise returns non-zero values.
 */
int TransformSetSessionParams(TransformConfigParams *config_params){}

/**
 * @brief Gets the session parameters used by Transform().
 *
 * @param[out] config_params    A pointer to a caller-allocated structure to be
 *                              populated with the session parameters used.
 *
 * @return Returns 0 if this function run successfully, otherwise returns non-zero values.
 */
int TransformGetSessionParams(TransformConfigParams *config_params){}

/**
 * @brief Performs a transformation on batched input images.
 *
 * @param[in]  src  A pointer to input batched buffers to be transformed.
 * @param[out] dst  A pointer to a caller-allocated location where
 *                  transformed output is to be stored.
 *                  @par When destination cropping is performed, memory outside
 *                  the crop location is not touched, and may contain stale
 *                  information. The caller must perform a memset before
 *                  calling this function if stale information must be
 *                  eliminated.
 * @param[in]  transform_params
 *                  A pointer to an CnBufSurfTransformParams structure
 *                  which specifies the type of transform to be performed. They
 *                  may include any combination of scaling, format conversion,
 *                  and cropping for both source and destination.
 * @return Returns 0 if this function run successfully, otherwise returns non-zero values.
 */
int Transform(BufSurface *src, BufSurface *dst, TransformParams *transform_params);

#ifdef __cplusplus
}
#endif