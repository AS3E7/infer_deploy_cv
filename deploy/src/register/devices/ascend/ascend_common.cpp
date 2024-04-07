#include "ascend_common.h"

// #include "common/logger.h"
#include "core/mem/buf_surface.h"

#include <string>


std::string GetAppErrCodeInfo(const APP_ERROR err)
{
    if ((err < APP_ERR_ACL_END) && (err >= APP_ERR_ACL_FAILURE)) {
        return APP_ERR_ACL_LOG_STRING[((err < 0) ? (err + APP_ERR_ACL_END + 1) : err)];
    } else if ((err < APP_ERR_COMM_END) && (err > APP_ERR_COMM_BASE)) {
        return (err - APP_ERR_COMM_BASE) <
            (int)sizeof(APP_ERR_COMMON_LOG_STRING) / (int)sizeof(APP_ERR_COMMON_LOG_STRING[0]) ?
            APP_ERR_COMMON_LOG_STRING[err - APP_ERR_COMM_BASE] :
            "Undefine the error code information";
    } else if ((err < APP_ERR_DVPP_END) && (err > APP_ERR_DVPP_BASE)) {
        return (err - APP_ERR_DVPP_BASE) <
            (int)sizeof(APP_ERR_DVPP_LOG_STRING) / (int)sizeof(APP_ERR_DVPP_LOG_STRING[0]) ?
            APP_ERR_DVPP_LOG_STRING[err - APP_ERR_DVPP_BASE] :
            "Undefine the error code information";
    } else if ((err < APP_ERR_QUEUE_END) && (err > APP_ERR_QUEUE_BASE)) {
        return (err - APP_ERR_QUEUE_BASE) <
            (int)sizeof(APP_ERR_QUEUE_LOG_STRING) / (int)sizeof(APP_ERR_QUEUE_LOG_STRING[0]) ?
            APP_ERR_QUEUE_LOG_STRING[err - APP_ERR_QUEUE_BASE] :
            "Undefine the error code information";
    } else {
        return "Error code unknown";
    }
}

void AssertErrorCode(int code, std::string file, std::string function, int line)
{
    if (code != APP_ERR_OK) {
        // GDDEPLOY_ERROR("Failed at {}->{}->{}: error code={}", file, function, line, code);
        exit(code);
    }
}

void CheckErrorCode(int code, std::string file, std::string function, int line)
{
    if (code != APP_ERR_OK) {
        // GDDEPLOY_ERROR("Failed at {}->{}->{}: error code={}", file, function, line, code);
    }
}



/*
 * @description: Get the size of buffer used to save image for VPC according to width, height and format
 * @param  width specifies the width of the output image
 * @param  height specifies the height of the output image
 * @param  format specifies the format of the output image
 * @param: vpcSize is used to save the result size
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR GetVpcDataSize(uint32_t width, uint32_t height, acldvppPixelFormat format, uint32_t &vpcSize)
{
    // Check the invalid format of VPC function and calculate the output buffer size
    if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
        // GDDEPLOY_ERROR("Format[{}] for VPC is not supported, just support NV12 or NV21.", format);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    uint32_t widthStride = DVPP_ALIGN_UP(width, VPC_WIDTH_ALIGN);
    uint32_t heightStride = DVPP_ALIGN_UP(height, VPC_HEIGHT_ALIGN);
    vpcSize = widthStride * heightStride * YUV_BGR_SIZE_CONVERT_3 / YUV_BGR_SIZE_CONVERT_2;
    return APP_ERR_OK;
}

/*
 * @description: Get the aligned width and height of the input image according to the image format
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: format specifies the image format
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: APP_ERR_OK if success, other values if failure
 */

APP_ERROR GetVpcInputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                            uint32_t &widthStride, uint32_t &heightStride)
{
    uint32_t inputWidthStride;
    // Check the invalidty of input format and calculate the input width stride
    if (format >= PIXEL_FORMAT_YUV_400 && format <= PIXEL_FORMAT_YVU_SEMIPLANAR_444) {
        // If format is YUV SP, keep widthStride not change.
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
    } else if (format >= PIXEL_FORMAT_YUYV_PACKED_422 && format <= PIXEL_FORMAT_VYUY_PACKED_422) {
        // If format is YUV422 packed, image size = H x W * 2;
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * YUV422_WIDTH_NU;
    } else if (format >= PIXEL_FORMAT_YUV_PACKED_444 && format <= PIXEL_FORMAT_BGR_888) {
        // If format is YUV444 packed or RGB, image size = H x W * 3;
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * YUV444_RGB_WIDTH_NU;
    } else if (format >= PIXEL_FORMAT_ARGB_8888 && format <= PIXEL_FORMAT_BGRA_8888) {
        // If format is XRGB8888, image size = H x W * 4
        inputWidthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH) * XRGB_WIDTH_NU;
    } else {
        // GDDEPLOY_ERROR("Input format[{}] for VPC is invalid, please check it.", format);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    uint32_t inputHeightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
    // Check the input validity width stride.
    if (inputWidthStride > MAX_RESIZE_WIDTH || inputWidthStride < MIN_RESIZE_WIDTH) {
        // GDDEPLOY_ERROR("Input width stride {} is invalid, not in [{}< {}].", inputWidthStride, MIN_RESIZE_WIDTH, MAX_RESIZE_WIDTH);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    // Check the input validity height stride.
    if (inputHeightStride > MAX_RESIZE_HEIGHT || inputHeightStride < MIN_RESIZE_HEIGHT) {
        // GDDEPLOY_ERROR("Input width stride {} is invalid, not in [{}< {}].", inputWidthStride, MIN_RESIZE_WIDTH, MAX_RESIZE_WIDTH);
        return APP_ERR_COMM_INVALID_PARAM;
    }
    widthStride = inputWidthStride;
    heightStride = inputHeightStride;
    return APP_ERR_OK;
}

/*
 * @description: Get the aligned width and height of the output image according to the image format
 * @param: width specifies the width before alignment
 * @param: height specifies the height before alignment
 * @param: format specifies the image format
 * @param: widthStride is used to save the width after alignment
 * @param: heightStride is used to save the height after alignment
 * @return: APP_ERR_OK if success, other values if failure
 */
APP_ERROR GetVpcOutputStrideSize(uint32_t width, uint32_t height, acldvppPixelFormat format,
                                             uint32_t &widthStride, uint32_t &heightStride)
{
    // Check the invalidty of output format and calculate the output width and height
    if (format != PIXEL_FORMAT_YUV_SEMIPLANAR_420 && format != PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
        // GDDEPLOY_ERROR("Output format[{}] for VPC is not supported, just support NV12 or NV21.", format);
        return APP_ERR_COMM_INVALID_PARAM;
    }

    widthStride = DVPP_ALIGN_UP(width, VPC_STRIDE_WIDTH);
    heightStride = DVPP_ALIGN_UP(height, VPC_STRIDE_HEIGHT);
    return APP_ERR_OK;
}

acldvppPixelFormat convertFormat(BufSurfaceColorFormat color_format)
{
    // 颜色空间格式转换
    if (color_format == GDDEPLOY_BUF_COLOR_FORMAT_BGR){
        return PIXEL_FORMAT_BGR_888;
    } else if (color_format == GDDEPLOY_BUF_COLOR_FORMAT_RGB){
        return PIXEL_FORMAT_RGB_888;
    } 
}