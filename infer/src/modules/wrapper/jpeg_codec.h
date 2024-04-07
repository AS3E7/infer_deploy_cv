/**
 * @file jpeg_codec.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-02-08
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#ifndef __JPEG_CODEC_H__
#define __JPEG_CODEC_H__

#include <functional>
#include <memory>
#include <vector>

namespace gddi {
namespace codec {

enum class CodecType { kDecoder, kEncoder };
enum class PixelFormat { kYUV420P, KNV12, kBGR888 };

struct JpegCodecParam {
    CodecType type;
    int width;
    int height;
    PixelFormat format{PixelFormat::kYUV420P};
    int quality{85};
};

using JpegCodecCallback = std::function<void(const uint8_t *data, const uint32_t dlen)>;

class JpegCodec {
public:
    JpegCodec();
    virtual ~JpegCodec();

    /**
     * @brief 
     * 
     * @param option 
     * @return true 
     * @return false 
     */
    virtual bool init_codecer(const JpegCodecParam &option) = 0;

    /**
     * @brief 编解码
     * 
     * @param input 
     * @param callback 
     * @return true 
     * @return false 
     */
    virtual bool codec_image(const uint8_t *input, const JpegCodecCallback &callback) = 0;

    /**
     * @brief 保存图片
     * 
     * @param input 
     * @param path 
     * @return true 
     * @return false 
     */
    virtual bool save_image(const uint8_t *input, const char *path);
};

}// namespace codec
}// namespace gddi

#endif