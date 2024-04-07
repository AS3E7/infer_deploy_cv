/**
 * @file rkmedia_jpeg_codec.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-02-07
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#ifndef __RKMEDIA_JPEG_CODEC_H__
#define __RKMEDIA_JPEG_CODEC_H__

#if defined(WITH_RV1126)
#include <functional>
#include <memory>
#include <vector>

namespace gddi {
namespace codec {

class RkMediaJpegCodecPrivate;

using JpegCodecCallback = std::function<void(const uint8_t *data, const size_t dlen)>;

class RkMediaJpegCodec {
public:
    RkMediaJpegCodec();
    ~RkMediaJpegCodec();

    /**
     * @brief 初始化
     * 
     * @param width 
     * @param height 
     * @param quality 
     * @return true 
     * @return false 
     */
    bool init_codecer(const size_t width, const size_t height, const int quality = 85);

    /**
     * @brief 编解码
     * 
     * @param input 
     * @param callback 
     * @return true 
     * @return false 
     */
    bool codec_image(const uint8_t *input, const JpegCodecCallback &callback);

    /**
     * @brief 编解码
     * 
     * @param input 
     * @param jepg_data 
     * @return true 
     * @return false 
     */
    bool codec_image(const uint8_t *input, std::vector<uint8_t> &jepg_data);

    /**
     * @brief 保存图片
     * 
     * @param input 
     * @param path 
     * @return true 
     * @return false 
     */
    const bool save_image(const uint8_t *input, const char *path);

private:
    std::unique_ptr<RkMediaJpegCodecPrivate> impl_;
};

}// namespace codec
}// namespace gddi

#endif
#endif