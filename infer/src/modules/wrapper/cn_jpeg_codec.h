/**
 * @file cn_jpeg_codec.h
 * @author zhdotcai (caizhehong@gddi.com.cn)
 * @brief 
 * @version 0.1
 * @date 2023-02-08
 * 
 * @copyright Copyright (c) 2023 by GDDI
 * 
 */

#ifndef __CN_JPEG_CODEC_H__
#define __CN_JPEG_CODEC_H__

#if defined(WITH_MLU220)
#include "jpeg_codec.h"
#include "cn_jpeg_dec.h"
#include "cn_jpeg_enc.h"
#include <functional>
#include <memory>
#include <vector>

namespace gddi {
namespace codec {

class CNJpegCodecPrivate;

class CNJpegCodec {
public:
    CNJpegCodec();
    ~CNJpegCodec();

    bool init_codecer(const JpegCodecParam &option);
    bool codec_image(const std::shared_ptr<cncodecFrame> &input, const JpegCodecCallback &callback);
    bool save_image(const std::shared_ptr<cncodecFrame> &input, const char *path);

private:
    std::unique_ptr<CNJpegCodecPrivate> impl_;
};

}// namespace codec
}// namespace gddi

#endif
#endif