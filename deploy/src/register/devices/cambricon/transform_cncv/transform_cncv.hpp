#pragma once

#include <map>
#include <memory>
#include <vector>

#include "cncv.h"

#include "core/mem/buf_surface.h"
#include "cncv_transform.h"

namespace gddeploy {

class CncvContext {
 public:
  explicit CncvContext(int dev_id) {
    device_id_ = dev_id;

#if CNRT_MAJOR_VERSION < 5
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, dev_id);
    cnrtSetCurrentDevice(dev);
#else
    cnrtSetDevice(device_id_);
#endif

#if CNRT_MAJOR_VERSION < 5
    cnrtSetDeviceFlag(1);
#endif

    TransformGetSessionParams(&params_);
    cncvCreate(&handle_);
    cncvSetQueue(handle_, params_.cnrt_queue);
  }

  virtual bool Process(const BufSurface& src, BufSurface* dst, TransformParams* transform_params) = 0;
  int GetDeviceId() { return device_id_; }

  virtual ~CncvContext() {
    if (handle_) cncvDestroy(handle_);
    TransformSetSessionParams(&params_);
  }

  static cncvPixelFormat GetPixFormat(BufSurfaceColorFormat format) {
    static std::map<BufSurfaceColorFormat, cncvPixelFormat> color_map{
        {GDDEPLOY_BUF_COLOR_FORMAT_YUV420, CNCV_PIX_FMT_I420}, {GDDEPLOY_BUF_COLOR_FORMAT_NV12, CNCV_PIX_FMT_NV12},
        {GDDEPLOY_BUF_COLOR_FORMAT_NV21, CNCV_PIX_FMT_NV21},   {GDDEPLOY_BUF_COLOR_FORMAT_BGR, CNCV_PIX_FMT_BGR},
        {GDDEPLOY_BUF_COLOR_FORMAT_RGB, CNCV_PIX_FMT_RGB},     {GDDEPLOY_BUF_COLOR_FORMAT_BGRA, CNCV_PIX_FMT_BGRA},
        {GDDEPLOY_BUF_COLOR_FORMAT_RGBA, CNCV_PIX_FMT_RGBA},   {GDDEPLOY_BUF_COLOR_FORMAT_ABGR, CNCV_PIX_FMT_ABGR},
        {GDDEPLOY_BUF_COLOR_FORMAT_ARGB, CNCV_PIX_FMT_ARGB},
    };
    return color_map[format];
  }

 protected:
  int device_id_ = 0;
  cncvHandle_t handle_ = nullptr;
  TransformConfigParams params_;
};  // class CncvContext

class YuvResizeCncvCtx : public CncvContext {
 public:
  explicit YuvResizeCncvCtx(int dev_id) : CncvContext(dev_id) {}

  bool Process(const BufSurface& src, BufSurface* dst, TransformParams* transform_params) override;
  ~YuvResizeCncvCtx() override {
    if (mlu_input_) cnrtFree(mlu_input_);
    if (mlu_output_) cnrtFree(mlu_output_);
    if (workspace_) cnrtFree(workspace_);
  };

 private:
  std::vector<void**> cpu_input_;
  std::vector<void**> cpu_output_;

  void** mlu_input_ = nullptr;
  void** mlu_output_ = nullptr;

  std::vector<cncvImageDescriptor> src_descs_;
  std::vector<cncvImageDescriptor> dst_descs_;
  std::vector<cncvRect> src_rois_;
  std::vector<cncvRect> dst_rois_;

  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
  size_t batch_size_ = 0;
  const int plane_number_ = 2;
};  // class YuvResizeCncvCtx

class Yuv2RgbxResizeCncvCtx : public CncvContext {
 public:
  explicit Yuv2RgbxResizeCncvCtx(int dev_id) : CncvContext(dev_id) {}

  bool Process(const BufSurface& src, BufSurface* dst, TransformParams* transform_params) override;
  ~Yuv2RgbxResizeCncvCtx() override {
    if (mlu_input_) cnrtFree(mlu_input_);
    if (mlu_output_) cnrtFree(mlu_output_);
    if (workspace_) cnrtFree(workspace_);
  }

 private:
  std::vector<void**> cpu_input_;
  std::vector<void**> cpu_output_;

  void** mlu_input_ = nullptr;
  void** mlu_output_ = nullptr;

  std::vector<cncvImageDescriptor> src_descs_;
  std::vector<cncvImageDescriptor> dst_descs_;
  std::vector<cncvRect> src_rois_;
  std::vector<cncvRect> dst_rois_;
  bool keep_aspect_ratio_;
  uint8_t pad_value_ = 0;
  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
  size_t batch_size_ = 0;
  const int plane_number_ = 2;
};

class RgbxToYuvCncvCtx : public CncvContext {
 public:
  explicit RgbxToYuvCncvCtx(int dev_id) : CncvContext(dev_id) {}
  bool Process(const BufSurface& src, BufSurface* dst, TransformParams* transform_params) override;
  ~RgbxToYuvCncvCtx() override {}

 private:
  cncvRect src_roi_;
  cncvImageDescriptor src_desc_;
  cncvImageDescriptor dst_desc_;
};  // class RgbxToYuvCncvCtx

class MeanStdCncvCtx : public CncvContext {
 public:
  explicit MeanStdCncvCtx(int dev_id) : CncvContext(dev_id) {}

  bool Process(const BufSurface& src, BufSurface* dst, TransformParams* transform_params) override;
  ~MeanStdCncvCtx() override {
    if (mlu_input_) cnrtFree(mlu_input_);
    if (mlu_output_) cnrtFree(mlu_output_);
    if (workspace_) cnrtFree(workspace_);
  }

 private:
  std::vector<void**> cpu_input_;
  std::vector<void**> cpu_output_;

  void** mlu_input_ = nullptr;
  void** mlu_output_ = nullptr;

  cncvImageDescriptor src_desc_;
  cncvImageDescriptor dst_desc_;

  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
  size_t batch_size_ = 0;
  float* mean_;
  float* std_;
};

class Rgbx2YuvResizeAndConvert {
 public:
  explicit Rgbx2YuvResizeAndConvert(int dev_id) {
    rgbx_yuv_ = std::make_shared<RgbxToYuvCncvCtx>(dev_id);
    yuv_resize_ = std::make_shared<YuvResizeCncvCtx>(dev_id);
    dev_id_ = dev_id;
  }
  ~Rgbx2YuvResizeAndConvert() {
    if (src_yuv_mlu_)  cnrtFree(src_yuv_mlu_);
  }
  bool Process(const BufSurface& src, BufSurface* dst, TransformParams* transform_params);
 private:
  int dev_id_;
  std::shared_ptr<RgbxToYuvCncvCtx> rgbx_yuv_;
  std::shared_ptr<YuvResizeCncvCtx> yuv_resize_;

  void* src_yuv_mlu_ = nullptr;
  uint32_t src_yuv_size_ = 0;
};

class Yuv2RgbxResizeWithMeanStdCncv {
 public:
  explicit Yuv2RgbxResizeWithMeanStdCncv(int dev_id) {
    mean_std_ = std::make_shared<MeanStdCncvCtx>(dev_id);
    resize_convert_ = std::make_shared<Yuv2RgbxResizeCncvCtx>(dev_id);
  }

  bool Process(const BufSurface& src, BufSurface* dst, TransformParams* transform_params);
  ~Yuv2RgbxResizeWithMeanStdCncv() {
    if (mlu_ptr_) cnrtFree(mlu_ptr_);
  }

 private:
  std::shared_ptr<MeanStdCncvCtx> mean_std_;
  std::shared_ptr<Yuv2RgbxResizeCncvCtx> resize_convert_;

  void* mlu_ptr_ = nullptr;
  size_t mlu_size_ = 0;
};

int GetBufSurfaceFromTensor(BufSurface* src, BufSurface* dst, TransformTensorDesc* tensor_desc);
int CncvTransform(BufSurface* src, BufSurface* dst, TransformParams* transform_params);

}  // namespace gddeploy