#ifndef GDDEPLOY_CORE_DATATYPE_H_
#define GDDEPLOY_CORE_DATATYPE_H_

#include <string>
#include <vector>
#include "common/logger.h"
#include "core/infer_server.h"
#include "core/shape.h"

// #include "core/gddeploy.h"

namespace gddeploy {
namespace detail {

inline std::string DataTypeStr(DataType type) noexcept {
  switch (type) {
#define DATATYPE2STR(type) \
  case DataType::type:     \
    return #type;
    DATATYPE2STR(UINT8)
    DATATYPE2STR(FLOAT16)
    DATATYPE2STR(FLOAT32)
    DATATYPE2STR(INT32)
    DATATYPE2STR(INT16)
#undef DATATYPE2STR
    default:
      GDDEPLOY_ERROR("[InferServer] [DataTypeStr] Unsupported data type");
      return "INVALID";
  }
}

inline std::string DimOrderStr(DimOrder order) noexcept {
  switch (order) {
#define DIMORDER2STR(order) \
  case DimOrder::order:     \
    return #order;
    DIMORDER2STR(NCHW)
    DIMORDER2STR(NHWC)
    DIMORDER2STR(HWCN)
    DIMORDER2STR(TNC)
    DIMORDER2STR(NTC)
#undef DIMORDER2STR
    default:
      GDDEPLOY_ERROR("[InferServer] [DimOrderStr] Unsupported dim order");
      return "INVALID";
  }
}

// shape corresponding to src_data
bool CastDataType(void *src_data, void *dst_data, DataType src_dtype, DataType dst_dtype, const Shape &shape);

// shape corresponding to src_data
bool TransLayout(void* src_data, void* dst_data, DataLayout src_layout, DataLayout dst_layout, const Shape& shape);

}  // namespace detail

template <typename dtype>
inline std::vector<dtype> DimNHWC2NCHW(const std::vector<dtype>& dim) {
  switch (dim.size()) {
    case 1:
      return dim;
    case 2:
      return dim;
    case 3:
      return std::vector<dtype>({dim[0], dim[2], dim[1]});
    case 4:
      return std::vector<dtype>({dim[0], dim[3], dim[1], dim[2]});
    case 5:
      return std::vector<dtype>({dim[0], dim[4], dim[1], dim[2], dim[3]});
    default:
      GDDEPLOY_ERROR("[InferServer] [DimNHWC2NCHW] Unsupported dimension");
  }
  return {};
}

template <typename dtype>
inline std::vector<dtype> DimNCHW2NHWC(const std::vector<dtype>& dim) {
  switch (dim.size()) {
    case 1:
      return dim;
    case 2:
      return dim;
    case 3:
      return std::vector<dtype>({dim[0], dim[2], dim[1]});
    case 4:
      return std::vector<dtype>({dim[0], dim[2], dim[3], dim[1]});
    case 5:
      return std::vector<dtype>({dim[0], dim[2], dim[3], dim[4], dim[1]});
    default:
      GDDEPLOY_ERROR("[InferServer] [DimNCHW2NHWC] Unsupported dimension");
  }
  return {};
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v_t) {
  os << "vec {";
  for (size_t i = 0; i < v_t.size() - 1; ++i) {
    os << v_t[i] << ", ";
  }
  if (v_t.size() > 0) os << v_t[v_t.size() - 1];
  os << "}";
  return os;
}
}  // namespace gddeploy

#endif  // GDDEPLOY_CORE_DATATYPE_H_
