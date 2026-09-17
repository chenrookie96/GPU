#pragma once
#include <torch_musa/csrc/aten/musa/MUSAContext.h>
#include <deep_ep/common/exception.cuh>

namespace at {
  namespace musa {
    using namespace c10::musa;
    static musaDataType  ScalarTypeToMusaDataType(const c10::ScalarType& scalar_type){
      switch (scalar_type) {
      case c10::ScalarType::BFloat16:
        return MUSA_R_16BF;
      case c10::ScalarType::Float8_e4m3fn:
        return MUSA_R_8F_E4M3;
      case c10::ScalarType::Float8_e5m2:
        return MUSA_R_8F_E5M2;
      default:
          EP_HOST_ASSERT(0);  
      }
    }
  }
}

namespace at {
    static constexpr const c10::DeviceType& kCUDA = kMUSA;
namespace cuda {
using CUDAStream = at::musa::MUSAStream;
inline CUDAStream getCurrentCUDAStream() {
    return at::musa::getCurrentMUSAStream();
}
inline CUDAStream getStreamFromPool(const bool is_high_priority = false, int device_index = -1) {
    return at::musa::getStreamFromPool(is_high_priority, device_index);
}
inline void setCurrentCUDAStream(const CUDAStream stream) {
    at::musa::setCurrentMUSAStream(stream);
}
inline auto ScalarTypeToCudaDataType(const at::ScalarType dtype) {
    return at::musa::ScalarTypeToMusaDataType(dtype);
}
}  // namespace cuda
}  // namespace at

namespace torch {
    // extern const c10::DeviceType kMUSA;
    static constexpr const c10::DeviceType& kCUDA = kMUSA;
}