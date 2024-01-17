#pragma once

#include <torch/torch.h>
#include <cuda_fp16.h>

namespace st::util {

template<typename T>
inline torch::ScalarType getTorchScalarType() {
	if (std::is_same<T, float>::value) {
		return torch::kFloat;
	} else if (std::is_same<T, half>::value) {
		return torch::kHalf;
	} else {
		throw std::runtime_error("Unsupported type");
	}
}

inline void* convertTensorToRawPtr(torch::Tensor& tensor) {
	if (tensor.scalar_type() == torch::kFloat) {
		return tensor.data_ptr<float>();
	} else if (tensor.scalar_type() == torch::kHalf) {
		return tensor.data_ptr<at::Half>();
	} else {
		throw std::runtime_error("Unsupported type");
	}
}

inline size_t getTensorSizeInBytes(torch::Tensor tensor)
{
    return tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
}

}	// namespace st::util
