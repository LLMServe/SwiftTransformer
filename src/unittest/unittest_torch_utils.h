#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

#include "util/cuda_utils.h"
#include "util/torch_utils.h"

// setupTorch - Set up pytorch's seed and default datatype
template<typename T>
inline void setupTorch() {
	torch::manual_seed(0);
	// torch::Device device(torch::kCUDA);
	torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(st::util::getTorchScalarType<T>()));
}

inline torch::Tensor getRandomTensor(at::IntArrayRef shape, float lower = -1, float upper = +1, torch::Device device = torch::kCUDA) {
	auto options = torch::TensorOptions().requires_grad(false).device(device);
	return torch::rand(shape, options) * (upper - lower) + lower;
}
