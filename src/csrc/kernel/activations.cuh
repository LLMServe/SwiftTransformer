#pragma once

#include <cassert>

#include "activation_types.h"
#include "util/cuda_utils.h"

// See comments in activation_types.h
namespace st::kernel {
	template<typename T, ActivationType activation_type>
	__forceinline__ __device__ T applyActivation(const T &x) {
		if constexpr (activation_type == ActivationType::RELU) {
			return x > (T)0 ? x : (T)0;
		}
		else if constexpr (activation_type == ActivationType::SILU) {
			return (T)((float)x / (1.0f + __expf((float)-x)));
		}
		else if constexpr (activation_type == ActivationType::GELU) {
			// NOTE. GELU has many different implementations, 
			// this is the one currently used in vllm-project/vllm repo (gelu_new_kernel).
			// file url: https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu
			const float x3 = (float) (x * x * x);
			const T t = (T) tanhf((T) (0.79788456f * (float) (x + (T) (0.044715f * x3))));
			return ((T) 0.5) * x * (((T) 1.0) + t);
		}
		else {
			// No activation matches, raise an error
			assert(false);
		}
	}
}
