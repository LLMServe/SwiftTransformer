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
			// NOTE. GELU has many different implementations, this is the one currently
			// used in Google BERT repo (identical to OpenAI GPT). Also see the Gaussian
			// Error Linear Units paper: https://arxiv.org/abs/1606.08415
			constexpr float constant = 0.7978845608028654;	// sqrtf(2.0f / M_PI);
			return (T)(0.5f * (float)x * (1.0f + tanhf(constant * ((float)x + 0.044715f * __powf((float)x, 3.0f)))));
		}
		else {
			// No activation matches, raise an error
			assert(false);
		}
	}
}
