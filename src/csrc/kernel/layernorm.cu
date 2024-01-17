#include "layernorm.h"

#include <cassert>
#include <cstdlib>

#include "util/cuda_utils.h"
#include "kernel/reduction.cuh"
#include "util/debug_utils.h"

namespace st::kernel {

constexpr int WARP_SIZE = 32; 
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

template<typename T, bool HAVE_PRE_LAYERNORM_BIAS>
__global__ void layernormKernel(
	T* __restrict__ out,			// [num_tokens, hidden_size]
	const T* __restrict__ input,	// [num_tokens, hidden_size]
	const T* __restrict__ weight,	// [hidden_size]
	const T* __restrict__ bias,		// [hidden_size]
	const float epsilon,
	const int64_t num_tokens,
	const int64_t hidden_size,
	T* __restrict__ biased_input,	// [num_tokens, hidden_size]
	const T* __restrict__ pre_layernorm_bias	// [hidden_size]
) {
	typedef std::conditional_t<std::is_same<T, half>::value, half2, float2> T2;
	
	extern __shared__ float shared_mem[];	// Have hidden_size * sizeof(T) bytes
	T2* input_buf = (T2*)shared_mem;	// [hidden_size/2], a cache for input[]

	// Step 1. Calculate (local) mean and variance
	__shared__ float s_mean, s_variance;	// We use float here, or the value may exceed the range of half
	float mean = 0.0, variance = 0.0;

	#pragma unroll 4
	for (int64_t idx = threadIdx.x; idx < hidden_size/2; idx += blockDim.x) {
		T2 elem = ((T2*)input)[blockIdx.x * hidden_size/2 + idx];
		if constexpr(HAVE_PRE_LAYERNORM_BIAS) {
			const T2 pre_layernorm_bias_elem = ((T2*)pre_layernorm_bias)[blockIdx.x * hidden_size/2 + idx];
			elem.x += pre_layernorm_bias_elem.x;
			elem.y += pre_layernorm_bias_elem.y;
			((T2*)biased_input)[blockIdx.x * hidden_size/2 + idx] = elem;
		}
		input_buf[idx] = elem;
		const float x = (float)elem.x;
		const float y = (float)elem.y;
		mean += x+y;
		variance += x*x + y*y;
	}

	// Step 2. Reduce mean and variance
	// Step 2.1 Reduce within the warp
	#pragma unroll
	for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
		mean += __shfl_down_sync(0xffffffff, mean, offset);
		variance += __shfl_down_sync(0xffffffff, variance, offset);
	}
	static __shared__ float reduction_wksp[2][NUM_WARPS];	// 32 = max block size (1024) / WARP_SIZE (32)
	if ((threadIdx.x & 31) == 0) {
		reduction_wksp[0][threadIdx.x >> 5] = mean;
		reduction_wksp[1][threadIdx.x >> 5] = variance;
	}
	__syncthreads();

	// Step 2.2 Reduce within the block
	if (threadIdx.x < NUM_WARPS) {
		mean = reduction_wksp[0][threadIdx.x];
		variance = reduction_wksp[1][threadIdx.x];
	}
	#pragma unroll
	for (int offset = NUM_WARPS/2; offset > 0; offset /= 2) {
		mean += __shfl_down_sync(0xffffffff, mean, offset);
		variance += __shfl_down_sync(0xffffffff, variance, offset);
	}

	if (threadIdx.x == 0) {
		float hidden_size_fp = (float)hidden_size;
		s_mean = mean / hidden_size_fp;
		s_variance = rsqrtf(variance / hidden_size_fp - s_mean * s_mean + epsilon);
	}
	__syncthreads();

	// Step 3. Normalize
	T final_mean = (T)s_mean;
	T final_variance = (T)s_variance;
	#pragma unroll 4
	for (int64_t idx = threadIdx.x; idx < hidden_size/2; idx += blockDim.x) {
		T2 x = input_buf[idx];
		T2 weight_elem = ((T2*)weight)[idx];
		T2 bias_elem = ((T2*)bias)[idx];
		((T2*)out)[blockIdx.x * hidden_size/2 + idx] =  {
			((x.x - final_mean) * final_variance) * weight_elem.x + bias_elem.x,
			((x.y - final_mean) * final_variance) * weight_elem.y + bias_elem.y
		};
	}
}

template<typename T>
void layernorm(
    T* out,
    const T* input,

    const T* weight,
    const T* bias,
    const float epsilon,

	const int64_t num_tokens,
	const int64_t hidden_size,

	T* biased_input,			// Default: nullptr
	const T* pre_layernorm_bias	// Default: nullptr
) {
	dim3 grid(num_tokens);
	dim3 block(NUM_THREADS);
	assert_whenever (hidden_size % NUM_THREADS == 0);

	if (pre_layernorm_bias == nullptr) {
		assert_whenever (biased_input == nullptr);
		layernormKernel<T, false><<<grid, block, hidden_size*sizeof(T)>>>(
			out,
			input,
			weight,
			bias,
			epsilon,
			num_tokens,
			hidden_size,
			nullptr,
			nullptr);
	} else {
		assert_whenever (biased_input != nullptr);
		layernormKernel<T, true><<<grid, block, hidden_size*sizeof(T)>>>(
			out,
			input,
			weight,
			bias,
			epsilon,
			num_tokens,
			hidden_size,
			biased_input,
			pre_layernorm_bias);
	}
}

template void layernorm(
    float* out, const float* input,
    const float* weight, const float* bias, const float epsilon,
	const int64_t num_tokens, const int64_t hidden_size, float* biased_input, const float* pre_layernorm_bias
);
template void layernorm(
    half* out, const half* input,
    const half* weight, const half* bias, const float epsilon,
	const int64_t num_tokens, const int64_t hidden_size, half* biased_input, const half* pre_layernorm_bias
);

}	// namespace st::kernel