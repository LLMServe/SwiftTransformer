#include "rmsnorm.h"

#include "util/cuda_utils.h"
#include "kernel/reduction.cuh"

namespace st::kernel {

template<typename T>
__global__ void rmsnormKernel(
	T* output,			// [num_tokens, hidden_size]
	const T* input,		// [num_tokens, hidden_size]
	const T* weight,	// [hidden_size]
	const float epsilon,
	const int64_t hidden_size
) {
	// Step 1. Every thread computes some part of the sum of squares
	float square_sum = 0.0;
	__shared__ float inv_rms;
	for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
		const float x = input[blockIdx.x * hidden_size + i];
		square_sum += x * x;
	}
	// Step 2. Sum the squares across threads
	square_sum = blockReduceSum(square_sum);
	// Step 3. Compute the inverse root mean square
	if (threadIdx.x == 0) {
		inv_rms = rsqrtf(square_sum / hidden_size + epsilon);
	}
	__syncthreads();
	// Step 4. Compute the output
	for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
		const float x = input[blockIdx.x * hidden_size + i];
		const float w = weight[i];
		output[blockIdx.x * hidden_size + i] = x * w * inv_rms;
	}
}

template<typename T>
void rmsnorm(
    T* out,
    const T* input,

    const T* weight,
    const float epsilon,

	const int64_t num_tokens,
	const int64_t hidden_size
) {
	const int64_t block_size = std::min(hidden_size, 1024L);
	const int64_t grid_size = num_tokens;
	rmsnormKernel<<<grid_size, block_size>>>(out, input, weight, epsilon, hidden_size);
}

#define INSTANTIATE(T) \
	template void rmsnorm<T>( \
		T* out, \
		const T* input, \
		const T* weight, \
		const float epsilon, \
		const int64_t num_tokens, \
		const int64_t hidden_size \
	);

INSTANTIATE(float)
INSTANTIATE(half)

} // namespace st::kernel