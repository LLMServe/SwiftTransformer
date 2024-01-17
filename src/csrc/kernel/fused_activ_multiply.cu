#include "fused_activ_multiply.h"

#include <cassert>

#include "util/cuda_utils.h"
#include "util/debug_utils.h"
#include "kernel/activations.cuh"

namespace st::kernel {

/*
	fusedSiluMultiplyKernel

	Given two arrays, input1 and input2, compute the following:
		output[i] = silu(input1[i]) * input2[i]
*/

template<typename T, ActivationType ACTIVATION_TYPE>
__global__ void fusedActivationMultiplyKernel(
	T* output,
	const T* input1,
	const T* input2,
	int64_t n
) {
	typedef std::conditional_t<std::is_same_v<T, half>, half2, float2> T2;
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n/2; i += blockDim.x * gridDim.x) {
		T2 input1_elem = ((const T2*)input1)[i];
		T2 input2_elem = ((const T2*)input2)[i];
		T2 output_elem = {
			applyActivation<T, ACTIVATION_TYPE>(input1_elem.x) * input2_elem.x,
			applyActivation<T, ACTIVATION_TYPE>(input1_elem.y) * input2_elem.y
		};
		((T2*)output)[i] = output_elem;
	}
}

template<typename T>
void fusedActivationMultiply(
	T* output,
	const T* input1,
	const T* input2,
	int64_t n,
	ActivationType activation_type
) {
	assert_whenever (n%2 == 0);
	int blockSize = 256;
	int gridSize = (n/2 + blockSize - 1) / blockSize;
	switch (activation_type) {
		case ActivationType::RELU:
			fusedActivationMultiplyKernel<T, ActivationType::RELU><<<gridSize, blockSize>>>(output, input1, input2, n);
			return;
		case ActivationType::SILU:
			fusedActivationMultiplyKernel<T, ActivationType::SILU><<<gridSize, blockSize>>>(output, input1, input2, n);
			return;
		case ActivationType::GELU:
			fusedActivationMultiplyKernel<T, ActivationType::GELU><<<gridSize, blockSize>>>(output, input1, input2, n);
			return;
		default:
			assert (false);
	}
}

#define INSTANTIALIZE(T) \
	template void fusedActivationMultiply<T>( \
		T* output, \
		const T* input1, \
		const T* input2, \
		int64_t n, \
		ActivationType activationType \
	);

INSTANTIALIZE(half)
INSTANTIALIZE(float)

}	// namespace st::kernel