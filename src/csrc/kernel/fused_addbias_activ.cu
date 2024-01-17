#include "fused_addbias_activ.h"

#include <cassert>

#include "activations.cuh"
#include "util/debug_utils.h"
#include "util/cuda_utils.h"

namespace st::kernel {

/*
	fusedAddbiasBatchedActivation

	This kernel is used to add bias to input and then apply the activation function.
	We fuse the two kernels into one to improve performance (by reduce the number of memory accesses).

	Input:
		- input: the input array, [size]
		- bias: the bias array, [size]
		- size: the size of input and bias
	Output:
		- output: the output array, [size]
		  output[i] = activation(input[i] + bias[i])
*/
template<typename T, ActivationType ACTIVATION_TYPE>
__global__ void fusedAddbiasBatchedActivationKernel(
	T* output,
	const T* input,
	const T* bias,
	const int64_t batch,
	const int64_t size
) {
	typedef std::conditional_t<std::is_same_v<T, half>, half2, float2> T2;
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size * batch / 2; i += blockDim.x * gridDim.x) {
		const int64_t s = i % (size/2);
		T2 input_elem = ((const T2*)input)[i];
		T2 bias_elem = ((const T2*)bias)[s];
		T2 output_elem = {
			applyActivation<T, ACTIVATION_TYPE>(input_elem.x + bias_elem.x),
			applyActivation<T, ACTIVATION_TYPE>(input_elem.y + bias_elem.y)
		};
		((T2*)output)[i] = output_elem;
	}
}


template<typename T>
void fusedAddbiasBatchedActivation(
	T* output,
	const T* input,
	const T* bias,
	const int64_t batch,
	const int64_t size,
	ActivationType activation_type
) {
	assert_whenever (size%2 == 0);
	const uint32_t blockSize = 256;
	const uint32_t gridSize = (size*batch/2 + blockSize - 1) / blockSize;
	switch (activation_type) {
		case ActivationType::RELU:
			fusedAddbiasBatchedActivationKernel<T, ActivationType::RELU><<<gridSize, blockSize>>>(output, input, bias, batch, size);
			break;
		case ActivationType::SILU:
			fusedAddbiasBatchedActivationKernel<T, ActivationType::SILU><<<gridSize, blockSize>>>(output, input, bias, batch, size);
			break;
		case ActivationType::GELU:
			fusedAddbiasBatchedActivationKernel<T, ActivationType::GELU><<<gridSize, blockSize>>>(output, input, bias, batch, size);
			break;
		default:
			assert(false);
	}
}

template void fusedAddbiasBatchedActivation(half* output, const half* input, const half* bias, const int64_t batch, const int64_t size, ActivationType activation_type);
template void fusedAddbiasBatchedActivation(float* output, const float* input, const float* bias, const int64_t batch, const int64_t size, ActivationType activation_type);

}	// namespace st::kernel