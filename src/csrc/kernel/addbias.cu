#include "addbias.h"

#include "util/debug_utils.h"
#include "util/cuda_utils.h"

namespace st::kernel {

/*
	addbiasKernel & addbias

	This performs point-wise addition of two arrays, `input` and `bias`, and
	store the output to `output`.

	Input:
		- input: the input array, [size]
		- bias: the bias array, [size]
		- size: the size of the input and bias array
	Output:
		- output: the output array, [size]
		  output[i] = input[i] + bias[i]
*/
template<typename T>
__global__ void addbiasKernel(
	T* output,
	const T* input,
	const T* bias,
	const int64_t size
) {
	typedef std::conditional_t<std::is_same_v<T, half>, half2, float2> T2;
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size/2; i += blockDim.x * gridDim.x) {
		T2 input_elem = ((const T2*)input)[i];
		T2 bias_elem = ((const T2*)bias)[i];
		T2 result_elem = {input_elem.x + bias_elem.x, input_elem.y + bias_elem.y};
		((T2*)output)[i] = result_elem;
	}
}

template<typename T>
void addbias(
	T* output,
	const T* input,
	const T* bias,
	const int64_t size
) {
	assert_whenever (size%2 == 0);
	const uint32_t blockSize = 256;
	const uint32_t gridSize = std::min((size/2 + blockSize - 1) / blockSize, 16384l);
	addbiasKernel<T><<<gridSize, blockSize>>>(output, input, bias, size);
}

template void addbias(half* output, const half* input, const half* bias, const int64_t size);
template void addbias(float* output, const float* input, const float* bias, const int64_t size);


/*
	addbiasBatchedKernel & addbiasBatched

	This performs batched addbias.

	Input:
		- input: the input array, [batch, size]
		- bias: the bias array, [size]
		- batch: the batch size
		- size: the size of the input and bias array
	Output:
		- output: the output array, [batch, size]
		  output[b][i] = input[b][i] + bias[i]
	TODO Optimize this kernel
*/
template<typename T>
__global__ void addbiasBatchedKernel(
	T* output,
	const T* input,
	const T* bias,
	const int64_t batch,
	const int64_t size
) {
	typedef std::conditional_t<std::is_same_v<T, half>, half2, float2> T2;
	#pragma unroll 4
	for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size * batch/2; i += blockDim.x * gridDim.x) {
		const int64_t s = i % (size/2);
		T2 input_elem = ((const T2*)input)[i];
		T2 bias_elem = ((const T2*)bias)[s];
		T2 result_elem = {input_elem.x + bias_elem.x, input_elem.y + bias_elem.y};
		((T2*)output)[i] = result_elem;
	}
}

template<typename T>
void addbiasBatched(
	T* output,
	const T* input,
	const T* bias,
	const int64_t batch,
	const int64_t size
) {
	assert_whenever (size%2 == 0);
	const uint32_t blockSize = 256;
	const uint32_t gridSize = std::min((size*batch/2 + blockSize - 1) / blockSize, 16384l);
	addbiasBatchedKernel<T><<<gridSize, blockSize>>>(output, input, bias, batch, size);
}

template void addbiasBatched(half* output, const half* input, const half* bias, const int64_t batch, const int64_t size);
template void addbiasBatched(float* output, const float* input, const float* bias, const int64_t batch, const int64_t size);

} // namespace st::kernel
