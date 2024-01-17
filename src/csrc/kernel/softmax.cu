#include "softmax.h"

#include "reduction.cuh"
#include "util/cuda_utils.h"

namespace st::kernel {

/*
    scaleMaskSoftmaxKernel &
    scaleMaskSoftmax

    This kernel applies scaling (*1/sqrt(dk)), masking, and softmax to the input matrix (attention matrix).

	Input:
		- input: [num_heads, input_len, input_len]
		- scale: 1/sqrt(dk)
	Output:
		- output: [num_heads, input_len, input_len]
		  output[head][row] = softmax(masking(input[head][row] * scale))
*/
template<typename T>
__global__ void scaleMaskSoftmaxKernel(
	T* output,
	const T* input,
	const float scale,
	const int64_t num_heads,
	const int64_t input_len
) {
    const int64_t h = blockIdx.x;
	for (int64_t r = 0; r < input_len; ++r) {
		float local_max = -1e20f, local_sum = 0.0;
		__shared__ float s_max, s_sum;
		for (int64_t c = threadIdx.x; c < input_len; c += blockDim.x) {
			float val = input[INDEX_3D(num_heads, input_len, input_len, h, r, c)];
			val *= scale;
			val += r >= c ? 0 : -10000.0;
			output[INDEX_3D(num_heads, input_len, input_len, h, r, c)] = val;
			local_max = local_max > val ? local_max : val;
		}
		float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
		if (threadIdx.x == 0) {
			s_max = max_val;
		}
		__syncthreads();
		for (int64_t c = threadIdx.x; c < input_len; c += blockDim.x) {
			float val = output[INDEX_3D(num_heads, input_len, input_len, h, r, c)];
			val = __expf(val - s_max);
			output[INDEX_3D(num_heads, input_len, input_len, h, r, c)] = val;
			local_sum += val;
		}
		float sum = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
		if (threadIdx.x == 0) {
			s_sum = sum;
		}
		__syncthreads();
		float to_mult = __fdividef((float)1.0, s_sum+(float)(1e-6));
		for (int64_t c = threadIdx.x; c < input_len; c += blockDim.x) {
			float val = output[INDEX_3D(num_heads, input_len, input_len, h, r, c)];
			val *= to_mult;
			output[INDEX_3D(num_heads, input_len, input_len, h, r, c)] = val;
		}
	}
}

template<typename T>
void scaleMaskSoftmax(
	T* output,
	const T* input,
	const float scale,
	const int64_t num_heads,
	const int64_t input_len
) {
	uint32_t block_dim = std::min(input_len, 256l);
    scaleMaskSoftmaxKernel<<<num_heads, block_dim>>>(
		output,
		input,
		scale,
		num_heads,
		input_len
	);
}

template void scaleMaskSoftmax(
	float* output, const float* input,
	const float scale,
	const int64_t num_heads, const int64_t input_len
);
template void scaleMaskSoftmax(
	half* output, const half* input,
	const float scale,
	const int64_t num_heads, const int64_t input_len
);



/*
	scaleSoftmaxKernel & scaleSoftmax

	This performs scale & softmax on a batch of 1-D array
	This function is used in the regression stage

	Input:
		- input: the input array, typically it is the last row of the attention matrix, [num_heads, seq_len]
		- scale: the scale factor, typically it is 1/sqrt(head_dim)
	Output:
		- output: the output array, [num_heads, seq_len]
		  output[head] = softmax(input[head] * scale)
*/
template<typename T>
__global__ void scaleSoftmaxKernel(
	T* output,
	const T* input,
	const float scale,
	const int64_t seq_len
) {
	__shared__ float s_max, s_sum;
	const int64_t index_start = seq_len*blockIdx.x + threadIdx.x;
	const int64_t index_end = seq_len*blockIdx.x + seq_len;

	float local_max = -1e20f;
	for (int64_t index = index_start; index < index_end; index += blockDim.x) {
		float val = input[index];
		val *= scale;
		local_max = local_max > val ? local_max : val;
	}

	float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
	if (threadIdx.x == 0) {
		s_max = max_val;
	}
	__syncthreads();

	float local_sum = 0;
	for (int64_t index = index_start; index < index_end; index += blockDim.x) {
		float val = input[index];
		val *= scale;
		val = __expf(val - s_max);
		local_sum += val;
		output[index] = val;
	}

	float sum = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
	if (threadIdx.x == 0) {
		s_sum = sum;
	}
	__syncthreads();

	float to_mult = __fdividef((float)1.0, s_sum+1e-6f);
	for (int64_t index = index_start; index < index_end; index += blockDim.x) {
		float val = output[index];
		val *= to_mult;
		output[index] = (T)val;
	}
}

template<typename T>
void scaleSoftmax(
	T* output,
	const T* input,
	const float scale,
	const int64_t num_heads,
	const int64_t seq_len
) {
	uint32_t block_dim = std::min(seq_len, 256l);
    scaleSoftmaxKernel<<<num_heads, block_dim>>>(output, input, scale, seq_len);
}

template void scaleSoftmax(
	float* output, const float* input,
	const float scale,
	const int64_t num_heads, const int64_t seq_len
);
template void scaleSoftmax(
	half* output, const half* input,
	const float scale,
	const int64_t num_heads, const int64_t seq_len
);

}	// namespace st::kernel