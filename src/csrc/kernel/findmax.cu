#include "findmax.h"

#include <cassert>

#include "util/cuda_utils.h"
#include "reduction.cuh"

namespace st::kernel {

/*
	findmaxKernel & findmax

	Find the maximum value in an array and return its index.
*/
template<typename T>
__global__ void findmaxBatchedKernel(
	int64_t* max_indices,
	const T* input,
	const int64_t batch_size,
	const int64_t length
) {
	__shared__ T s_max;
	T local_max = -65400;
	int64_t local_max_index;
	for (int64_t i = threadIdx.x; i < length; i += blockDim.x) {
		if (input[i + length*blockIdx.x] > local_max) {
			local_max = input[i + length*blockIdx.x];
			local_max_index = i;
		}
	}
	T max_val = (T)(blockDim.x <= 32 ? warpReduceMax((float)local_max) : blockReduceMax((float)local_max));
	if (threadIdx.x == 0) {
		s_max = max_val;
	}
	__syncthreads();
	if (local_max == s_max) {
		max_indices[blockIdx.x] = local_max_index;
	}
}

template<typename T>
void findmaxBatched(
	int64_t* max_indices,		// [batch_size]
	const T* input,			// [batch_size, length]
	const int64_t batch_size,
	const int64_t length
) {
	int64_t threads = 1024;
	findmaxBatchedKernel<T><<<batch_size, threads>>>(max_indices, input, batch_size, length);
}

template void findmaxBatched(int64_t* max_indices, const half* input, const int64_t batch_size, const int64_t length);
template void findmaxBatched(int64_t* max_indices, const float* input, const int64_t batch_size, const int64_t length);


}	// namespace st::kernel