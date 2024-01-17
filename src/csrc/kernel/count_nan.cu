#include "count_nan.h"

#include "util/cuda_utils.h"

namespace st::kernel {

template<typename T>
__global__ void countNanKernel(
	int* count,
	const T* arr,
	int n
) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
		if (arr[i] != arr[i]) {
			atomicAdd(count, 1);
		}
	}
}

template<typename T>
int countNan(
	const T* arr,
	int n
) {
	int* count;
	cudaMalloc(&count, sizeof(int));
	cudaMemset(count, 0, sizeof(int));

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	countNanKernel<T><<<numBlocks, blockSize>>> (count, arr, n);

	int res;
	cudaMemcpy(&res, count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(count);
	return res;
}

#define INSTANTIATE(T) \
	template int countNan<T>(const T* arr, int n);

INSTANTIATE(float)
INSTANTIATE(half)

}