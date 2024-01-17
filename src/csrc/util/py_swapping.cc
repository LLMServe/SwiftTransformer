#include "py_swapping.h"

#include "torch_utils.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>	// for at::cuda::getCurrentCUDAStream()

namespace st::util {

// swap - Perform swapping between GPU blocks and CPU blocks
// The source_block_ids and target_block_ids are the block ids of the blocks to be swapped.
// source_block_ids[0] will be copied to target_block_ids[0] and so on
// `is_swap_in` defines whether the swap is a swap-in or swap-out (swap-in means
// to swap from CPU to GPU, swap-out means to swap from GPU to CPU)
//
// Here we do not pass a cudaStream to the function. Instead we use the current
// stream indicated by at::cuda::getCurrentCUDAStream(). So it is python's
// responsibility to set the current stream before calling this function.
// 
// Future work: Now the number of cudaMemcpyAsync calls is equal to 2x the number
// of blocks to swap. We can reduce the number of cudaMemcpyAsync calls by
// grouping nearby blocks together and perform a single invocation
void swap(
	const std::vector<int64_t> &source_block_ids,
	const std::vector<int64_t> &target_block_ids,
	const bool is_swap_in,

	torch::Tensor k_cache,
	torch::Tensor v_cache,
	torch::Tensor k_swap,
	torch::Tensor v_swap
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	size_t block_size_in_bytes = getTensorSizeInBytes(k_cache) / k_cache.size(0);
	int num_blocks_to_swap = source_block_ids.size();
	for (int i = 0; i < num_blocks_to_swap; i++) {
		int64_t source_block_id = source_block_ids[i];
		int64_t target_block_id = target_block_ids[i];

		if (is_swap_in) {
			// Copy from CPU to GPU
			cudaMemcpyAsync(
				(char*)k_cache.data_ptr() + target_block_id * block_size_in_bytes,
				(char*)k_swap.data_ptr() + source_block_id * block_size_in_bytes,
				block_size_in_bytes,
				cudaMemcpyHostToDevice,
				stream
			);
			cudaMemcpyAsync(
				(char*)v_cache.data_ptr() + target_block_id * block_size_in_bytes,
				(char*)v_swap.data_ptr() + source_block_id * block_size_in_bytes,
				block_size_in_bytes,
				cudaMemcpyHostToDevice,
				stream
			);
		} else {
			// Copy from GPU to CPU
			cudaMemcpyAsync(
				(char*)k_swap.data_ptr() + target_block_id * block_size_in_bytes,
				(char*)k_cache.data_ptr() + source_block_id * block_size_in_bytes,
				block_size_in_bytes,
				cudaMemcpyDeviceToHost,
				stream
			);
			cudaMemcpyAsync(
				(char*)v_swap.data_ptr() + target_block_id * block_size_in_bytes,
				(char*)v_cache.data_ptr() + source_block_id * block_size_in_bytes,
				block_size_in_bytes,
				cudaMemcpyDeviceToHost,
				stream
			);
		}
	}
}

} // namespace st::util