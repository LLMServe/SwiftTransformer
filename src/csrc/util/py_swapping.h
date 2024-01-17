#pragma once

#include <vector>

#include <torch/torch.h>
#include <cuda_runtime.h>

namespace st::util {

void swap(
	const std::vector<int64_t> &source_block_ids,
	const std::vector<int64_t> &target_block_ids,
	const bool is_swap_in,

	torch::Tensor k_cache,
	torch::Tensor v_cache,
	torch::Tensor k_swap,
	torch::Tensor v_swap
);

} // namespace st::util
