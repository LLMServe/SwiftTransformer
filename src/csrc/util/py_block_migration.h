#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <torch/extension.h>

namespace st::util {

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor);

bool register_ipc_mem_handle(
	std::vector<int64_t> k_cache_handle_vec,
	std::vector<int64_t> v_cache_handle_vec,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,
	const std::vector<int64_t> &decoding_parallel_config
);

void migrate_blocks(
	const int64_t context_pp_size,
	const int64_t context_tp_size,

	const std::vector<int64_t> &context_block_indexes,

	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	const std::vector<int64_t> &decoding_block_indexes,

	torch::Tensor decoding_worker_k_cache,
	torch::Tensor decoding_worker_v_cache
);

} // namespace st::util
