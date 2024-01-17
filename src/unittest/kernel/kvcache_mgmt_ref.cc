#include "kvcache_mgmt_ref.h"

#include <torch/torch.h>

namespace st::reference::kernel {

using torch::Tensor;

void saveContextStageKVCacheKernelRef(
	Tensor &k_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
	Tensor &v_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
	const Tensor &qkvs,	// [num_tokens, num_q_heads+2*num_kv_heads, head_dim]
	const Tensor &block_table_cpu,	// [num_reqs, max_num_block_per_seq]
	const Tensor &input_len_cpu,	// [num_reqs]
	const Tensor &is_context_stage_cpu,	// [num_reqs]
	const int64_t layer_id
) {
	const int64_t num_reqs = input_len_cpu.size(0);
	const int64_t block_size = k_cache.size(3);
	const int64_t num_kv_heads = k_cache.size(2);
	const int64_t num_q_heads = qkvs.size(1) - 2 * num_kv_heads;

	int64_t first_token_index = 0;
	for (int64_t req_index = 0; req_index < num_reqs; ++req_index) {
		const int64_t input_len = input_len_cpu[req_index].item<int64_t>();
		const bool is_context_stage = is_context_stage_cpu[req_index].item<bool>();

		if (is_context_stage) {
			for (int64_t token_index = 0; token_index < input_len; ++token_index) {
				int64_t block_index = block_table_cpu[req_index][token_index / block_size].item<int64_t>();
				int64_t block_offset = token_index % block_size;
				k_cache[block_index][layer_id].select(1, block_offset) = qkvs[first_token_index + token_index].slice(0, num_q_heads, num_q_heads+num_kv_heads);
				v_cache[block_index][layer_id].select(1, block_offset) = qkvs[first_token_index + token_index].slice(0, num_q_heads+num_kv_heads, num_q_heads+2*num_kv_heads);
			}
		}

		first_token_index += is_context_stage ? input_len : 1;
	}
}

}