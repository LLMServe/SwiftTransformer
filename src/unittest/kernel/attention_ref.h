#pragma once

#include <torch/torch.h>

namespace st::reference::kernel {

using torch::Tensor;

void attentionKernelRef(
	Tensor &result,		// [num_tokens, hidden_size]
	Tensor &k_cache,	// [num_blocks, num_layers, num_heads, block_size, head_dim]
	Tensor &v_cache,	// [num_blocks, num_layers, num_heads, block_size, head_dim]

	const Tensor &qkvs,	// [num_tokens, 3, num_heads, head_dim]
	const float qk_scale,
	const Tensor &block_table_cpu,	// [num_reqs, max_num_block_per_seq]
	const Tensor &input_len_cpu,		// [num_reqs]
	const Tensor &is_context_stage_cpu,

	bool run_context_stage,
	bool run_decoding_stage
);

}	// namespace st::reference::kernel
