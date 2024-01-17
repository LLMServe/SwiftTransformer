#include "attention_ref.h"

#include <iostream>
#include <torch/torch.h>

#include "../kernel/attention_ref.h"
#include "../kernel/kvcache_mgmt_ref.h"

namespace st::reference::layer {

using torch::Tensor;

void attentionLayerRef(
	Tensor &result,		// [num_tokens, hidden_size]
	Tensor &k_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
	Tensor &v_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]

	const Tensor &input,	// [num_tokens, hidden_size]
	const Tensor &input_len_cpu,	// [num_reqs]
	const Tensor &is_context_stage_cpu,	// [num_reqs]
	const Tensor &block_table_cpu,	// [num_reqs, max_num_block_per_seq]

	const float qk_scale, 

	const Tensor &qkv_weight_kernel,	// [hidden_size, num_q_heads + 2*num_kv_heads, head_dim]
	const Tensor &qkv_weight_bias,		// [num_q_heads+2*num_kv_heads, head_dim]
	const Tensor &out_weight_kernel,	// [num_q_heads, head_dim, hidden_size]
	const Tensor &out_weight_bias,		// [hidden_size]

	const int64_t layer_id
) {
	const int64_t num_tokens = input.size(0);
	const int64_t num_q_heads = out_weight_kernel.size(0);
	const int64_t num_kv_heads = k_cache.size(2);
	const int64_t head_dim = qkv_weight_kernel.size(2);
	const int64_t hidden_size = qkv_weight_kernel.size(0);

	// Step 1. QKV GEMM
	Tensor qkvs = torch::matmul(input, qkv_weight_kernel.view({hidden_size, (num_q_heads+2*num_kv_heads)*head_dim}));
	qkvs += qkv_weight_bias.view({(num_q_heads+2*num_kv_heads)*head_dim});
	qkvs = qkvs.view({num_tokens, num_q_heads+2*num_kv_heads, head_dim}); // [num_tokens, num_q_heads+2*num_kv_heads, head_dim]

	// Step 2. Attention
	// result: [num_tokens, hidden_size]
	st::reference::kernel::attentionKernelRef(
		result,
		k_cache,
		v_cache,

		qkvs,
		qk_scale,
		block_table_cpu,
		input_len_cpu,
		is_context_stage_cpu,

		true, true
	);

	// Step 3. Save KV Cache
	st::reference::kernel::saveContextStageKVCacheKernelRef(
		k_cache,
		v_cache,
		qkvs,
		block_table_cpu,
		input_len_cpu,
		is_context_stage_cpu,
		layer_id
	);

	// Step 4. Output GEMM
	result = torch::matmul(result, out_weight_kernel.view({hidden_size, hidden_size})) + out_weight_bias;
}

}