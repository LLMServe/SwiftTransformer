#include "attention_ref.h"

#include <iostream>
#include <torch/torch.h>

#include "util/cuda_utils.h"

namespace st::reference::kernel {

using torch::Tensor;

// attention_kernel_ref: The reference implementation of attention_kernel, using LibTorch
// Now it only supports num_layers = 1, e.g. the 1st dimention of k_cache and v_cache is 1
void attentionKernelRef(
	Tensor &result,		// [num_tokens, hidden_size]
	Tensor &k_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
	Tensor &v_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]

	const Tensor &qkvs,	// [num_tokens, num_q_heads+2*num_kv_heads, head_dim]
	const float qk_scale,
	const Tensor &block_table_cpu,	// [num_reqs, max_num_block_per_seq]
	const Tensor &input_len_cpu,		// [num_reqs]
	const Tensor &is_context_stage_cpu,

	bool run_context_stage,
	bool run_decoding_stage
) {
	const int64_t num_tokens = qkvs.size(0);
	const int64_t num_layers = k_cache.size(1);
	const int64_t num_kv_heads = k_cache.size(2);
	const int64_t num_q_heads = qkvs.size(1) - 2 * num_kv_heads;
	const int64_t block_size = k_cache.size(3);
	const int64_t head_dim = k_cache.size(4);
	const int64_t num_reqs = input_len_cpu.size(0);
	const int64_t hidden_size = num_q_heads * head_dim;
	const int64_t max_num_block_per_seq = block_table_cpu.size(1);
	const auto dtype = qkvs.dtype();

	// Dimension check
	assert ((k_cache.sizes() == std::vector<int64_t>{k_cache.size(0), num_layers, num_kv_heads, block_size, head_dim}));
	assert ((v_cache.sizes() == std::vector<int64_t>{v_cache.size(0), num_layers, num_kv_heads, block_size, head_dim}));
	assert ((qkvs.sizes() == std::vector<int64_t>{num_tokens, num_q_heads+2*num_kv_heads, head_dim}));
	assert ((block_table_cpu.sizes() == std::vector<int64_t>{num_reqs, max_num_block_per_seq}));
	assert ((input_len_cpu.sizes() == std::vector<int64_t>{num_reqs}));
	assert ((is_context_stage_cpu.sizes() == std::vector<int64_t>{num_reqs}));

	result = torch::zeros({num_tokens, hidden_size}, torch::kCUDA);
	int64_t cur_req_token_index = 0;
	for (int64_t req_index = 0; req_index < num_reqs; ++req_index) {
		const int64_t input_len = input_len_cpu[req_index].item<int64_t>();
		const bool is_context_stage = is_context_stage_cpu[req_index].item<bool>();

		if (is_context_stage) {
			if (run_context_stage) {
				// Retrieve Q, K, and V
				Tensor cur_input = qkvs.slice(0, cur_req_token_index, cur_req_token_index + input_len);	// [input_len, num_q_heads+2*num_kv_heads, head_dim]
				Tensor cur_query = cur_input.slice(1, 0, num_q_heads).transpose(0, 1);	// [num_q_heads, input_len, head_dim]
				Tensor cur_key = cur_input.slice(1, num_q_heads, num_q_heads+num_kv_heads).transpose(0, 1);	// [num_kv_heads, input_len, head_dim]
				Tensor cur_value = cur_input.slice(1, num_q_heads+num_kv_heads, num_q_heads+2*num_kv_heads).transpose(0, 1);	// [num_kv_heads, input_len, head_dim]
				cur_key = cur_key.repeat_interleave(num_q_heads/num_kv_heads, 0);		// [num_q_heads, input_len, head_dim]
				cur_value = cur_value.repeat_interleave(num_q_heads/num_kv_heads, 0);	// [num_q_heads, input_len, head_dim]

				cur_query = cur_query.to(torch::kFloat32);
				cur_key = cur_key.to(torch::kFloat32);
				cur_value = cur_value.to(torch::kFloat32);
				
				// Calculate the attention matrix
				Tensor attn_mat = torch::matmul(cur_query, cur_key.transpose(1, 2));	// [num_heads, input_len, input_len]
				attn_mat = attn_mat * qk_scale;

				// Mask
				Tensor attn_mask = torch::zeros({input_len, input_len}, torch::kInt64);
				for (int64_t i = 0; i < input_len; ++i) {
					for (int64_t j = i+1; j < input_len; ++j) {
						attn_mask.accessor<int64_t, 2>()[i][j] = -10000;
					}
				}
				attn_mat = attn_mat + attn_mask.to(torch::kCUDA).to(attn_mat.scalar_type());

				// Softmax
				attn_mat = torch::softmax(attn_mat, 2);	// [num_heads, input_len, input_len]

				// Calculate the weighted sum
				Tensor cur_result = torch::matmul(attn_mat, cur_value).transpose(0, 1);	// [input_len, num_heads, head_dim]

				cur_result = cur_result.to(dtype);
				result.slice(0, cur_req_token_index, cur_req_token_index + input_len) = cur_result.reshape({input_len, hidden_size});
			}
		} else {
			if (run_decoding_stage) {
				Tensor cur_qkv = qkvs[cur_req_token_index];	// [num_q_heads+2*num_kv_heads, head_dim]
				Tensor cur_query = cur_qkv.slice(0, 0, num_q_heads);	// [num_q_heads, head_dim]
				Tensor cur_key = cur_qkv.slice(0, num_q_heads, num_q_heads+num_kv_heads);	// [num_kv_heads, head_dim]
				Tensor cur_value = cur_qkv.slice(0, num_q_heads+num_kv_heads, num_q_heads+2*num_kv_heads);	// [num_kv_heads, head_dim]

				// Construct k_cache and v_cache
				const int64_t num_blocks = (input_len+1 + block_size - 1) / block_size;
				Tensor cur_k_cache = torch::zeros({num_kv_heads, 0, head_dim}, torch::kCUDA);
				Tensor cur_v_cache = torch::zeros({num_kv_heads, 0, head_dim}, torch::kCUDA);
				for (int64_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
					int64_t block_index = block_table_cpu[req_index][block_idx].item<int64_t>();
					cur_k_cache = torch::cat({cur_k_cache, k_cache[block_index][0]}, 1);
					cur_v_cache = torch::cat({cur_v_cache, v_cache[block_index][0]}, 1);
				}
				// Now k/v_cache: [num_kv_heads, num_blocks*block_size, head_dim]
				cur_k_cache = cur_k_cache.slice(1, 0, input_len+1);
				cur_v_cache = cur_v_cache.slice(1, 0, input_len+1);
				// Now k/v_cache: [num_kv_heads, input_len+1, head_dim]

				// Save the new k/v cache
				cur_k_cache.select(1, input_len) = cur_key;
				cur_v_cache.select(1, input_len) = cur_value;

				cur_k_cache = cur_k_cache.repeat_interleave(num_q_heads/num_kv_heads, 0);
				cur_v_cache = cur_v_cache.repeat_interleave(num_q_heads/num_kv_heads, 0);
				// Now k/v_cache: [num_q_heads, input_len+1, head_dim]

				cur_query = cur_query.to(torch::kFloat32);
				cur_k_cache = cur_k_cache.to(torch::kFloat32);
				cur_v_cache = cur_v_cache.to(torch::kFloat32);
				
				// Calculate the attention matrix (vector)
				Tensor attn_vec = torch::matmul(
					cur_query.unsqueeze(1),	// [num_q_heads, 1, head_dim]
					cur_k_cache.transpose(1, 2)			// [num_q_heads, head_dim, input_len+1]
				);	// [num_heads, 1, input_len+1]

				// Scale & Softmax
				attn_vec = attn_vec * qk_scale;
				// std::cout << attn_vec << std::endl;
				attn_vec = torch::softmax(attn_vec, 2);	// [num_q_heads, 1, input_len+1]

				// Calculate attn_vec * V
				Tensor cur_result = torch::matmul(
					attn_vec,	// [num_q_heads, 1, input_len+1]
					cur_v_cache	// [num_q_heads, input_len+1, head_dim]
				).squeeze(1);	// [num_q_heads, head_dim]

				cur_result = cur_result.to(dtype);
				result[cur_req_token_index] = cur_result.reshape({hidden_size});
			}
		}
		cur_req_token_index += is_context_stage ? input_len : 1;
	}
}

}	// namespace st::reference::kernel