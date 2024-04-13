#include "attention.h"

#include <cassert>

#include "kernel/addbias.h"
#include "kernel/rotary_posi_embedding.h"
#include "kernel/softmax.h"
#include "kernel/unfused_attention.h"
#include "kernel/fused_context_stage_attention.h"
#include "kernel/fused_decoding_stage_attention.h"
#include "kernel/kvcache_mgmt.h"
#include "kernel/xformers_attention.h"
#include "util/cublas_wrapper.h"
#include "util/cuda_utils.h"

namespace st::layer {

// attention - The attention layer
//
// This function is called in the context stage and decoder stage
//
// This incorporates the following optimizations:
// - Selective batching from Orca
// - Paged attention from VLLM
// - FlashAttention
// - Tensor parallelism
//
// For the architecture of the attention layer, please refer to the comments inside this function:

// Tensor Parallelism
// Main parallelism: num_q_heads and num_kv_heads will be split into multiple GPUs
// local_q_head_num = num_q_heads / nccl_comm.size
// local_kv_head_num = num_kv_heads / nccl_comm.size

template<typename T>
void attention(
	T* output,							// [num_tokens, hidden_size]
	T* k_cache,							// [num_blocks, num_layers, local_kv_head_num, block_size, head_dim]
	T* v_cache,							// [num_blocks, num_layers, local_kv_head_num, block_size, head_dim]

	const T* input,						// [num_tokens, hidden_size]
	const int64_t* input_len,				// [batch_size], gpu. The length of the i-th input. For context stage it is the number of tokens in user's request, while for decoder stage it is the number of previous tokens 
	const bool* is_context_stage_cpu,	// [batch_size]. Whether the i-th input is in context stage or decoder stage
	const int64_t* block_table,				// [batch_size, max_num_block_per_req], gpu. The block table of the i-th input. For context stage it is the block table of user's request, while for decoder stage it is the block table of previous tokens
	const int64_t* d_token_indexes,		// [num_tokens], gpu. The index of every token within its request.

	int64_t num_context_reqs,
	int64_t num_decoding_reqs,
	const int64_t* ith_context_req_req_index,		// [num_context_reqs]
	const int32_t* ith_context_req_token_index,		// [num_context_reqs]
	const int64_t* ith_decoding_req_req_index,		// [num_decoding_reqs]
	const int64_t* ith_decoding_req_token_index,	// [num_decoding_reqs]
	const int64_t max_context_req_len,
	const int64_t max_decoding_req_len,

	const T* qkv_weight_kernel,		// [hidden_size, local_q_head_num+2*local_kv_head_num, head_dim]
	const T* qkv_weight_bias,		// [local_q_head_num+2*local_kv_head_num, head_dim]
	const T* out_weight_kernel,		// [local_q_head_num, head_dim, hidden_size]
	const T* out_weight_bias,		// [hidden_size]

	const int64_t batch_size,
	const int64_t num_tokens,			// The number of input tokens. Equal to \sum_{i=0}^{batch_size-1} is_context_stage[i] ? input_len[i] : 1
	const int64_t hidden_size,
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const bool perform_rotary_embedding,

	const int64_t layer_id,
	const int64_t max_num_block_per_req,
	const int64_t block_size,

	T* qkv_buf,		// [num_tokens, local_q_head_num+2*local_kv_head_num, head_dim]
	T* attn_out_buf,// [num_tokens, local_q_head_num, head_dim]
	float* context_stage_kernel_m_buf,	// [local_q_head_num, num_tokens]
	float* context_stage_kernel_l_buf,	// [local_q_head_num, num_tokens]

	util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
) {
	assert(num_q_heads*head_dim == hidden_size);
	assert(nccl_comm.size != 0);
	if (num_q_heads % nccl_comm.size != 0 || num_kv_heads % nccl_comm.size != 0) {
		throw std::runtime_error("num_q_heads or num_kv_heads is not divisible by gpu number, unimpemented");
		exit(1);
	}

	const int64_t local_q_head_num = num_q_heads / nccl_comm.size;
	const int64_t local_kv_head_num = num_kv_heads / nccl_comm.size;
	const float qk_scale = 1.0f / sqrtf(head_dim * 1.0f);

	// Step1. QKV Gemm
	// This calculates the Q, K and V matrix
	// The result is stored in qkv_buf_ ([num_tokens, local_q_head_num+2*local_kv_head_num, head_dim])
	// 
	// Input:
	//	- input: [num_tokens, hidden_size]
	//	- qkv_weight_kernel: [hidden_size, local_q_head_num+2*local_kv_head_num, head_dim]
	// Output:
	//	- qkv_buf: [num_tokens, local_q_head_num+2*local_kv_head_num, head_dim]
	cublas_wrapper.gemm(
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		num_tokens,
		(local_q_head_num+2*local_kv_head_num) * head_dim,
		hidden_size,
		input,
		qkv_weight_kernel,
		qkv_buf
	);
	sync_check_cuda_error();

	// Step 2: Add bias to qkv_buf
	// 
	// Input:
	// 	- qkv_buf: [num_tokens, local_q_head_num+2*local_kv_head_num, head_dim]
	// 	- qkv_weight_bias: [local_q_head_num+2*local_kv_head_num, head_dim]
	// Output:
	// 	- qkv_buf: [num_tokens, local_q_head_num+2*local_kv_head_num, head_dim]

	if (qkv_weight_bias != nullptr) {
		kernel::addbiasBatched(
			qkv_buf,
			qkv_buf,
			qkv_weight_bias,
			num_tokens,
			(local_q_head_num+2*local_kv_head_num) * head_dim
		);
	}
	sync_check_cuda_error();

	if (perform_rotary_embedding) {
		kernel::rotaryPosiEmbeddingBatched(
			qkv_buf,
			d_token_indexes,
			num_tokens,
			local_q_head_num + 2*local_kv_head_num,
			local_q_head_num,
			head_dim
		);
		sync_check_cuda_error();

		kernel::rotaryPosiEmbeddingBatched(
			qkv_buf + local_q_head_num*head_dim,
			d_token_indexes,
			num_tokens,
			local_q_head_num + 2*local_kv_head_num,
			local_kv_head_num,
			head_dim
		);
		sync_check_cuda_error();
	}
	
	// Step3: fusdAttention， get softmax(qk)V
	// attn_out_buf: [num_tokens, local_q_head_num, head_dim]

	if (num_context_reqs != 0) {
		kernel::saveContextStageKVCache<T>(
			k_cache,
			v_cache,

			qkv_buf,
			block_table,
			input_len,
			num_context_reqs,
			ith_context_req_req_index,
			ith_context_req_token_index,
			block_size,
			max_num_block_per_req,
			num_layers,
			local_q_head_num,
			local_kv_head_num,
			head_dim,
			layer_id
		);
		sync_check_cuda_error();

		if (local_q_head_num == local_kv_head_num && std::is_same_v<T, half>) {
			// Use xformers' attention kernel when GQA (group query attention) is disabled
			kernel::xformersContextStageAttention<T>(
				attn_out_buf,
				qkv_buf,
				qk_scale,
				input_len,
				num_context_reqs,
				ith_context_req_req_index,
				ith_context_req_token_index,
				local_q_head_num,
				local_kv_head_num,
				head_dim,
				num_tokens,
				max_context_req_len
			);
			sync_check_cuda_error();
		} else {
			// If GQA is enabled we use our own context stage attention kernel
			kernel::fusedContextStageAttention<T>(
				attn_out_buf,

				qkv_buf,
				qk_scale,

				input_len,
				num_context_reqs,
				ith_context_req_req_index,
				ith_context_req_token_index,

				local_q_head_num,
				local_kv_head_num,
				head_dim,
				num_tokens,
				context_stage_kernel_m_buf,
				context_stage_kernel_l_buf
			);
			sync_check_cuda_error();
		}
	}


	if (num_decoding_reqs != 0) {
		kernel::fusedDecodingStageAttention(
			attn_out_buf,

			qkv_buf,
			k_cache,
			v_cache,
			qk_scale,

			block_table,
			input_len,
			num_decoding_reqs,
			ith_decoding_req_req_index,
			ith_decoding_req_token_index,
			max_decoding_req_len,

			num_layers,
			local_q_head_num,
			local_kv_head_num,
			head_dim,
			layer_id,
			block_size,
			max_num_block_per_req
		);
		sync_check_cuda_error();
	}

	// The last step: Output GEMM
	//
	// Input:
	// 	- attn_out_buf: [num_tokens, local_q_head_num, head_dim]
	// 	- out_weight_kernel: [local_q_head_num, head_dim, hidden_size]
	// Output:
	// 	- output_buf: [num_tokens, hidden_size]
	cublas_wrapper.gemm(
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		num_tokens,
		hidden_size,
		head_dim*local_q_head_num,
		attn_out_buf,
		out_weight_kernel,
		output
	);
	sync_check_cuda_error();

	if (nccl_comm.size != 1) {
		st::util::stNcclAllReduce(
			output,
			output,
			num_tokens * hidden_size,
			st::util::stNcclGetDataType<T>(),
			ncclSum,
			nccl_comm.comm,
			nccl_comm.stream
		);
	}

	if (out_weight_bias != nullptr) {
		kernel::addbiasBatched(
			output,
			output,
			out_weight_bias,
			num_tokens,
			hidden_size
		);
		sync_check_cuda_error();
	}

}

#define INSTANTIATE_ATTENTION(T) \
template void attention( \
	T* output, T* k_cache, T* v_cache, \
	const T* input, const int64_t* input_len, const bool* is_context_stage_cpu, const int64_t* block_table, const int64_t* d_token_indexes, \
	int64_t num_context_reqs, int64_t num_decoding_reqs, \
	const int64_t* ith_context_req_req_index, const int32_t* ith_context_req_token_index, const int64_t* ith_decoding_req_req_index, const int64_t* ith_decoding_req_token_index, const int64_t max_context_req_len, const int64_t max_decoding_req_len, \
	const T* qkv_weight_kernel, const T* qkv_weight_bias, const T* out_weight_kernel, const T* out_weight_bias, \
	const int64_t batch_size, const int64_t num_tokens, const int64_t hidden_size, const int64_t num_layers, const int64_t num_q_heads, const int64_t num_kv_heads, const int64_t head_dim, const bool perform_rotary_embedding, const int64_t layer_id, \
	const int64_t max_num_block_per_req, const int64_t block_size, \
	T* qkv_buf, T* attn_out_buf, \
	float* context_stage_kernel_m_buf, float* context_stage_kernel_l_buf, \
	util::CublasWrapper cublas_wrapper, \
	util::NcclComm nccl_comm \
);

INSTANTIATE_ATTENTION(float)
INSTANTIATE_ATTENTION(half)

}	// namespace st::layer
