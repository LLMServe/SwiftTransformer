#pragma once

#include "util/cublas_wrapper.h"
#include "util/nccl_utils.h"

namespace st::layer {

template<typename T>
void attention(
	T* output,
	T* k_cache,
	T* v_cache,

	const T* input,
	const int64_t* input_len,
	const bool* is_context_stage_cpu,
	const int64_t* block_table,
	const int64_t* d_token_indexes,
	
	int64_t num_context_reqs,
	int64_t num_decoding_reqs,
	const int64_t* ith_context_req_req_index,
	const int32_t* ith_context_req_token_index,
	const int64_t* ith_decoding_req_req_index,
	const int64_t* ith_decoding_req_token_index,
	const int64_t max_context_req_len,
	const int64_t max_decoding_req_len,
	
	const T* qkv_weight_kernel,
	const T* qkv_weight_bias,
	const T* out_weight_kernel,
	const T* out_weight_bias,

	const int64_t batch_size,
	const int64_t num_tokens,
	const int64_t hidden_size,
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const bool perform_rotary_embedding,
	const int64_t layer_id,
	const int64_t max_num_block_per_req,
	const int64_t block_size,

	T* qkv_buf,
	T* attn_out_buf,
	float* context_stage_kernel_m_buf,
	float* context_stage_kernel_l_buf,

	util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
);

}	// namespace st::layer
