#pragma once

namespace st::kernel {

template<typename T>
void fusedDecodingStageAttention(
	T* __restrict__ result,
	const T* __restrict__ qkvs,
	T* k_cache,
	T* v_cache,
	const float scale,
	const int64_t* __restrict__ block_table,
	const int64_t* __restrict__ input_lens,
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,
	const int64_t* __restrict__ ith_decoding_req_token_index,
	const int64_t max_decoding_req_len,
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
);

template<typename T>
void fusedDecodingStageAttentionMHA(
	T* __restrict__ result,
	const T* __restrict__ qkvs,
	T* k_cache,
	T* v_cache,
	const float scale,
	const int64_t* __restrict__ block_table,
	const int64_t* __restrict__ input_lens,
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,
	const int64_t* __restrict__ ith_decoding_req_token_index,
	const int64_t max_decoding_req_len,
	const int64_t num_layers,
	const int64_t num_heads,
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
);

}	// namespace st::kernel