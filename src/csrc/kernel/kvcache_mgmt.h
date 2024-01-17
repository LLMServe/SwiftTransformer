#pragma once

namespace st::kernel {

template<typename T>
void saveContextStageKVCache(
	T* k_cache,
	T* v_cache,

	const T* qkvs,
	const int64_t* block_table,

	const int64_t* input_lens,
	const int64_t num_context_reqs,
	const int64_t* ith_context_req_req_index,
	const int32_t* ith_context_req_token_index,

	const int64_t block_size,
	const int64_t max_num_block_per_seq,
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t layer_id
);

}	// namespace st::kernel