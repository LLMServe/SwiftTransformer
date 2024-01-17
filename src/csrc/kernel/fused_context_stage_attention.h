#pragma once

#include "util/cuda_utils.h"

namespace st::kernel {

template<typename T>
void fusedContextStageAttention(
	T* __restrict__ result,
	const T* __restrict__ qkvs,
	const float qk_scale,
	const int64_t* __restrict__ input_lens,
	const int64_t num_context_reqs,
	const int64_t* __restrict__ ith_context_req_req_index,
	const int32_t* __restrict__ ith_context_req_token_index,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t num_tokens,
	float* __restrict__ m_buf,
	float* __restrict__ l_buf
);

}	// namespace st::kernel