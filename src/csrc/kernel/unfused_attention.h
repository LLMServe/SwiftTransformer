#pragma once

namespace st::kernel {

template<typename T>
void transposeQKV(
	T* q_buf,
	T* k_buf,
	T* v_buf,
	const T* QKV,
	const int64_t cur_input_start,
	const int64_t input_len,
	const int64_t num_heads,
	const int64_t head_dim
);


template<typename T>
void mergeOutput(
	T* output,
	const T* cur_input,
	const int64_t cur_input_len,
	const int64_t cur_input_start,
	const int64_t num_heads,
	const int64_t head_dim
);

}	// namespace st::kernel
