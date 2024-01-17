#pragma once

namespace st::kernel {

template<typename T>
void scaleMaskSoftmax(
	T* output,
	const T* input,
	const float scale,
	const int64_t num_heads,
	const int64_t input_len
);

template<typename T>
void scaleSoftmax(
	T* output,
	const T* input,
	const float scale,
	const int64_t num_heads,
	const int64_t seq_len
);

}	// namespace st::kernel