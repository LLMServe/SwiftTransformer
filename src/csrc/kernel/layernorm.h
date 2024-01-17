#pragma once

namespace st::kernel {

template<typename T>
void layernorm(
    T* out,
    const T* input,

    const T* weight,
    const T* bias,
    const float epsilon,

	const int64_t num_tokens,
	const int64_t hidden_size,

    T* biased_input = nullptr,
    const T* pre_layernorm_bias = nullptr
);

} // namespace st::kernel