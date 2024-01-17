#pragma once

#include <inttypes.h>

namespace st::kernel {

template<typename T>
void rmsnorm(
    T* out,
    const T* input,

    const T* weight,
    const float epsilon,

	const int64_t num_tokens,
	const int64_t hidden_size
);

} // namespace st::kernel