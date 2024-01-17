#pragma once

#include <inttypes.h>
#include <cassert>

#include "util/cuda_utils.h"

namespace st::kernel {

template<typename T>
void gatherLastTokens(
	T* result,
	const T* tokens,
	const int64_t num_tokens,
	const int64_t batch_size,
	const int64_t hidden_dim,
	const int64_t* sum_prev_input_lens
);

}	// namespace st::kernel