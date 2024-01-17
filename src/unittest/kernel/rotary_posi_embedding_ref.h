#pragma once

#include <vector>
#include <torch/torch.h>

namespace st::reference::kernel {

using torch::Tensor;

void rotaryPosiEmbeddingKernelRef(
	Tensor &target,		// [num_tokens, num_heads, head_dim]
	const std::vector<int64_t> &indexes	// [num_tokens]
);

}	// namespace st::reference::kernel
