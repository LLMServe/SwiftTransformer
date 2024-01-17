#pragma once

#include <torch/torch.h>

namespace st::reference::kernel {

using torch::Tensor;

void saveContextStageKVCacheKernelRef(
	Tensor &k_cache,
	Tensor &v_cache,
	const Tensor &qkvs,
	const Tensor &block_table_cpu,
	const Tensor &input_len_cpu,
	const Tensor &is_context_stage_cpu,
	const int64_t layer_id
);

}	// namespace st::reference::kernel
