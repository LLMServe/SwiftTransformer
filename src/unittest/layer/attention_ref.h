#pragma once

#include <torch/torch.h>

namespace st::reference::layer {

using torch::Tensor;

void attentionLayerRef(
	Tensor &result,
	Tensor &k_cache,
	Tensor &v_cache,

	const Tensor &input,
	const Tensor &input_len_cpu,
	const Tensor &is_context_stage_cpu,
	const Tensor &block_table_cpu,

	const float qk_scale, 

	const Tensor &qkv_weight_kernel,
	const Tensor &qkv_weight_bias,
	const Tensor &out_weight_kernel,
	const Tensor &out_weight_bias,

	const int64_t layer_id
);

}