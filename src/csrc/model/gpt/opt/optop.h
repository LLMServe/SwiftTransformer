#pragma once

#include <torch/script.h>

#include "model/gpt/gptop_base.h"

namespace st::model {

// Please refer to gpt_base.h for the design of GPTBase, Gpt, GptOpBase, and XXXop.
class OptOp : public GptOpBase {
public:
    OptOp(const int64_t vocab_size, // GPT Params
	         const int64_t max_position_embeddings,
	         const int64_t hidden_size,
	         const int64_t num_layers,
	         const int64_t num_heads,
	         const int64_t head_dim,
             const std::string inference_dtype,
             const int64_t block_size, // Cache Params
             const int64_t max_num_block_per_req,
             const std::vector<int64_t> parallel_config);
};

} // namespace st::model