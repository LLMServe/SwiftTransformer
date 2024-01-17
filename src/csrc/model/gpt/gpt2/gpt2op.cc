#include <stdexcept>

#include "gpt2op.h"

#include "util/torch_utils.h"

namespace st::model {

Gpt2Op::Gpt2Op(const int64_t vocab_size,
	               const int64_t max_position_embeddings,
	               const int64_t hidden_size,
	               const int64_t num_layers,
	               const int64_t num_heads,
	               const int64_t head_dim,
                   std::string inference_dtype,
                   const int64_t block_size,
                   const int64_t max_num_block_per_req,
                   const std::vector<int64_t> parallel_config
                   ):
    GptOpBase(inference_dtype,
              GptHyperParam::GetGpt2HyperParam(
                  vocab_size,
                  max_position_embeddings,
                  hidden_size,
                  num_layers,
                  num_heads,
                  head_dim,
                  4 * hidden_size
              ),
              GptPagedAttnParam{
                  .block_size = block_size,
                  .max_num_block_per_req = max_num_block_per_req,
              },
              GptParallelismParam(parallel_config)
    ) {
}

} // namespace st::model
