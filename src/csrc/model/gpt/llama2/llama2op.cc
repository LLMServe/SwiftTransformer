#include <stdexcept>

#include "llama2op.h"

#include "util/torch_utils.h"

namespace st::model {

Llama2Op::Llama2Op(const int64_t vocab_size,
	               const int64_t max_position_embeddings,
	               const int64_t hidden_size,
	               const int64_t num_layers,
	               const int64_t num_q_heads,
                   const int64_t num_kv_heads,
	               const int64_t head_dim,
                   const int64_t ffn_inter_dim,
                   std::string inference_dtype,
                   const int64_t block_size,
                   const int64_t max_num_block_per_req,
                   const std::vector<int64_t> parallel_config):
    GptOpBase(inference_dtype,
        GptHyperParam::GetLlama2HyperParam(
            vocab_size,
            max_position_embeddings,
            hidden_size,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            ffn_inter_dim
        ),
        GptPagedAttnParam{
            .block_size = block_size,
            .max_num_block_per_req = max_num_block_per_req,
        },
        GptParallelismParam(parallel_config)
    ) {
};

} // namespace st::model
