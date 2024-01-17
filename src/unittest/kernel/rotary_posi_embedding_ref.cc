#include "rotary_posi_embedding_ref.h"

#include <iostream>

namespace st::reference::kernel {

using torch::Tensor;

void rotaryPosiEmbeddingKernelRef(
	Tensor &target,		// [num_tokens, num_heads, head_dim]
	const std::vector<int64_t> &indexes	// [num_tokens]
) {
	const int64_t NUM_TOKENS = target.size(0);
	const int64_t NUM_HEADS = target.size(1);
	const int64_t HEAD_DIM = target.size(2);

	// To guarantee precision, we use float32 instead of half here.
	torch::Tensor angles = torch::arange(0, HEAD_DIM/2, 1, torch::kCUDA).to(torch::kFloat32).mul(-2).div(HEAD_DIM);
	torch::Tensor thetas = torch::pow(10000, angles);	// thetas[i] = 10000 ** (-2i/d), float32

	// Deal each token one by one
	for (int64_t token_index = 0; token_index < NUM_TOKENS; ++token_index) {
		int64_t index = indexes[token_index];
		torch::Tensor coses = torch::cos(thetas.mul(index));	// coses[i] = cos(10000 ** (-2i/d) * index)
		torch::Tensor sins = torch::sin(thetas.mul(index));	// sins[i] = sin(10000 ** (-2i/d) * index)
		coses = coses.unsqueeze(0).repeat_interleave(NUM_HEADS, 0);	// [num_heads, head_dim/2]
		sins = sins.unsqueeze(0).repeat_interleave(NUM_HEADS, 0);		// [num_heads, head_dim/2]

		torch::Tensor cur_token = target[token_index].to(torch::kFloat32);	// [num_heads, head_dim]
		torch::Tensor cur_token_even = cur_token.index({torch::indexing::Slice(), torch::indexing::Slice(0, HEAD_DIM, 2)});	// [num_heads, head_dim/2]
		torch::Tensor cur_token_odd = cur_token.index({torch::indexing::Slice(), torch::indexing::Slice(1, HEAD_DIM, 2)});	// [num_heads, head_dim/2]

		torch::Tensor cur_token_even_rot = cur_token_even.mul(coses) - cur_token_odd.mul(sins);	// [num_heads, head_dim/2]
		torch::Tensor cur_token_odd_rot = cur_token_even.mul(sins) + cur_token_odd.mul(coses);	// [num_heads, head_dim/2]

		target[token_index].index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, HEAD_DIM, 2)}, cur_token_even_rot.to(torch::kFloat16));
		target[token_index].index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, HEAD_DIM, 2)}, cur_token_odd_rot.to(torch::kFloat16));
		// Thanks to CoPilot, otherwise I would never be able to write the code above...
	}
}

}	// namespace st::reference::kernel