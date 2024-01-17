#include "gather_last_tokens.h"

namespace st::kernel {

/*
	gatherLastTokens

	Gather the last token from each request into a new array.

	When we just finished forwardDecoder, the layout of the decoder output contains a bunch of
	tokens, including prompts of requests that are in context stage and the last token of the
	requests that are in decoding stage. For example, assume request 0 and 2 are in context
	stage while request 1, 3, 4 are in decoding stage, then the array looks like:
	| t00 | t01 | t02 | t03 | t12 | t20 | t21 | t22 | t23 | t24 | t34 | t42 |
	Where tij means the j-th token of request i.

	However, only the last token from each request needs to be sampled, so we need to gather
	the last token of each request together. For the above example, the result should be:
	| t03 | t12 | t24 | t34 | t42 |

	That is what this kernel performs.
*/
template<typename T>
__global__ void gatherLastTokensKernel(
	T* __restrict__ result,			// [batch_size, hidden_dim]
	const T* __restrict__ tokens,	// [num_tokens, hidden_dim]
	const int64_t num_tokens,
	const int64_t batch_size,
	const int64_t hidden_dim,
	const int64_t* __restrict__ sum_prev_input_lens
) {
	int64_t token_index = blockIdx.x == batch_size-1 ? num_tokens-1 : sum_prev_input_lens[blockIdx.x+1]-1;
	#pragma unroll 4
	for (int64_t hidden_dim_index = threadIdx.x; hidden_dim_index < hidden_dim; hidden_dim_index += blockDim.x) {
		result[blockIdx.x*hidden_dim + hidden_dim_index] = tokens[token_index*hidden_dim + hidden_dim_index];
	}
}

template<typename T>
void gatherLastTokens(
	T* result,
	const T* tokens,
	const int64_t num_tokens,
	const int64_t batch_size,
	const int64_t hidden_dim,
	const int64_t* sum_prev_input_lens
) {
	gatherLastTokensKernel<<<batch_size, 512>>>(result, tokens, num_tokens, batch_size, hidden_dim, sum_prev_input_lens);
}

#define INSTANTIATE(T) \
	template void gatherLastTokens<T>( \
		T* result, \
		const T* tokens, \
		const int64_t num_tokens, \
		const int64_t batch_size, \
		const int64_t hidden_dim, \
		const int64_t* sum_prev_input_lens \
	);

INSTANTIATE(half)
INSTANTIATE(float)

}	// namespace st::kernel