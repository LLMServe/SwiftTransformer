#include "embedding.h"

namespace st::kernel {

/*
	embedAndPosiEncodeBatched

	Perform batched input embedding & positional encoding.

	This kernel applies embedding and positional encoding on a batch of input sequences.

	Note. If you do not want to perform positional encoding (e.g. when using models that
	adopts rotary embedding), pass a nullptr to embed_positions_weight.
*/

template<typename T, bool DO_POSI_ENCODING>
__global__ void embedAndPosiEncodeBatchedKernel (
	T* __restrict__ result,
	const int64_t* __restrict__ token_ids,
	const int64_t* __restrict__ position_ids,
	const T* __restrict__ embed_tokens_weight,
	const T* __restrict__ embed_positions_weight,
	const int64_t hidden_size
) {
	const int64_t my_token_id = token_ids[blockIdx.x];
	const int64_t my_position_id = DO_POSI_ENCODING ? position_ids[blockIdx.x] : 0;
	#pragma unroll 4
	for (int64_t hidden_size_index = threadIdx.x; hidden_size_index < hidden_size; hidden_size_index += blockDim.x) {
		T cur_result = embed_tokens_weight[my_token_id * hidden_size + hidden_size_index];
		if constexpr (DO_POSI_ENCODING) {
			cur_result += embed_positions_weight[my_position_id * hidden_size + hidden_size_index];
		}
		result[blockIdx.x * hidden_size + hidden_size_index] = cur_result;
	}
}

template<typename T>
void embedAndPosiEncodeBatched(
	T* result,
	const int64_t* token_ids,		// [num_tokens]
	const int64_t* position_ids,	// [num_tokens]
	const T* embed_tokens_weight,	// [vocab_size, hidden_size]
	const T* embed_positions_weight,	// [max_position_embeddings, hidden_size]
	const int64_t num_tokens,
	const int64_t hidden_size
) {
	bool perform_posi_encoding = embed_positions_weight != nullptr;
	if (perform_posi_encoding) {
		embedAndPosiEncodeBatchedKernel<T, true><<<num_tokens, 512>>>(
			result,
			token_ids,
			position_ids,
			embed_tokens_weight,
			embed_positions_weight,
			hidden_size
		);
	} else {
		embedAndPosiEncodeBatchedKernel<T, false><<<num_tokens, 512>>>(
			result,
			token_ids,
			position_ids,
			embed_tokens_weight,
			embed_positions_weight,
			hidden_size
		);
	}
}

#define INSTANTIATE(T) \
	template void embedAndPosiEncodeBatched<T>( \
		T* result, \
		const int64_t* token_ids, \
		const int64_t* position_ids, \
		const T* embed_tokens_weight, \
		const T* embed_positions_weight, \
		const int64_t num_tokens, \
		const int64_t hidden_size \
	);

INSTANTIATE(half)
INSTANTIATE(float)

}	// namespace st::kernel