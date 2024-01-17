#include "rotary_posi_embedding.h"

namespace st::kernel {

/*
	rotaryPosiEmbeddingBatched

	Perform rotary positional embedding on a batch of tokens.

	## Background

	Rotary positional embedding (RoPE), as proposed in "ROFORMER : ENHANCED
	TRANSFORMER WITH ROTARY POSITION EMBEDDING", is a method of positional
	embedding that encodes the absolute position while incorporates the
	relative position between tokens. Models like LLaMA and LLaMA2 are based
	on RoPE.

	## Introduction

	This kernel takes a bunch of tokens and their absolute positions with in the
	request, and performs RoPE on them.

	## Implementation Details

	We summon a grid of shape (num_tokens), i.e. each thread block is
	responsible for one token. Each thread block has head_dim/2 threads. The 
	i-th thread will deal with the (2i) and (2i+1) th elements in the head_dim
	in every head.

	## Notes

	In practice we perform RoPE on both the query and the key. Note that when
	performing RoPE on the key, we need to pass num_local_kv_heads as num_heads,
	while performing on the query we need to pass num_local_q_heads as num_heads.
*/

template<typename T>
__global__ void rotaryPosiEmbeddingBatchedKernel (
	T* __restrict__ target,		// [num_tokens, target_1st_dim_size, head_dim]. We will only use [num_tokens, :num_heads, head_dim]
	const int64_t* __restrict__ token_indexes,	// [num_tokens]
	const int64_t num_heads,
	const int64_t target_1st_dim_size,
	const int64_t head_dim
) {
	const int64_t rel_pos = token_indexes[blockIdx.x];
	float cur_sin_f, cur_cos_f;
	__sincosf(rel_pos*__powf(10000.0f, -2.0f*threadIdx.x/head_dim), &cur_sin_f, &cur_cos_f);
	const T cur_sin = (T)cur_sin_f, cur_cos = (T)cur_cos_f;

	typedef typename std::conditional<std::is_same<T, float>::value, float2, half2>::type T2;
	for (int64_t head_id = 0; head_id < num_heads; head_id += 1) {
		// Read x1 and x2 in pack
		const T2 x1_x2 = reinterpret_cast<T2*>(target)[INDEX_3D(
			0, target_1st_dim_size, head_dim/2,
			blockIdx.x, head_id, threadIdx.x
		)];
		const T x1 = x1_x2.x, x2 = x1_x2.y;
		const T new_x1 = x1*cur_cos - x2*cur_sin;
		const T new_x2 = x1*cur_sin + x2*cur_cos;
		// Write back
		reinterpret_cast<T2*>(target)[INDEX_3D(
			0, target_1st_dim_size, head_dim/2,
			blockIdx.x, head_id, threadIdx.x
		)] = T2{new_x1, new_x2};
	}
}

template<typename T>
void rotaryPosiEmbeddingBatched(
	T* __restrict__ target,
	const int64_t* __restrict__ token_indices,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim
) {
	rotaryPosiEmbeddingBatchedKernel<T><<<num_tokens, head_dim/2>>>(
		target, token_indices, num_heads, target_1st_dim_size, head_dim
	);
}

#define INTANTIATE(T) \
	template void rotaryPosiEmbeddingBatched<T>( \
		T* __restrict__, \
		const int64_t* __restrict__, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t \
	);

INTANTIATE(half)
INTANTIATE(float)

}	// namespace st::kernel
