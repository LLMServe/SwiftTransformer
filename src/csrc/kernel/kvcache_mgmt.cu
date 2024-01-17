#include "kvcache_mgmt.h"

#include <cassert>
#include <cstdio>

#include "util/cuda_utils.h"

namespace st::kernel {

#define WARP_SIZE 32

// Tuneable parameters
constexpr int64_t DEFAULT_THREAD_BLOCK_SIZE = 512;

/*
	saveContextStageKVCache

	This kernel takes q/k/vs that are just calculated and stores them into
	k/v_cache.

	## Implementation Details

	The size of the grid is (num_kv_heads, num_context_reqs). In other words,
	each thread block is assigned to a particular request and head.

	Each token ([HEAD_DIM]) is assigned to a particular warp.
*/

template<
	typename T,
	int64_t HEAD_DIM,
	int64_t BLOCK_SIZE,
	int64_t THREAD_BLOCK_SIZE
> __global__ void saveContextStageKVCacheKernel(
	T* __restrict__ k_cache_offseted,	// [num_block, num_layers, num_kv_heads, block_size, head_dim]
										// The OFFSETed k_cache.
										// The shape of k_cache is [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
										// This k_cache_offseted is real k_cache + layer_id*num_kv_heads*block_size*head_dim
										// So we does not need another register for storing layer_id
	T* __restrict__ v_cache_offseted,

	const T* __restrict__ qkvs,					// [num_tokens, num_q_heads+2*num_kv_heads, head_dim]
	const int64_t* __restrict__ block_table,	// [num_reqs, max_num_block_per_seq]

	const int64_t* __restrict__ input_lens,					// [num_reqs]
	const int64_t* __restrict__ ith_context_req_req_index,	// [num_context_reqs]
	const int32_t* __restrict__ ith_context_req_token_index,	// [num_context_reqs]

	const int64_t max_num_block_per_seq,
	const int64_t num_layers,
	const int64_t num_q_heads
) {
	typedef std::conditional_t<std::is_same<T, half>::value, half2, float2> T2;
	constexpr int64_t NUM_WARPS = THREAD_BLOCK_SIZE / WARP_SIZE;

	const int64_t num_kv_heads = gridDim.x;
	const int64_t head_id = blockIdx.x;

	const int64_t req_index = ith_context_req_req_index[blockIdx.y];
	const int64_t first_token_index = ith_context_req_token_index[blockIdx.y];
	const int64_t input_len = input_lens[req_index];

	const int64_t warp_id = threadIdx.x / WARP_SIZE;
	const int64_t lane_id = threadIdx.x % WARP_SIZE;

	for (int64_t token_index = warp_id; token_index < input_len; token_index += NUM_WARPS) {
		int64_t block_index = block_table[INDEX_2D(0, max_num_block_per_seq, req_index, token_index/BLOCK_SIZE)];
		int64_t offset_in_block = token_index % BLOCK_SIZE;
		#pragma unroll
		for (int64_t hd_index = lane_id; hd_index < HEAD_DIM/2; hd_index += WARP_SIZE) {
			// "hd" stands for "head dim"
			int64_t kvcache_index = INDEX_5D(
				0, num_layers, num_kv_heads, BLOCK_SIZE, HEAD_DIM/2,
				block_index, 0, head_id, offset_in_block, hd_index
			);
			((T2*)k_cache_offseted)[kvcache_index] = ((T2*)qkvs)[INDEX_3D(0, num_q_heads+2*num_kv_heads, HEAD_DIM/2, first_token_index+token_index, num_q_heads+head_id, hd_index)];
			((T2*)v_cache_offseted)[kvcache_index] = ((T2*)qkvs)[INDEX_3D(0, num_q_heads+2*num_kv_heads, HEAD_DIM/2, first_token_index+token_index, num_q_heads+num_kv_heads+head_id, hd_index)];
		}
	}
}

#define LAUNCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL(T, HEAD_DIM, BLOCK_SIZE) \
	saveContextStageKVCacheKernel<T, HEAD_DIM, BLOCK_SIZE, DEFAULT_THREAD_BLOCK_SIZE> \
		<<<grid_dim, DEFAULT_THREAD_BLOCK_SIZE>>> \
		(k_cache_offseted, v_cache_offseted, qkvs, block_table, input_lens, \
			ith_context_req_req_index, ith_context_req_token_index, max_num_block_per_seq, num_layers, num_q_heads)

#define DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM_BLOCK_SIZE(T, HEAD_DIM) \
	switch (block_size) { \
		case 1: LAUNCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL(T, HEAD_DIM, 1); break; \
		case 2: LAUNCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL(T, HEAD_DIM, 2); break; \
		case 4: LAUNCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL(T, HEAD_DIM, 4); break; \
		case 8: LAUNCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL(T, HEAD_DIM, 8); break; \
		case 16: LAUNCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL(T, HEAD_DIM, 16); break; \
		case 32: LAUNCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL(T, HEAD_DIM, 32); break; \
		default: fprintf(stderr, "Unsupported block_size: %ld\n", block_size); assert(0); \
	}

#define DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM(T) \
	switch (head_dim) {	\
		case 64: DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM_BLOCK_SIZE(T, 64); break;	\
		case 80: DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM_BLOCK_SIZE(T, 80); break;	\
		case 96: DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM_BLOCK_SIZE(T, 96); break;	\
		case 112: DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM_BLOCK_SIZE(T, 112); break;	\
		case 128: DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM_BLOCK_SIZE(T, 128); break;	\
		default: fprintf(stderr, "Unsupported head_dim: %ld\n", head_dim); assert(0);			\
	}

template<typename T>
void saveContextStageKVCache(
	T* k_cache,
	T* v_cache,

	const T* qkvs,
	const int64_t* block_table,

	const int64_t* input_lens,
	const int64_t num_context_reqs,
	const int64_t* ith_context_req_req_index,
	const int32_t* ith_context_req_token_index,

	const int64_t block_size,
	const int64_t max_num_block_per_seq,
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t layer_id
) {
	T* k_cache_offseted = k_cache + layer_id * num_kv_heads * block_size * head_dim;
	T* v_cache_offseted = v_cache + layer_id * num_kv_heads * block_size * head_dim;
	dim3 grid_dim(num_kv_heads, num_context_reqs);
	DISPATCH_SAVE_CONTEXT_STAGE_KVCACHE_KERNEL_HEAD_DIM(T);
}

#define INSTANTIATE_SAVE_CONTEXT_STAGE_KVCACHE(T) \
	template void saveContextStageKVCache<T>( \
		T* k_cache, \
		T* v_cache, \
		const T* qkvs, \
		const int64_t* block_table, \
		const int64_t* input_lens, \
		const int64_t num_context_reqs, \
		const int64_t* ith_context_req_req_index, \
		const int32_t* ith_context_req_token_index, \
		const int64_t block_size, \
		const int64_t max_num_block_per_seq, \
		const int64_t num_layers, \
		const int64_t num_q_heads, \
		const int64_t num_kv_heads, \
		const int64_t head_dim, \
		const int64_t layer_id \
	);

INSTANTIATE_SAVE_CONTEXT_STAGE_KVCACHE(float);
INSTANTIATE_SAVE_CONTEXT_STAGE_KVCACHE(half);

}	// st::kernel