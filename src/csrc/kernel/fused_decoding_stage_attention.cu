#include "fused_decoding_stage_attention.h"

#include <cassert>
#include <cstdio>

#include "util/cuda_utils.h"

namespace st::kernel {

#define WARP_SIZE 32

// Tuneable parameters
constexpr int64_t DEFAULT_THREAD_BLOCK_SIZE = 256;

/*
	# fusedDecodingStageAttentionKernel

	## Overview

	This kernel saves K/V cache of the latest token and performs batched & fused decoding stage attention.
	
	Recall how the array of token (the `input` array) looks like in `attention()` in `layer/attention.cc`:
	| Prompt or last token | Prompt or last token | ... | Prompt or last token |

	If a request is in the context stage, then "Prompt or last token" contains (the length of) the prompt
	number of tokens. Otherwise, if a request is in the decoding stage, then it contains the last token.

	This kernel only focuses on requests that are in the latter case, i.e. the decoding stage. It
	takes input tokens, k cache and v cache, calculate softmax(qK^T)V (Here q is a vector of length
	num_heads*head_dim), and store the result in `result`

	## Parameters

	Since not every request is going to be processed, we need to pass an array `ith_decoding_request_index`, 
	which contains the index of the request in the decoding stage. For example, if the input is
	| Context tokens | Context tokens | Decoding token | Decoding token | Context tokens | Decoding token |,
	then `ith_decoding_request_index` should be [2, 3, 5].

	## Algorithm & Implementation Details

	Similar to FlashAttention's but the number of query vectors = 1
*/

template<
	typename T,
	int64_t Q_HEADS_PER_THREAD_BLOCK,
	int64_t HEAD_DIM,
	int64_t BLOCK_SIZE,
	int64_t THREAD_BLOCK_SIZE
> __global__ void fusedDecodingStageAttentionKernel(
	// The output
	T* __restrict__ result,			// [num_tokens, num_q_heads, head_dim]

	// QKVs
	const T* __restrict__ qkvs,	// [num_tokens, num_q_heads+2*num_kv_heads, head_dim]
	T* __restrict__ k_cache_offseted,		// The OFFSETed k_cache.
								// The shape of k_cache is [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
								// This k_cache_offseted is real k_cache + layer_id*num_kv_heads*block_size*head_dim
								// So we does not need another register for storing layer_id
	T* __restrict__ v_cache_offseted,		// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]

	// Other inputs
	const float qk_scale,				// 1/sqrt(head_dim)
	const int64_t* __restrict__ block_table,	// [num_reqs, max_num_block_per_seq]
	const int64_t* __restrict__ input_lens,		// [num_reqs]. Here input_lens DOES NOT INCLUDE the latest token!
	const int64_t* __restrict__ ith_decoding_req_req_index,	// [num_decoding_reqs]
	const int64_t* __restrict__ ith_decoding_req_token_index,	// [num_decoding_reqs]
	const int64_t max_num_block_per_seq,
	const int64_t num_layers,
	const int64_t num_kv_heads
) {
	constexpr int64_t NUM_WARPS = THREAD_BLOCK_SIZE / WARP_SIZE;

	// Grid: num_q_heads/Q_HEADS_PER_THREAD_BLOCK x num_decoding_reqs
	const int64_t num_q_heads = gridDim.x*Q_HEADS_PER_THREAD_BLOCK;
	const int64_t thread_blocks_per_kv_head = (num_q_heads/num_kv_heads)/Q_HEADS_PER_THREAD_BLOCK;
	const int64_t kv_head_index = blockIdx.x/thread_blocks_per_kv_head;

	const int64_t my_q_head_begin = blockIdx.x*Q_HEADS_PER_THREAD_BLOCK;
	const int64_t my_q_head_end = (blockIdx.x+1)*Q_HEADS_PER_THREAD_BLOCK;

	const int64_t warp_id = threadIdx.x / WARP_SIZE;
	const int64_t lane_id = threadIdx.x % WARP_SIZE;

	const int64_t req_index = ith_decoding_req_req_index[blockIdx.y];
	const int64_t token_index = ith_decoding_req_token_index[blockIdx.y];
	const int64_t input_len = input_lens[req_index];	// Here input_lens DOES NOT INCLUDE the latest token!
	const int64_t num_blocks = (input_len+1 + BLOCK_SIZE-1) / BLOCK_SIZE;

	// We organize threads into thread groups. When calculating attn_mat*V, each element
	// in the final result is calculated by one thread group
	constexpr int64_t THREAD_GROUP_SIZE = 8;	// TODO Tune this
	constexpr int64_t NUM_THREAD_GROUPS = THREAD_BLOCK_SIZE / THREAD_GROUP_SIZE;
	const int64_t thread_group_id = threadIdx.x / THREAD_GROUP_SIZE;
	const int64_t thread_group_offset = threadIdx.x % THREAD_GROUP_SIZE;

	// Step 0: Save the KV cache
	if (threadIdx.x < HEAD_DIM && blockIdx.x%thread_blocks_per_kv_head == 0) {
		int64_t kvcache_index = INDEX_5D(
			0, num_layers, num_kv_heads, BLOCK_SIZE, HEAD_DIM,
			block_table[INDEX_2D(0, max_num_block_per_seq, req_index, input_len/BLOCK_SIZE)],
			0, kv_head_index, input_len%BLOCK_SIZE, threadIdx.x
		);
		k_cache_offseted[kvcache_index] = qkvs[INDEX_3D(0, num_q_heads+2*num_kv_heads, HEAD_DIM, token_index, num_q_heads+kv_head_index, threadIdx.x)];
		v_cache_offseted[kvcache_index] = qkvs[INDEX_3D(0, num_q_heads+2*num_kv_heads, HEAD_DIM, token_index, num_q_heads+num_kv_heads+kv_head_index, threadIdx.x)];
	}
	__syncthreads();	// Since we are going to use k_cache and v_cache later

	__shared__ T q_buf[Q_HEADS_PER_THREAD_BLOCK][HEAD_DIM];
	__shared__ T k_block[BLOCK_SIZE][HEAD_DIM];
	__shared__ float attn_score[Q_HEADS_PER_THREAD_BLOCK][BLOCK_SIZE];	// TODO Transpose this to avoid bank conflict.
	__shared__ T v_block[BLOCK_SIZE][HEAD_DIM];
	__shared__ float mi[Q_HEADS_PER_THREAD_BLOCK], li[Q_HEADS_PER_THREAD_BLOCK];
	__shared__ float mult_old_elem[Q_HEADS_PER_THREAD_BLOCK], mult_new_elem[Q_HEADS_PER_THREAD_BLOCK];
	__shared__ float result_buf[Q_HEADS_PER_THREAD_BLOCK][HEAD_DIM];

	// Step 0: Initialize variables
	if (threadIdx.x < Q_HEADS_PER_THREAD_BLOCK) {
		mi[threadIdx.x] = -__FLT_MAX__;
		li[threadIdx.x] = 0;
	}
	#pragma unroll
	for (int64_t i = threadIdx.x; i < Q_HEADS_PER_THREAD_BLOCK*HEAD_DIM; i += blockDim.x) {
		q_buf[i/HEAD_DIM][i%HEAD_DIM] = qkvs[INDEX_3D(
			0, num_q_heads+2*num_kv_heads, HEAD_DIM,
			token_index, my_q_head_begin+i/HEAD_DIM, i%HEAD_DIM
		)];
		result_buf[i/HEAD_DIM][i%HEAD_DIM] = 0.0;
	}
	__syncthreads();

	// Now we iterate over each k/v block, copy them to shared memory, calculate attention score
	// and modify result_buf[]
	for (int64_t kv_block_idx = 0; kv_block_idx < num_blocks; kv_block_idx += 1) {
		int64_t kv_block_index = block_table[req_index*max_num_block_per_seq + kv_block_idx];

		// Step 1. Copy k/v to shared memory
		#pragma unroll
		for (int64_t i = threadIdx.x; i < BLOCK_SIZE*HEAD_DIM; i += blockDim.x) {
			int64_t block_size_index = i/HEAD_DIM;
			int64_t head_dim_index = i%HEAD_DIM;
			bool is_valid = kv_block_idx*BLOCK_SIZE + block_size_index < input_len+1;
			int64_t kvcache_index = INDEX_5D(
				0, num_layers, num_kv_heads, BLOCK_SIZE, HEAD_DIM,
				kv_block_index, 0, kv_head_index, block_size_index, head_dim_index
			);
			k_block[i/HEAD_DIM][i%HEAD_DIM] = is_valid ? k_cache_offseted[kvcache_index] : (T)0.0;
			v_block[i/HEAD_DIM][i%HEAD_DIM] = is_valid ? v_cache_offseted[kvcache_index] : (T)0.0;
		}
		__syncthreads();

		// Step 2. Calculate attention score
		// Each warp calculates some elements in the attention matrix
		// And by the way we scale it by qk_scale
		#pragma unroll
		for (int64_t attn_mat_elem_index = warp_id; attn_mat_elem_index < Q_HEADS_PER_THREAD_BLOCK*BLOCK_SIZE; attn_mat_elem_index += NUM_WARPS) {
			int64_t block_size_index = attn_mat_elem_index / Q_HEADS_PER_THREAD_BLOCK;
			int64_t q_head_index = attn_mat_elem_index % Q_HEADS_PER_THREAD_BLOCK;
			float result = 0.0f;
			#pragma unroll
			for (int64_t hd_index = lane_id; hd_index < HEAD_DIM; hd_index += WARP_SIZE) {
				result += (float)(q_buf[q_head_index][hd_index] * k_block[block_size_index][hd_index]);
			}
			// Reduction within the warp to calculate the result
			#pragma unroll
			for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
				result += __shfl_down_sync(0xffffffff, result, offset);
			}
			if (lane_id == 0) {
				attn_score[q_head_index][block_size_index] = kv_block_idx*BLOCK_SIZE+block_size_index < input_len+1 ? result * qk_scale : -__FLT_MAX__;
			}
		}
		__syncthreads();

		// Step 3. Calculate mij (the maximum value in every row in attn_score[])
		// and lij (the sum of exp(x - mij) in attn_score[])
		#pragma unroll
		for (int64_t q_head_index = warp_id; q_head_index < Q_HEADS_PER_THREAD_BLOCK; q_head_index += NUM_WARPS) {
			if (lane_id < BLOCK_SIZE) {
				const uint32_t shfl_mask = (uint32_t)((1ull<<BLOCK_SIZE)-1);
				float mij = attn_score[q_head_index][lane_id];
				// Reduction to get mij
				#pragma unroll
				for (int offset = BLOCK_SIZE/2; offset > 0; offset /= 2) {
					mij = max(mij, __shfl_xor_sync(shfl_mask, mij, offset));
				}
				float lij = attn_score[q_head_index][lane_id] = __expf(attn_score[q_head_index][lane_id] - mij);
				// Reduction to get lij
				#pragma unroll
				for (int offset = BLOCK_SIZE/2; offset > 0; offset /= 2) {
					lij += __shfl_down_sync(shfl_mask, lij, offset);
				}
				if (lane_id == 0) {
					const float mi_new = fmaxf(mi[q_head_index], mij);
					const float li_new = __expf(mi[q_head_index]-mi_new)*li[q_head_index] + __expf(mij-mi_new)*lij;
					mult_old_elem[q_head_index] = __expf(mi[q_head_index]-mi_new);
					mult_new_elem[q_head_index] = __expf(mij-mi_new);
					mi[q_head_index] = mi_new;
					li[q_head_index] = li_new;
				}
			}
		}
		__syncthreads();

		// Step 4. Calculate attn_score * v_block, and modify result
		#pragma unroll
		for (int64_t i = thread_group_id; i < Q_HEADS_PER_THREAD_BLOCK*HEAD_DIM; i += NUM_THREAD_GROUPS) {
			int64_t q_head_index = i / HEAD_DIM;
			int64_t hd_index = i % HEAD_DIM;
			float cur_result = 0.0f;
			#pragma unroll
			for (int64_t block_size_index = thread_group_offset; block_size_index < BLOCK_SIZE; block_size_index += THREAD_GROUP_SIZE)
				cur_result += (float)((T)attn_score[q_head_index][block_size_index] * v_block[block_size_index][hd_index]);
			// Reduction within the thread group to calculate the result
			#pragma unroll
			for (int offset = THREAD_GROUP_SIZE/2; offset > 0; offset /= 2) {
				cur_result += __shfl_down_sync(0xffffffff, cur_result, offset);
			}
			if (thread_group_offset == 0) {
				result_buf[q_head_index][hd_index] = mult_old_elem[q_head_index]*result_buf[q_head_index][hd_index] + mult_new_elem[q_head_index]*cur_result;
			}
		}
		__syncthreads();
	}

	// Step -1 (the final step): Copy result_buf[] to result[]
	if (warp_id < Q_HEADS_PER_THREAD_BLOCK) {
		#pragma unroll
		for (int64_t hd_index = lane_id; hd_index < HEAD_DIM; hd_index += WARP_SIZE) {
			result[INDEX_3D(
				0, num_q_heads, HEAD_DIM,
				token_index, my_q_head_begin+warp_id, hd_index
			)] = result_buf[warp_id][hd_index] / (li[warp_id]+1e-6);
		}
	}
}

#define LAUNCH_DECODING_STAGE_ATTENTION_KERNEL(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, BLOCK_SIZE) \
	fusedDecodingStageAttentionKernel<T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, BLOCK_SIZE, DEFAULT_THREAD_BLOCK_SIZE><<<grid_dim, DEFAULT_THREAD_BLOCK_SIZE>>>( \
		result, qkvs, k_cache_offseted, v_cache_offseted, scale, block_table, input_lens, ith_decoding_req_req_index, ith_decoding_req_token_index, max_num_block_per_seq, num_layers, num_kv_heads \
	)

#define FUSED_DECODING_STAGE_ATTENTION_DISPATCH_BLOCK_SIZE(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM) \
	switch (block_size) { \
		case 1: LAUNCH_DECODING_STAGE_ATTENTION_KERNEL(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, 1); break; \
		case 2: LAUNCH_DECODING_STAGE_ATTENTION_KERNEL(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, 2); break; \
		case 4: LAUNCH_DECODING_STAGE_ATTENTION_KERNEL(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, 4); break; \
		case 8: LAUNCH_DECODING_STAGE_ATTENTION_KERNEL(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, 8); break; \
		case 16: LAUNCH_DECODING_STAGE_ATTENTION_KERNEL(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, 16); break; \
		case 32: LAUNCH_DECODING_STAGE_ATTENTION_KERNEL(T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, 32); break; \
		default: fprintf(stderr, "Unsupported block_size: %ld\n", block_size); assert(0); \
	}

#define FUSED_DECODING_STAGE_ATTENTION_DISPATCH_HEAD_DIM(T, Q_HEADS_PER_THREAD_BLOCK) \
	switch (head_dim) {	\
		case 64: FUSED_DECODING_STAGE_ATTENTION_DISPATCH_BLOCK_SIZE(T, Q_HEADS_PER_THREAD_BLOCK, 64); break;	\
		case 80: FUSED_DECODING_STAGE_ATTENTION_DISPATCH_BLOCK_SIZE(T, Q_HEADS_PER_THREAD_BLOCK, 80); break;	\
		case 128: FUSED_DECODING_STAGE_ATTENTION_DISPATCH_BLOCK_SIZE(T, Q_HEADS_PER_THREAD_BLOCK, 128); break;	\
		default: fprintf(stderr, "Unsupported head_dim: %ld\n", head_dim); assert(0);			\
	}

#define FUSED_DECODING_STAGE_ATTENTION_DISPATCH_Q_HEADS_PER_THREAD_BLOCK(T) \
	switch (q_heads_per_thread_block) {	\
		case 1: FUSED_DECODING_STAGE_ATTENTION_DISPATCH_HEAD_DIM(T, 1); break;	\
		case 2: FUSED_DECODING_STAGE_ATTENTION_DISPATCH_HEAD_DIM(T, 2); break;	\
		/*case 4: FUSED_DECODING_STAGE_ATTENTION_DISPATCH_HEAD_DIM(T, 4); break;*/	\
		/*case 8: FUSED_DECODING_STAGE_ATTENTION_DISPATCH_HEAD_DIM(T, 8); break;*/	\
		default: fprintf(stderr, "Unsupported q_heads_per_thread_block: %ld\n", q_heads_per_thread_block); assert(0);	\
	}

template<typename T>
void fusedDecodingStageAttention(
	T* __restrict__ result,
	const T* __restrict__ qkvs,
	T* k_cache,
	T* v_cache,
	const float scale,
	const int64_t* __restrict__ block_table,
	const int64_t* __restrict__ input_lens,
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,
	const int64_t* __restrict__ ith_decoding_req_token_index,
	const int64_t max_decoding_req_len,
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
) {
	#ifdef DEBUG
		assert (block_size <= WARP_SIZE);
		assert (DEFAULT_THREAD_BLOCK_SIZE >= head_dim);
	#endif
	if (num_q_heads == num_kv_heads) {
		fusedDecodingStageAttentionMHA(
			result,
			qkvs,
			k_cache,
			v_cache,
			scale,
			block_table,
			input_lens,
			num_decoding_reqs,
			ith_decoding_req_req_index,
			ith_decoding_req_token_index,
			max_decoding_req_len,
			num_layers,
			num_q_heads,
			head_dim,
			layer_id,
			block_size,
			max_num_block_per_seq
		);
		return;
	}
	int64_t q_heads_per_thread_block = 2;	// TODO Tune this
	T* k_cache_offseted = k_cache + layer_id * num_kv_heads * block_size * head_dim;
	T* v_cache_offseted = v_cache + layer_id * num_kv_heads * block_size * head_dim;
	dim3 grid_dim(num_q_heads/q_heads_per_thread_block, num_decoding_reqs);
	FUSED_DECODING_STAGE_ATTENTION_DISPATCH_Q_HEADS_PER_THREAD_BLOCK(T);
}

#define INSTANTIATE_FUSED_DECODING_STAGE_ATTENTION(T) \
	template void fusedDecodingStageAttention( \
		T* __restrict__, \
		const T* __restrict__, \
		T* __restrict__, \
		T* __restrict__, \
		const float, \
		const int64_t* __restrict__, \
		const int64_t* __restrict__, \
		const int64_t, \
		const int64_t* __restrict__, \
		const int64_t* __restrict__, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t \
	);

INSTANTIATE_FUSED_DECODING_STAGE_ATTENTION(float)
INSTANTIATE_FUSED_DECODING_STAGE_ATTENTION(half)

} // namespace st::kernel