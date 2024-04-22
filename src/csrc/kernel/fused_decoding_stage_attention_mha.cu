#include "fused_decoding_stage_attention.h"

#include <cassert>
#include <cstdio>

#include "util/cuda_utils.h"
#include "util/debug_utils.h"
#include "reduction.cuh"

namespace st::kernel {

#define WARP_SIZE 32

// Tuneable parameters
constexpr int64_t DEFAULT_THREAD_BLOCK_SIZE = 256;

/*
	fusedDecodingStageAttentionMHAKernel - The kernel for fused decoding stage attention without GQA

	Generally speaking, we can set num_q_heads == num_kv_heads and use the same kernel 
	(fusedDecodingStageAttentionGQAKernel) for both MHA models and GQA models. However,
	when num_q_heads == num_kv_heads, we can use some more aggressive optimizations
	which may not be applicable when num_q_heads != num_kv_heads. So we use this kernel
	(fusedDecodingStageAttentionMHAKernel) for MHA models and use the other kernel
	(fusedDecodingStageAttentionGQAKernel) for GQA models.

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

	In PagedAttention, we divide the k/v cache into blocks, where each block contains keys/values of `BLOCK_SIZE`
	tokens. We can represent k_cache as K^T = [K0, K1, ..., Kr], and V = [V0, V1, ..., Vr]^T.

	The grid size should be num_heads x num_decoding_reqs, i.e. each thread block is responsible
	for one request and one head.

	How do we calculate softmax(qK^T)? Well, softmax(qK^T) = softmax([qK0, qK1, ..., qKr]). Every
	block (Ki) is handled by a warp (a warp may handle multiple blocks). The warp is responsible
	for calculating qKi. After all warps finish their work, we need to do a reduction to get
	max(qK^T) and the sum of exp(qK^T-max(qK^T)) (required by softmax), then we can calculate softmax(qK^T).	

	A warp contains 32 threads, and they are divided into BLOCK_SIZE number of groups. Each group is responsible
	for calculating qki (here ki means a column in Kj). Each thread in a group is responsible for
	calculating some elements in qki (I call it qkij below), and then we can use a reduction
	to get the final result (qki).

	Next we need to calculate attn_score * V. We can represent V as [V0, V1, ..., Vr]^T, and a = attn_score = 
	[a0, a1, ..., ar] (each ai has a shape of 1 x BLOCK_SIZE), so aV = a0V0 + a1V1 + ... + arVr (each aiVi has
	a shape of 1 x HEAD_DIM).

	Similar to the softmax(qK^T) case, each block is handled by a warp. So each warp is responsible for
	calculating some aiVi. Each thread calculates ai * (some rows in Vi). After all warps finish their work,
	we use a reduction to get the final result.

	## Performance Tuning

	- Since each block is handled by a warp, the number of blocks should be large enough
	  (>= DEFAULT_THREAD_BLOCK_SIZE/WARP_SIZE) to fully utilize the GPU.
*/

template<
	typename T,
	int64_t HEAD_DIM,
	int64_t BLOCK_SIZE,
	int64_t THREAD_BLOCK_SIZE
> __global__ void fusedDecodingStageAttentionMHAKernel(
	// The output
	T* __restrict__ result,			// [num_tokens, num_heads, head_dim]

	// QKVs
	const T* __restrict__ qkvs,	// [num_tokens, 3, num_heads, head_dim]
	T* __restrict__ k_cache_offseted,		// The OFFSETed k_cache.
								// The shape of k_cache is [num_blocks, num_layers, num_heads, block_size, head_dim]
								// This k_cache_offseted is real k_cache + layer_id*num_heads*block_size*head_dim
								// So we does not need another register for storing layer_id
	T* __restrict__ v_cache_offseted,		// [num_blocks, num_layers, num_heads, block_size, head_dim]

	// Other inputs
	const float qk_scale,				// 1/sqrt(head_dim)
	const int64_t* __restrict__ block_table,	// [num_reqs, max_num_block_per_seq]
	const int64_t* __restrict__ input_lens,		// [num_reqs]. Here input_lens DOES NOT INCLUDE the latest token!
	const int64_t* __restrict__ ith_decoding_req_req_index,	// [num_decoding_reqs]
	const int64_t* __restrict__ ith_decoding_req_token_index,	// [num_decoding_reqs]
	const int64_t max_num_block_per_seq,
	const int64_t num_layers
) {
	constexpr int64_t NUM_THREAD_PER_KEY = WARP_SIZE / BLOCK_SIZE;	// The size of the thread group
	constexpr int64_t THREAD_GROUP_SIZE = NUM_THREAD_PER_KEY;		// Just an alias
	constexpr int64_t NUM_WARPS = THREAD_BLOCK_SIZE / WARP_SIZE;
	constexpr int64_t NUM_ELEM_PER_THREAD = (HEAD_DIM/2) / NUM_THREAD_PER_KEY;
	typedef std::conditional_t<std::is_same<T, half>::value, half2, float2> T2;

	const int64_t head_id = blockIdx.x;		// Grid: num_heads x num_decoding_reqs
	const int64_t num_heads = gridDim.x;	// TODO Pass it as a template parameter to save a register

	const int64_t req_index = ith_decoding_req_req_index[blockIdx.y];
	const int64_t token_index = ith_decoding_req_token_index[blockIdx.y];
	const int64_t input_len = input_lens[req_index];	// Here input_lens DOES NOT INCLUDE the latest token!
	const int64_t num_blocks = (input_len+1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

	const int64_t warp_id = threadIdx.x / WARP_SIZE;	// Which warp we are in
	const int64_t lane_id = threadIdx.x % WARP_SIZE;
	const int64_t thread_group_id = lane_id / NUM_THREAD_PER_KEY;	// Which thread group we are in, i.e. which column in K^T we are responsible for
	const int64_t thread_id_in_group = lane_id % NUM_THREAD_PER_KEY;	// Which thread we are in the thread group

	extern __shared__ float shared_mem[];
	float* attn_score = shared_mem; // [\lceil max_input_len/BLOCK_SIZE \rceil * BLOCK_SIZE]
	T2* qkv_reduction_wksp = (T2*)shared_mem;	// [NUM_WARPS, WARP_SIZE]
	__shared__ float reduction_wksp[32];	// Workspace for reduction. Here 32 >= NUM_WARPS

	// Step 0: Save the KV cache
	if (threadIdx.x < HEAD_DIM) {
		int64_t kvcache_index = INDEX_5D(
			0, num_layers, num_heads, BLOCK_SIZE, HEAD_DIM,
			block_table[INDEX_2D(0, max_num_block_per_seq, req_index, input_len/BLOCK_SIZE)],
			0, head_id, input_len%BLOCK_SIZE, threadIdx.x
		);
		k_cache_offseted[kvcache_index] = qkvs[INDEX_4D(0, 3, num_heads, HEAD_DIM, token_index, 1, head_id, threadIdx.x)];
		v_cache_offseted[kvcache_index] = qkvs[INDEX_4D(0, 3, num_heads, HEAD_DIM, token_index, 2, head_id, threadIdx.x)];
	}
	__syncthreads();	// Since we are going to use k_cache and v_cache later

	// Step 1: Load q into registers
	// 
	// We do this since we must multiply q with every column in K^T, and we can save a lot of
	// global memory access by doing this.
	// 
	// To leverage the memory coalescing, the i-th thread in the thread group
	// is responsible for q[i], q[i+THREAD_GROUP_SIZE], and so on.
	T2 q_cache[NUM_ELEM_PER_THREAD];
	#pragma unroll
	for (int64_t i = 0; i < NUM_ELEM_PER_THREAD; i++) {
		q_cache[i] = ((const T2 *)qkvs)[INDEX_4D(0, 3, num_heads, HEAD_DIM/2, token_index, 0, head_id, thread_id_in_group + i*THREAD_GROUP_SIZE)];
	}

	// Variables for softmax-ing
	float max_qki = -__FLT_MAX__;

	// Iterate over all blocks
	for (int64_t block_idx = warp_id; block_idx < num_blocks; block_idx += NUM_WARPS) {
		const int64_t block_index = block_table[req_index*max_num_block_per_seq + block_idx];
		const T2* k_block = (const T2*)(k_cache_offseted + (block_index*num_layers*num_heads + head_id)*BLOCK_SIZE*HEAD_DIM);
		const int64_t token_idx = block_idx*BLOCK_SIZE + thread_group_id;

		// Step 2: Calculate qkij
		float qkij = 0;
		#pragma unroll
		for (int64_t i = 0; i < NUM_ELEM_PER_THREAD; ++i) {
			const T2 q_elem = q_cache[i];
			const T2 k_elem = k_block[INDEX_2D(0, HEAD_DIM/2, thread_group_id, thread_id_in_group + i*NUM_THREAD_PER_KEY)];
			qkij += (float)(q_elem.x * k_elem.x + q_elem.y * k_elem.y);
		}

		// Step 3: Reduce qkij to get qki
		float qki = qkij;
		#pragma unroll
		for (int64_t mask = THREAD_GROUP_SIZE/2; mask; mask >>= 1) {
			qki += __shfl_xor_sync(0xffffffff, qki, mask);
		}
		// Now all threads with thread_id_in_group == 0 has the correct value of qki
		if (thread_id_in_group == 0) {
			qki = token_idx <= input_len ? qki*qk_scale : -__FLT_MAX__;
			max_qki = fmaxf(max_qki, qki);
			attn_score[token_idx] = qki;
		}
	}

	// Step 4: Perform reduction on max_qki within each warp
	#pragma unroll
	for (int mask = WARP_SIZE/2; mask >= THREAD_GROUP_SIZE; mask >>= 1) {
		max_qki = fmaxf(max_qki, __shfl_xor_sync(0xffffffff, max_qki, mask));
	}
	// Now all threads with lane == 0 has max_qki = max(qki | i is in the same warp)
	if (lane_id == 0) {
		reduction_wksp[warp_id] = max_qki;
	}
	__syncthreads();

	// Step 5: Perform reduction on max_qki within the whole thread group
	if (warp_id == 0) {
		max_qki = lane_id < NUM_WARPS ? reduction_wksp[lane_id] : -__FLT_MAX__;
		#pragma unroll
		for (int mask = NUM_WARPS/2; mask; mask >>= 1) {
			max_qki = fmaxf(max_qki, __shfl_xor_sync(0xffffffff, max_qki, mask));
		}
		// Now thread #0 has the correct max_qki
		if (lane_id == 0) {
			reduction_wksp[0] = max_qki;
		}
	}
	__syncthreads();

	// Step 6: Get the sum of exp(qki - max_qki)
	max_qki = reduction_wksp[0];
	float sum_exp_qki = 0;
	#pragma unroll
	for (int i = threadIdx.x; i < num_blocks*BLOCK_SIZE; i += THREAD_BLOCK_SIZE) {
		float val = __expf(attn_score[i] - max_qki);
		sum_exp_qki += val;
		attn_score[i] = val;
	}
	__syncthreads();

	// Perform reduction within warp
	#pragma unroll
	for (int mask = WARP_SIZE/2; mask; mask >>= 1) {
		sum_exp_qki += __shfl_xor_sync(0xffffffff, sum_exp_qki, mask);
	}
	if (lane_id == 0) {
		reduction_wksp[warp_id] = sum_exp_qki;
	}
	__syncthreads();

	// Perform reduction within thread group
	if (warp_id == 0) {
		sum_exp_qki = lane_id < NUM_WARPS ? reduction_wksp[lane_id] : 0;
		#pragma unroll
		for (int mask = NUM_WARPS/2; mask; mask >>= 1) {
			sum_exp_qki += __shfl_xor_sync(0xffffffff, sum_exp_qki, mask);
		}
		if (lane_id == 0) {
			reduction_wksp[0] = 1.0f / (sum_exp_qki + 1e-6f);
		}
	}
	__syncthreads();

	// Step 7: Calculate softmax
	float softmax_denorm = reduction_wksp[0];
	#pragma unroll
	for (int token_index = threadIdx.x; token_index <= input_len; token_index += THREAD_BLOCK_SIZE) {
		attn_score[token_index] *= softmax_denorm;
	}
	__syncthreads();

	// Step 8: calculate attn_score * V
	constexpr int64_t NUM_COL_PER_THREAD = (HEAD_DIM/2 + WARP_SIZE - 1) / WARP_SIZE;
	T2 acc[NUM_COL_PER_THREAD];
	#pragma unroll
	for (int i = 0; i < NUM_COL_PER_THREAD; ++i) {
		acc[i].x = acc[i].y = 0;
	}

	// Iterate over all blocks
	#pragma unroll
	for (int block_idx = warp_id; block_idx < num_blocks; block_idx += NUM_WARPS) {
		const int64_t block_index = block_table[req_index*max_num_block_per_seq + block_idx];
		const T2* v_block = (const T2*)(v_cache_offseted + (block_index*num_layers*num_heads + head_id)*BLOCK_SIZE*HEAD_DIM);
		const int64_t token_idx = block_idx*BLOCK_SIZE;

		#pragma unroll
		for (int col = lane_id; col < HEAD_DIM/2; col += WARP_SIZE) {
			T2 acc_elem = acc[col/WARP_SIZE];
			T sum_x = 0, sum_y = 0;
			#pragma unroll
			for (int i = 0; i < BLOCK_SIZE; i += 1) {
				T attn_score_elem = (T)attn_score[token_idx + i];
				T2 v_elem = v_block[INDEX_2D(BLOCK_SIZE, HEAD_DIM/2, i, col)];
				sum_x += attn_score_elem * v_elem.x;
				sum_y += attn_score_elem * v_elem.y;
			}
			acc[col/WARP_SIZE] = {acc_elem.x + sum_x, acc_elem.y + sum_y};
		}
	}
	__syncthreads();

	// Reduce accs among threads with the same lane_id
	#pragma unroll
	for (int i = 0; i < NUM_COL_PER_THREAD; ++i) {
		// In the "Iterate over all blocks" above, each thread is responsible
		// for column `lane_id`, `lane_id+WARP_SIZE`, `lane_id+2*WARP_SIZE`, ...

		// Now we focus on col = i*WARP_SIZE + lane_id
		// Copy the cols to the shared memory
		{
			int col = i*WARP_SIZE + lane_id;
			if (col < HEAD_DIM/2) {
				qkv_reduction_wksp[warp_id*WARP_SIZE + lane_id] = acc[i];
			}
			__syncthreads();
		}

		// Now our task is to, for every i in 0..WARP_SIZE-1,
		// summing up qkv_reduction_wksp[0][i], qkv_reduction_wksp[1][i], ... qkv_reduction_wksp[NUM_WARPS-1][i]
		// The i-th thread calculates the sum above for column threadIdx.x
		{
			const int thread_id = threadIdx.x;
			if (thread_id < WARP_SIZE && i*WARP_SIZE + thread_id < HEAD_DIM/2) {
				T sum_x = 0, sum_y = 0;
				#pragma unroll
				for (int j = 0; j < NUM_WARPS; ++j) {
					const T2 elem = qkv_reduction_wksp[j*WARP_SIZE + thread_id];
					sum_x += elem.x;
					sum_y += elem.y;
				}
				((T2*)result)[INDEX_3D(0, num_heads, HEAD_DIM/2, token_index, head_id, i*WARP_SIZE + thread_id)] = {sum_x, sum_y};
			}
			__syncthreads();
		}
	}
}

#define LAUNCH_DECODING_STAGE_ATTENTION_MHA_KERNEL(T, HEAD_DIM, BLOCK_SIZE) \
	fusedDecodingStageAttentionMHAKernel<T, HEAD_DIM, BLOCK_SIZE, DEFAULT_THREAD_BLOCK_SIZE><<<grid_dim, DEFAULT_THREAD_BLOCK_SIZE, shared_mem_size>>>( \
		result, qkvs, k_cache_offseted, v_cache_offseted, scale, block_table, input_lens, ith_decoding_req_req_index, ith_decoding_req_token_index, max_num_block_per_seq, num_layers \
	)

#define FUSED_DECODING_STAGE_ATTENTION_MHA_DISPATCH_HEAD_DIM_BLOCK_SIZE(T, HEAD_DIM) \
	switch (block_size) { \
		case 1: LAUNCH_DECODING_STAGE_ATTENTION_MHA_KERNEL(T, HEAD_DIM, 1); break; \
		case 2: LAUNCH_DECODING_STAGE_ATTENTION_MHA_KERNEL(T, HEAD_DIM, 2); break; \
		case 4: LAUNCH_DECODING_STAGE_ATTENTION_MHA_KERNEL(T, HEAD_DIM, 4); break; \
		case 8: LAUNCH_DECODING_STAGE_ATTENTION_MHA_KERNEL(T, HEAD_DIM, 8); break; \
		case 16: LAUNCH_DECODING_STAGE_ATTENTION_MHA_KERNEL(T, HEAD_DIM, 16); break; \
		case 32: LAUNCH_DECODING_STAGE_ATTENTION_MHA_KERNEL(T, HEAD_DIM, 32); break; \
		default: fprintf(stderr, "Unsupported block_size: %ld\n", block_size); assert(0); \
	}

#define FUSED_DECODING_STAGE_ATTENTION_MHA_DISPATCH_HEAD_DIM(T) \
	switch (head_dim) {	\
		case 64: FUSED_DECODING_STAGE_ATTENTION_MHA_DISPATCH_HEAD_DIM_BLOCK_SIZE(T, 64); break;	\
		case 80: FUSED_DECODING_STAGE_ATTENTION_MHA_DISPATCH_HEAD_DIM_BLOCK_SIZE(T, 80); break;	\
		case 128: FUSED_DECODING_STAGE_ATTENTION_MHA_DISPATCH_HEAD_DIM_BLOCK_SIZE(T, 128); break;	\
		default: fprintf(stderr, "Unsupported head_dim: %ld\n", head_dim); assert(0);			\
	}

template<typename T>
void fusedDecodingStageAttentionMHA(
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
	const int64_t num_heads,
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
) {
#ifdef DEBUG
#endif
	const int64_t NUM_THREAD_PER_KEY = WARP_SIZE / block_size;	// The size of the thread group
	assert_whenever((head_dim/2) % NUM_THREAD_PER_KEY == 0);
	T* k_cache_offseted = k_cache + layer_id * num_heads * block_size * head_dim;
	T* v_cache_offseted = v_cache + layer_id * num_heads * block_size * head_dim;
	dim3 grid_dim(num_heads, num_decoding_reqs);
	int shared_mem_size = std::max(((max_decoding_req_len+1 + block_size-1) / block_size) * block_size * sizeof(float), DEFAULT_THREAD_BLOCK_SIZE*2*sizeof(T));
	FUSED_DECODING_STAGE_ATTENTION_MHA_DISPATCH_HEAD_DIM(T);
}

template void fusedDecodingStageAttentionMHA(
	half* __restrict__ result,
	const half* __restrict__ qkvs,
	half* k_cache,
	half* v_cache,
	const float scale,
	const int64_t* __restrict__ block_table,
	const int64_t* __restrict__ input_lens,
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,
	const int64_t* __restrict__ ith_decoding_req_token_index,
	const int64_t max_decoding_req_len,
	const int64_t num_layers,
	const int64_t num_heads,
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
);
template void fusedDecodingStageAttentionMHA(
	float* __restrict__ result,
	const float* __restrict__ qkvs,
	float* k_cache,
	float* v_cache,
	const float scale,
	const int64_t* __restrict__ block_table,
	const int64_t* __restrict__ input_lens,
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,
	const int64_t* __restrict__ ith_decoding_req_token_index,
	const int64_t max_decoding_req_len,
	const int64_t num_layers,
	const int64_t num_heads,
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
);

}