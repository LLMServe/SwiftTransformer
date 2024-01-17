#include "fused_context_stage_attention.h"

#include <cassert>
#include <cstdio>

#include <cuda_pipeline.h>	// For pipelined memcopy
#include <mma.h>	// For Tensor Core

using namespace nvcuda;

#include "util/cuda_utils.h"

namespace st::kernel {

#define WARP_SIZE 32ul

// TILE_SIZE: The size of the tile (called Br/Bc in FlashAttention's paper)
// when processing QKV. We use "tile" instead of "block" to avoid confusion.
// NOTE. Tensor core only supports TILE_SIZE = 16!
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-type-sizes
#define TILE_SIZE 16ul

// Tuneable parameters
constexpr uint32_t DEFAULT_THREAD_BLOCK_SIZE = 256;	// Do not set too big since we need
	// a NUM_WARPS x TILE_SIZE x TILE_SIZE float array in shared memory

/*
	# fusedContextStageAttentionKernel

	## Overview

	The TensorCore-ed, FlashAttention-ed, and GroupQueryAttention-ed version of fusedContextAttentionKernel.

	This kernel is invoked on each input's context stage. It calculates the
	attention score of each token in the input (prompt). In other words, it
	is given the Q, K, and V matrices, and it calculates softmax(QK^T·scale+mask)*V.

	## Optimizations Implemented

	### Tensor Core

	This kernel leverages the Tensor Core when computing QK^T, which is generally
	faster than the original cudacore-based implementation.

	However, there are two pitiful things:
		- Tensor Core does not support mulplication between two fp32 matrices,
			so when datatype = fp32, this kernel fallbacks to CUDA Cores.
		- When calculating attn_mat x V, we need to multiply two fp32 matrices,
			which cannot leverage the tensor core.	(TODO. is fp32 necessary?)
	
	### FlashAttention & FlashAttention 2

	FlashAttention is a technique to speedup the context kernel which can dramatically
	speed up the context stage kernel. This kernel uses similar techniques to FlashAttention's,
	including tiling and shared memory.

	### Group Query Attention

	Group Query Attention (GQA) is a technique that maps one K/V to multiple queries. Strictly
	speaking, this is indeed an optimization on the whole model architecture instead of
	a kernel-level optimization. This kernel supports group query attention.

	## Parameters

	Since not every request is going to be processed, we need to pass an array
	`ith_context_request_index`. Please refer to `fused_decoding_stage_attention.cu`
	for more information about this array.

	## Algorithm & Implementation Details
	
	The implementation is similar to FlashAttention's.

	The grid shape is (num_heads, num_context_reqs). The (i,j)-th thread block
	calculates softmax(QK^T+mask)*V for head i of request j.

	## Note

	!!!PAY ATTENTION!!!VERY IMPORTANT!!! To avoid accessing out-of-bound memory,
	the actual first dimension of `qkvs` must >= num_tokens+15
*/

template<
	typename T,
	int64_t Q_HEADS_PER_THREAD_BLOCK,
	int64_t HEAD_DIM,
	int64_t THREAD_BLOCK_SIZE
> __global__ void fusedContextStageAttentionKernel(
	// The output
	T* __restrict__ result,		// [num_tokens, num_q_heads, head_dim]

	// QKVs
	const T* __restrict__ qkvs,	// [num_tokens(+15), num_q_heads+2*num_kv_heads, head_dim]

	// Other inputs
	const float qk_scale,
	const int64_t* __restrict__ input_lens,					// [num_reqs]
	const int64_t* __restrict__ ith_context_req_req_index,	// [num_context_reqs]
	const int32_t* __restrict__ ith_context_req_token_index,	// [num_context_reqs]
	int64_t num_tokens,
	int64_t num_kv_heads,

	// Buffers
	float* __restrict__ m_buf,	// [num_q_heads, num_tokens], rowmax
	float* __restrict__ l_buf	// [num_q_heads, num_tokens], sum of exp(x-m_buf[req_index])
) {
	constexpr int64_t NUM_WARPS = THREAD_BLOCK_SIZE / WARP_SIZE;

	// Grid: (num_q_heads/Q_HEADS_PER_THREAD_BLOCK) x num_context_reqs
	const int64_t num_q_heads = gridDim.x*Q_HEADS_PER_THREAD_BLOCK;
	const int64_t thread_blocks_per_kv_head = (num_q_heads/num_kv_heads)/Q_HEADS_PER_THREAD_BLOCK;
	const int64_t kv_head_index = blockIdx.x/thread_blocks_per_kv_head;

	// The indexes of q_heads that I (the current thread block) will deal with
	const int64_t my_q_head_begin = blockIdx.x*Q_HEADS_PER_THREAD_BLOCK;
	const int64_t my_q_head_end = (blockIdx.x+1)*Q_HEADS_PER_THREAD_BLOCK;

	const int64_t req_index = ith_context_req_req_index[blockIdx.y];
	const int64_t first_token_index = ith_context_req_token_index[blockIdx.y];
	const int64_t input_len = input_lens[req_index];
	const int64_t num_tiles = (input_len + TILE_SIZE - 1) / TILE_SIZE;

	const uint32_t warp_id = threadIdx.x / WARP_SIZE;	// Which warp we are in
	const uint32_t lane_id = threadIdx.x % WARP_SIZE;

	// To process every elements (16*16 = 256 elements in total), we 
	// group threads in the same warp into thread groups. Every thread
	// group contains two threads. Thread 0 and 16, thread 1 and 17, and
	// so on, are in the same group. Each thread group is assigned with
	// one row to proceed.
	#define THREAD_GROUP_SIZE 2
	const int64_t row = lane_id & 0x0Fu;	// The row, i.e. the thread group id
	const int64_t thread_group_offset = lane_id >> 4;	// The offset in the thread group (0 or 1)

	__shared__ T cur_k_tile[TILE_SIZE*HEAD_DIM];
	__shared__ T cur_v_tile[TILE_SIZE*HEAD_DIM];
	__shared__ float cur_qk_tile[NUM_WARPS*TILE_SIZE*TILE_SIZE];
	__shared__ T cur_qk_tile_T[NUM_WARPS*TILE_SIZE*TILE_SIZE];	// The "T" here means its type is T instead of float. It DOES NOT mean "Transposed"
	__shared__ float new_mi[NUM_WARPS*TILE_SIZE];
	__shared__ float new_li[NUM_WARPS*TILE_SIZE];
	__shared__ float cur_mi_cache[NUM_WARPS*TILE_SIZE];
	__shared__ float cur_li_cache[NUM_WARPS*TILE_SIZE];

	// Fragments for computing GEMM by Tensor Core
	// This is only used when T == FP16
	wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag;

	wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, wmma::row_major> d_frag;
	wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half> e_frag;

	// Step 0. Clear m_buf, l_buf, and result
	// Each warp s assigned with some rows (tokens);
	for (int64_t token_index = warp_id; token_index < input_len; token_index += NUM_WARPS) {
		for (int64_t q_head_index = my_q_head_begin; q_head_index < my_q_head_end; ++q_head_index) {
			if (lane_id == 0) {
				int64_t mlbuf_index = INDEX_2D(num_q_heads, num_tokens, q_head_index, first_token_index+token_index);
				m_buf[mlbuf_index] = -__FLT_MAX__;
				l_buf[mlbuf_index] = 0.0f;
			}
			#pragma unroll
			for (int64_t feature_index = lane_id; feature_index < HEAD_DIM; feature_index += WARP_SIZE)
				result[INDEX_3D(num_tokens, num_q_heads, HEAD_DIM,
					first_token_index+token_index, q_head_index, feature_index)] = 0.0f;
		}
	}
	// __syncthreads(); Omit this since we are going to call __syncthreads() later in the `for` loop

	for (int64_t k_tile_index = 0; k_tile_index < num_tiles; k_tile_index += 1) {
		// Here k_tile_index is `j` in line #5 in "Algorithm 1" in FlashAttention's paper

		// Step 1. Load K_j and V_j into shared memory
		// Every feature vector ([head-dim]) is handled by a warp
		__syncthreads();
		{
			#pragma unroll
			for (int64_t row = warp_id; row < TILE_SIZE; row += NUM_WARPS) {
				int64_t token_index = first_token_index + k_tile_index*TILE_SIZE + row;
				// We use pipelined memory copy here to increase performance
				constexpr int NUM_OF_T_IN_16_BYTES = 16 / sizeof(T);
				#pragma unroll
				for (int64_t feature_index = lane_id; feature_index < HEAD_DIM/NUM_OF_T_IN_16_BYTES; feature_index += WARP_SIZE) {
					__pipeline_memcpy_async(
						cur_k_tile + row*HEAD_DIM + feature_index*NUM_OF_T_IN_16_BYTES,
						qkvs + INDEX_3D(0, num_q_heads+2*num_kv_heads, HEAD_DIM,
							token_index, num_q_heads+kv_head_index, feature_index*NUM_OF_T_IN_16_BYTES),
						16,
						0
					);
					__pipeline_memcpy_async(
						cur_v_tile + row*HEAD_DIM + feature_index*NUM_OF_T_IN_16_BYTES,
						qkvs + INDEX_3D(0, num_q_heads+2*num_kv_heads, HEAD_DIM,
							token_index, num_q_heads+num_kv_heads+kv_head_index, feature_index*NUM_OF_T_IN_16_BYTES),
						16,
						0
					);
				}
			}
			__pipeline_commit();
			__pipeline_wait_prior(0);
		}
		__syncthreads();

		// Process each q_head separately
		for (int64_t q_head_index = my_q_head_begin; q_head_index < my_q_head_end; ++q_head_index) {
			// Pick out current m_buf & l_buf.
			float* cur_m_buf = m_buf + INDEX_2D(num_q_heads, num_tokens, q_head_index, first_token_index);	// [input_len]
			float* cur_l_buf = l_buf + INDEX_2D(num_q_heads, num_tokens, q_head_index, first_token_index);	// [input_len]

			// Now we process each q_tile one by one...
			#pragma unroll
			for (int64_t q_tile_index = warp_id; q_tile_index < num_tiles; q_tile_index += NUM_WARPS) {
				// Here q_block_idx is `i` in line #7 in "Algorithm 1" in FlashAttention's paper
				float* cur_qk_frag = cur_qk_tile + warp_id*TILE_SIZE*TILE_SIZE;
				T* cur_qk_frag_T = cur_qk_tile_T + warp_id*TILE_SIZE*TILE_SIZE;
				
				if (q_tile_index < k_tile_index) continue;

				// "Prefetch" in step 8. See the comments there for more details.
				float m_buf_prefetched, l_buf_prefetched;
				if (lane_id < TILE_SIZE) {
					m_buf_prefetched = cur_m_buf[q_tile_index*TILE_SIZE + lane_id];
					l_buf_prefetched = cur_l_buf[q_tile_index*TILE_SIZE + lane_id];
				}

				// Step 2. Compute Q_i * K_j^T. (Now K_j is stored in cur_k_block)
				// We use Tensor Core to calculate it if T == FP16
				if constexpr (std::is_same<T, __half>::value) {
					wmma::fill_fragment(c_frag, 0.0f);
					#pragma unroll
					for (int64_t hd_index = 0; hd_index < HEAD_DIM; hd_index += TILE_SIZE) {
						// Here `hd` stand for `head_dim`
						// Load the fragment from Q_i into a_frag
						// TODO (Optimize): This may cause uncoalesced global memory access
						#pragma unroll
						for (int row = lane_id/TILE_SIZE; row < TILE_SIZE; row += WARP_SIZE/TILE_SIZE) {
							int col = lane_id%TILE_SIZE;
							cur_qk_frag_T[row*TILE_SIZE+col] = qkvs[INDEX_3D(0, num_q_heads+2*num_kv_heads, HEAD_DIM,
								first_token_index + q_tile_index*TILE_SIZE + row, q_head_index, hd_index+ col)];
						}
						wmma::load_matrix_sync(
							a_frag,
							cur_qk_frag_T,
							TILE_SIZE
						);
						// Load the fragment from K_j into b1_frag
						wmma::load_matrix_sync(
							b_frag,
							cur_k_tile + hd_index,
							HEAD_DIM
						);
						// mma!
						wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
					}

					// Step 4. Store Q_i*K_j^T*qk_scale to cur_qk_tile
					// Since the following ops are mainly row-wise ops, to avoid
					// reduction between threads and shared memory bank conflicts,
					// we store cur_qk_frag in a transposed way.
					wmma::store_matrix_sync(
						cur_qk_frag,
						c_frag,
						TILE_SIZE,
						wmma::mem_col_major	// Store in the transposed way
					);
				} else {
					// Fallback to implementation with CUDA Core
					// Each thread calculates one answer in the result matrix
					#pragma unroll
					for (int64_t r = thread_group_offset; r < TILE_SIZE; r += THREAD_GROUP_SIZE) {
						int64_t c = lane_id&0x0Fu;
						float result = 0;
						for (int64_t hd_index = 0; hd_index < HEAD_DIM; hd_index += 1) {
							result += qkvs[INDEX_3D(num_tokens, num_q_heads+2*num_kv_heads, HEAD_DIM,
								first_token_index + q_tile_index*TILE_SIZE + r, q_head_index, hd_index)] *
								cur_k_tile[INDEX_2D(TILE_SIZE, HEAD_DIM, c, hd_index)];
						}
						cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, c, r)] = result;
					}
				}

				// Step 5. Scale & Mask
				#pragma unroll
				for (int64_t col = thread_group_offset; col < TILE_SIZE; col += THREAD_GROUP_SIZE) {
					float x = cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, col, row)];
					x *= qk_scale;
					x += col+k_tile_index*TILE_SIZE > row+q_tile_index*TILE_SIZE ? -10000.0f : 0.0f;
					if (row+q_tile_index*TILE_SIZE >= input_len ||
						col+k_tile_index*TILE_SIZE >= input_len) {
						x = -10000.0f;
					}
					cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, col, row)] = x;
				}

				// Save the result (attention score matrix) directly if you want to debug (I wish you good luck!)
				// if (lane_id < TILE_SIZE) {
				// 	const int row = lane_id;
				// 	const int token_index = q_tile_index*TILE_SIZE + row;
				// 	for (int i = 0; i < TILE_SIZE; ++i) {
				// 		const int feature_index = k_tile_index*TILE_SIZE + i;
				// 		if (token_index >= input_len || feature_index >= input_len) break;
				// 		result[INDEX_3D(num_tokens, num_heads, HEAD_DIM,
				// 			first_token_index+token_index, head_id, feature_index)] =
				// 		(__half)cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, i, row)];
				// 	}
				// }
				// continue;

				// Step 7. Compute:
				// - mij (containing the maximum value in every row)
				// - lij (containing the sum of exp(Sij-mij) in every row)
				// And then store them into new_mi and new_li, respectively.
				{
					float mij = -__FLT_MAX__;
					#pragma unroll
					for (int64_t col = thread_group_offset; col < TILE_SIZE; col += THREAD_GROUP_SIZE)
						mij = fmaxf(mij, cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, col, row)]);
					mij = fmaxf(mij, __shfl_xor_sync(0xFFFFFFFFu, mij, TILE_SIZE));	// Reduction within the same thread block
					float lij = 0.0f;
					#pragma unroll
					for (int64_t col = thread_group_offset; col < TILE_SIZE; col += THREAD_GROUP_SIZE) {
						float x = 
							__expf(cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, col, row)] - mij);
						lij += x;
						if constexpr (std::is_same<T, __half>::value) {
							cur_qk_frag_T[INDEX_2D(TILE_SIZE, TILE_SIZE, row, col)] = (T)x;
						} else {
							cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, col, row)] = x;
						}
					}
					lij += __shfl_xor_sync(0xFFFFFFFFu, lij, TILE_SIZE);
					new_mi[warp_id*TILE_SIZE + row] = mij;
					new_li[warp_id*TILE_SIZE + row] = lij;
				}

				// Step 8. Compute PijVj, and modify result[]
				// First we load cur_m_buf and cur_l_buf into shared memory (cur_mi_cache and cur_li_cache)
				// to avoid uncoalesced global memory access
				// Furthermore, we "prefetch" it before Step 2.
				if (lane_id < TILE_SIZE) {
					cur_mi_cache[warp_id*TILE_SIZE + lane_id] = m_buf_prefetched;
					cur_li_cache[warp_id*TILE_SIZE + lane_id] = l_buf_prefetched;
				}

				if constexpr (std::is_same<T, __half>::value) {
					// Calculate coefficents for each row
					// The definition of mi, mij, mi_new, li, lij, and li_new
					// is identical to that in Algorithm 1 in FlashAttention's
					// paper
					__shared__ T mult_old_elem[NUM_WARPS*TILE_SIZE];
					__shared__ T mult_new_elem[NUM_WARPS*TILE_SIZE];
					if (lane_id < TILE_SIZE) {
						float mi_new, li_new;
						{
							const float mi = cur_mi_cache[warp_id*TILE_SIZE + lane_id];
							const float mij = new_mi[warp_id*TILE_SIZE + lane_id];
							const float li = cur_li_cache[warp_id*TILE_SIZE + lane_id];
							const float lij = new_li[warp_id*TILE_SIZE + lane_id];
							mi_new = fmaxf(mi, mij);
							li_new = __expf(mi-mi_new)*li + __expf(mij-mi_new)*lij;

							mult_old_elem[warp_id*TILE_SIZE+lane_id] = li*__expf(mi-mi_new) / (li_new+1e-6f);
							mult_new_elem[warp_id*TILE_SIZE+lane_id] = __expf(mij-mi_new) / (li_new+1e-6f);
						}
						cur_m_buf[q_tile_index*TILE_SIZE + lane_id] = mi_new;
						cur_l_buf[q_tile_index*TILE_SIZE + lane_id] = li_new;
					}
					// Use implementation based on Tensor core
					// Load cur_qk_frag_T into wmma fragment
					wmma::load_matrix_sync(
						a_frag,
						cur_qk_frag_T,
						TILE_SIZE
					);
					const int64_t max_row = min(TILE_SIZE, input_len-q_tile_index*TILE_SIZE);
					#pragma unroll
					for (int64_t hd_index = 0; hd_index < HEAD_DIM; hd_index += TILE_SIZE) {
						wmma::fill_fragment(e_frag, 0.0f);
						wmma::load_matrix_sync(
							d_frag,
							cur_v_tile + hd_index,
							HEAD_DIM
						);
						wmma::mma_sync(
							e_frag,
							a_frag,
							d_frag,
							e_frag
						);
						wmma::store_matrix_sync(
							cur_qk_frag_T,
							e_frag,
							TILE_SIZE,
							wmma::mem_row_major
						);
						#pragma unroll
						for (int64_t row = lane_id/TILE_SIZE; row < max_row; row += WARP_SIZE/TILE_SIZE) {
							const int64_t col = lane_id % TILE_SIZE;
							const int64_t result_index = INDEX_3D(num_tokens, num_q_heads, HEAD_DIM,
							first_token_index+q_tile_index*TILE_SIZE+row, q_head_index, hd_index+col);
							result[result_index] = mult_old_elem[warp_id*TILE_SIZE+row]*result[result_index] + 
								mult_new_elem[warp_id*TILE_SIZE+row]*cur_qk_frag_T[row*TILE_SIZE+col];
						}
					}
				} else {
					// Fallback to implementation with CUDA core since Tensor core cannot deal with FP32
					const int64_t row_max = min(TILE_SIZE, input_len-q_tile_index*TILE_SIZE);
					#pragma unroll
					for (int64_t row = 0; row < row_max; row += 1) {
						// Calculate coefficents for each row
						// The definition of mi, mij, mi_new, li, lij, and li_new
						// is identical to that in Algorithm 1 in FlashAttention's
						// paper
						float mi_new, li_new, mult_old_elem, mult_new_elem;
						{
							const float mi = cur_mi_cache[warp_id*TILE_SIZE + row];
							const float mij = new_mi[warp_id*TILE_SIZE + row];
							const float li = cur_li_cache[warp_id*TILE_SIZE + row];
							const float lij = new_li[warp_id*TILE_SIZE + row];
							mi_new = fmaxf(mi, mij);
							li_new = __expf(mi-mi_new)*li + __expf(mij-mi_new)*lij;

							mult_old_elem = li*__expf(mi-mi_new) / (li_new+1e-6f);
							mult_new_elem = __expf(mij-mi_new) / (li_new+1e-6f);
						}

						// Modify Oi
						const int64_t result_index_0 = INDEX_3D(num_tokens, num_q_heads, HEAD_DIM,
							first_token_index+q_tile_index*TILE_SIZE+row, q_head_index, 0);
						#pragma unroll
						for (int64_t col = lane_id; col < HEAD_DIM; col += WARP_SIZE) {
							float new_result = 0;
							#pragma unroll
							for (int64_t i = 0; i < TILE_SIZE; i += 1) {
								// NOTE (Opt). MIO (Memory IO) queue is throttled here
								new_result += cur_qk_frag[INDEX_2D(TILE_SIZE, TILE_SIZE, i, row)] *
									(float)cur_v_tile[INDEX_2D(TILE_SIZE, HEAD_DIM, i, col)];
							}
							result[result_index_0 + col] = mult_old_elem*(float)result[result_index_0 + col] + 
								mult_new_elem*new_result;
						}
						
						// Save mi_new and li_new
						if (lane_id == 0) {
							cur_m_buf[q_tile_index*TILE_SIZE + row] = mi_new;
							cur_l_buf[q_tile_index*TILE_SIZE + row] = li_new;
						}
					}
				}
			}
		}
	}
}

#define LAUNCH_CONTEXT_STAGE_ATTENTION_KERNEL(Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM) \
	fusedContextStageAttentionKernel<T, Q_HEADS_PER_THREAD_BLOCK, HEAD_DIM, DEFAULT_THREAD_BLOCK_SIZE> \
	<<<grid_dim, DEFAULT_THREAD_BLOCK_SIZE>>>(result, qkvs, qk_scale, input_lens, \
		ith_context_req_req_index, ith_context_req_token_index, \
		num_tokens, num_kv_heads, m_buf, l_buf)

#define FUSED_CONTEXT_STAGE_ATTENTION_DISPATCH_Q_HEADS_PER_THREAD_BLOCK() \
	switch (q_heads_per_thread_block) {\
		case 1: FUSED_CONTEXT_STAGE_ATTENTION_DISPATCH_HEAD_DIM(1); break; \
		case 2: FUSED_CONTEXT_STAGE_ATTENTION_DISPATCH_HEAD_DIM(2); break; \
		case 4: FUSED_CONTEXT_STAGE_ATTENTION_DISPATCH_HEAD_DIM(4); break; \
		case 8: FUSED_CONTEXT_STAGE_ATTENTION_DISPATCH_HEAD_DIM(8); break; \
		default: fprintf(stderr, "Unsupported q_heads_per_thread_block: %ld\n", q_heads_per_thread_block); assert(0); \
	}

#define FUSED_CONTEXT_STAGE_ATTENTION_DISPATCH_HEAD_DIM(Q_HEADS_PER_THREAD_BLOCK) \
	switch (head_dim) { \
		case 64: LAUNCH_CONTEXT_STAGE_ATTENTION_KERNEL(Q_HEADS_PER_THREAD_BLOCK, 64); break; \
		case 80: LAUNCH_CONTEXT_STAGE_ATTENTION_KERNEL(Q_HEADS_PER_THREAD_BLOCK, 80); break; \
		case 128: LAUNCH_CONTEXT_STAGE_ATTENTION_KERNEL(Q_HEADS_PER_THREAD_BLOCK, 128); break; \
		default: fprintf(stderr, "Unsupported head_dim: %ld\n", head_dim); assert(0); \
	}

template<typename T>
void fusedContextStageAttention(
	T* __restrict__ result,
	const T* __restrict__ qkvs,
	const float qk_scale,
	const int64_t* __restrict__ input_lens,
	const int64_t num_context_reqs,
	const int64_t* __restrict__ ith_context_req_req_index,
	const int32_t* __restrict__ ith_context_req_token_index,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t num_tokens,
	float* __restrict__ m_buf,
	float* __restrict__ l_buf
) {
	int64_t q_heads_per_thread_block = num_q_heads == num_kv_heads ? 1 : 2;	// TODO Tune this
	#ifdef DEBUG
		assert(WARP_SIZE%TILE_SIZE == 0);
		assert(TILE_SIZE <= WARP_SIZE);
		assert(head_dim%TILE_SIZE == 0);
		assert((long)(qkvs)%256 == 0);	// Requirement posed by wmma::load_matrix_sync
		assert((num_q_heads/num_kv_heads) % q_heads_per_thread_block == 0);
	#endif
	dim3 grid_dim(num_q_heads/q_heads_per_thread_block, num_context_reqs);
	FUSED_CONTEXT_STAGE_ATTENTION_DISPATCH_Q_HEADS_PER_THREAD_BLOCK();
}

#define INSTANTIATE_FUSED_CONTEXT_STAGE_ATTENTION(T) \
	template void fusedContextStageAttention( \
		T* __restrict__, \
		const T* __restrict__, \
		const float, \
		const int64_t* __restrict__, \
		const int64_t, \
		const int64_t* __restrict__, \
		const int32_t* __restrict__, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		float* __restrict__, \
		float* __restrict__ \
	);

INSTANTIATE_FUSED_CONTEXT_STAGE_ATTENTION(__half)
INSTANTIATE_FUSED_CONTEXT_STAGE_ATTENTION(float)

}