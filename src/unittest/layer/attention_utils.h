#pragma once

#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "util/cublas_wrapper.h"
#include "layer/attention.h"

constexpr int64_t NUM_TOTAL_BLOCKS = 2048;	

struct PagedAttnParam {
	int64_t block_size;
	int64_t max_num_block_per_req;
};

class Indexes {
public:
	// CPU
	int64_t* ith_context_req_req_index_cpu;
	int32_t* ith_context_req_token_index_cpu;
	int64_t* ith_decoding_req_req_index_cpu;
	int64_t* ith_decoding_req_token_index_cpu;
	int64_t batch_size;

	// GPU
	int64_t* ith_context_req_req_index;
	int32_t* ith_context_req_token_index;
	int64_t* ith_decoding_req_req_index;
	int64_t* ith_decoding_req_token_index;

	Indexes(){
		ith_context_req_req_index_cpu = nullptr;
		ith_context_req_token_index_cpu = nullptr;
		ith_decoding_req_req_index_cpu = nullptr;
		ith_decoding_req_token_index_cpu = nullptr;
	}

	Indexes(const int64_t batch_size): batch_size(batch_size) {
		// CPU
		ith_context_req_req_index_cpu = new int64_t[batch_size];
		ith_context_req_token_index_cpu = new int32_t[batch_size+1];
		ith_decoding_req_req_index_cpu = new int64_t[batch_size];
		ith_decoding_req_token_index_cpu = new int64_t[batch_size];

		// GPU
		CUDA_CHECK(cudaMalloc(&ith_context_req_req_index, batch_size * sizeof(int64_t)));
		CUDA_CHECK(cudaMalloc(&ith_context_req_token_index, (batch_size+1) * sizeof(int32_t)));
		CUDA_CHECK(cudaMalloc(&ith_decoding_req_req_index, batch_size * sizeof(int64_t)));
		CUDA_CHECK(cudaMalloc(&ith_decoding_req_token_index, batch_size * sizeof(int64_t)));
	}

	~Indexes() {
		if (ith_context_req_req_index_cpu != nullptr) {
			delete[] ith_context_req_req_index_cpu;
			delete[] ith_context_req_token_index_cpu;
			delete[] ith_decoding_req_req_index_cpu;
			delete[] ith_decoding_req_token_index_cpu;
			CUDA_CHECK(cudaFree(ith_context_req_req_index));
			CUDA_CHECK(cudaFree(ith_context_req_token_index));
			CUDA_CHECK(cudaFree(ith_decoding_req_req_index));
			CUDA_CHECK(cudaFree(ith_decoding_req_token_index));
		}
	}

	void toGPU(){
		CUDA_CHECK(cudaMemcpy(
			ith_context_req_req_index,
			ith_context_req_req_index_cpu,
			sizeof(int64_t) * batch_size,
			cudaMemcpyHostToDevice
		));
		CUDA_CHECK(cudaMemcpy(
			ith_context_req_token_index,
			ith_context_req_token_index_cpu,
			sizeof(int32_t) * (batch_size+1),
			cudaMemcpyHostToDevice
		));
		CUDA_CHECK(cudaMemcpy(
			ith_decoding_req_req_index,
			ith_decoding_req_req_index_cpu,
			sizeof(int64_t) * batch_size,
			cudaMemcpyHostToDevice
		));
		CUDA_CHECK(cudaMemcpy(
			ith_decoding_req_token_index,
			ith_decoding_req_token_index_cpu,
			sizeof(int64_t) * batch_size,
			cudaMemcpyHostToDevice
		));
	}
};

void build_block_table(
	int64_t* block_table, 
	const int64_t batch_size, 
	const PagedAttnParam pagedattn_param,
	const int64_t *input_len_cpu
){
    constexpr int64_t num_total_blocks = NUM_TOTAL_BLOCKS;
    // Construct the initial block table
	int64_t num_allocated_blocks = 0;
	std::function<int64_t(void)> allocateNewBlock = [&]() -> int64_t {
		num_allocated_blocks += 1;
		assert(num_allocated_blocks < num_total_blocks);
		return num_allocated_blocks;
	};
	int64_t* allocated_block_cnt = new int64_t[batch_size];
	int64_t* block_table_cpu = new int64_t[batch_size * pagedattn_param.max_num_block_per_req];
	for (int64_t i = 0; i < batch_size; i++) {
		int64_t block_needed = (input_len_cpu[i]+1 + pagedattn_param.block_size-1) / pagedattn_param.block_size;
		allocated_block_cnt[i] = block_needed;
		assert (block_needed <= pagedattn_param.max_num_block_per_req);
		for (int64_t j = 0; j < block_needed; j++) {
			block_table_cpu[i * pagedattn_param.max_num_block_per_req + j] = allocateNewBlock();
		}
		for (int64_t j = block_needed; j < pagedattn_param.max_num_block_per_req; ++j) {
			block_table_cpu[i * pagedattn_param.max_num_block_per_req + j] = -10000000;
		}
	}
	CUDA_CHECK(cudaMemcpy(block_table, block_table_cpu, sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice));
	delete[] block_table_cpu;
	sync_check_cuda_error();
}

Indexes get_req_index(const int64_t batch_size, const int64_t* input_len_cpu, const bool* is_context_stage_cpu){
	// Calculate indexes of requests in context stage and regression stage
	// Will be used in the attention layer (fusedDecodingStageAttentionKernel and fusedContextStageAttentionKernel)
	int64_t num_context_reqs = 0, num_decoding_reqs = 0;
	Indexes indexes(batch_size);
	int64_t cur_token_index = 0;
	for (int64_t i = 0; i < batch_size; ++i) {
		if (is_context_stage_cpu[i]) {
			indexes.ith_context_req_req_index_cpu[num_context_reqs] = i;
			indexes.ith_context_req_token_index_cpu[num_context_reqs] = cur_token_index;
			num_context_reqs += 1;
			cur_token_index += input_len_cpu[i];
		} else {
			indexes.ith_decoding_req_req_index_cpu[num_decoding_reqs] = i;
			indexes.ith_decoding_req_token_index_cpu[num_decoding_reqs] = cur_token_index;
			num_decoding_reqs += 1;
			cur_token_index += 1;
		}
	}
	indexes.ith_context_req_token_index_cpu[num_context_reqs] = cur_token_index;
	return indexes;
}