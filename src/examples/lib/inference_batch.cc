#include "inference_batch.h"

#include <mpi.h>

#include "unistd.h"
#include "util/cuda_utils.h"

#define CHRONO_TIMEPOINT_TO_MILLISECONDS(x) (std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(x).count())

// run_batched_inference: Inference (compute the output tokens) for a batch of requests
// This function takes a batch of requests as input, and then run context stage once and
// decoding stage multiple times (for a particular request, this stops when `end_token`
// is generated or the number of decoding steps reaches `max_decoding_step`)
template<typename T>
RuntimeUsage run_batched_inference(
	std::vector<std::vector<int64_t>> &output_tokens_batched,
	const std::vector<std::vector<int64_t>> &input_tokens_batched,
	st::model::Gpt<T> &gpt,		// The GPT model. Please load the model weights before calling this function
	const st::model::GptHyperParam &hyper_param,
	const st::model::GptPagedAttnParam &pagedattn_param,
	const st::model::GptParallelismParam &parallel_param,
	const int64_t max_decoding_step,	// Maximum number of decoding steps
	const int64_t end_token,			// Stop inference when this token is generated. If you don't want to use this, set it to -1
	const int64_t num_total_blocks,		// Allocate this number of blocks at the beginning
									// NOTE. Since our k/v cache is a continuous array, we need to allocate all blocks at the beginning
	bool print_debug_info,
	bool exit_after_context_stage,
	std::optional<SimpleVocabDecoder> vocab_decoder
) {
	output_tokens_batched.clear();
	
	int64_t batch_size = input_tokens_batched.size();
	if (print_debug_info) {
		std::cout << hyper_param << std::endl;
		std::cout << pagedattn_param << std::endl;
		printf("batch_size: %ld\n", batch_size);
		printf("parallel_param.tensor parallel: %ld\n", parallel_param.tensor_para_size);
		printf("num_total_blocks: %ld\n", num_total_blocks);
	}

	// Allocate space for output & k/v cache
	if (print_debug_info) {
		printf("Allocating space for k/v cache...\n");
	}
	T* *d_k_cache, *d_v_cache;
	// const int64_t local_num_q_heads = hyper_param.num_q_heads / parallel_param.tensor_para_size;
	const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallel_param.tensor_para_size;
	CUDA_CHECK(cudaMalloc(&d_k_cache, (long long)sizeof(T) * num_total_blocks * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim));
	CUDA_CHECK(cudaMalloc(&d_v_cache, (long long)sizeof(T) * num_total_blocks * hyper_param.num_layers * local_num_kv_heads * pagedattn_param.block_size * hyper_param.head_dim));
	CUDA_FREE_AT_RETURN(d_k_cache);
	CUDA_FREE_AT_RETURN(d_v_cache);
	sync_check_cuda_error();

	// Construct the initial block table
	int64_t* d_block_table;
	CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req));
	CUDA_FREE_AT_RETURN(d_block_table);
	int64_t num_allocated_blocks = 0;
	std::function<int64_t(void)> allocateNewBlock = [&]() -> int64_t {
		num_allocated_blocks += 1;
		if(num_allocated_blocks >= num_total_blocks) {
			printf("Allocating new block %ld, but num_total_blocks is %ld\n", num_allocated_blocks, num_total_blocks);
			exit(1);
		}
		return num_allocated_blocks;
	};
	int64_t* allocated_block_cnt = new int64_t[batch_size];
	if (print_debug_info) {
		printf("Constructing the initial block table...\n");
	}
	int64_t* h_block_table = new int64_t[batch_size * pagedattn_param.max_num_block_per_req];
	for (int64_t i = 0; i < batch_size; i++) {
		int64_t block_needed = (input_tokens_batched[i].size() + pagedattn_param.block_size-1) / pagedattn_param.block_size;
		allocated_block_cnt[i] = block_needed;
		assert (block_needed <= pagedattn_param.max_num_block_per_req);
		for (int64_t j = 0; j < block_needed; j++) {
			h_block_table[i * pagedattn_param.max_num_block_per_req + j] = allocateNewBlock();
		}
		for (int64_t j = block_needed; j < pagedattn_param.max_num_block_per_req; ++j) {
			h_block_table[i * pagedattn_param.max_num_block_per_req + j] = -10000000;
		}
	}
	cudaMemcpy(d_block_table, h_block_table, sizeof(int64_t) * batch_size * pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice);
	delete[] h_block_table;
	sync_check_cuda_error();

	// Allocate arrays for decoding stage
	// We put them here to avoid interference with profiling
	// Output tokens
	output_tokens_batched.resize(batch_size);
	// Input of the current decoding step, initially = output of the last token in the context stage
	T* d_decoding_input;
	CUDA_CHECK(cudaMalloc(&d_decoding_input, sizeof(T) * batch_size * hyper_param.hidden_size));
	CUDA_FREE_AT_RETURN(d_decoding_input);
	// Output of the current decoding step
	T* d_decoding_output;
	CUDA_CHECK(cudaMalloc(&d_decoding_output, sizeof(T) * batch_size * hyper_param.hidden_size));
	CUDA_FREE_AT_RETURN(d_decoding_output);
	sync_check_cuda_error();

	// Context stage
	sync_check_cuda_error_force();
	if (parallel_param.pipeline_para_size != 1 || parallel_param.tensor_para_size != 1)
		MPI_Barrier(MPI_COMM_WORLD);
	auto context_stage_start_time = std::chrono::steady_clock::now();
	if (print_debug_info) {
		printf("Running context stage...\n");
	}

	std::vector<int64_t> first_token_indexes(batch_size, 0);
	auto cur_iter_output_tokens = gpt.forward(
		input_tokens_batched,
		first_token_indexes,

		d_k_cache,
		d_v_cache,
		d_block_table	
	);
	sync_check_cuda_error_force();

	// add generated tokens to the output
	for (int64_t i = 0; i < batch_size; ++i) {
		output_tokens_batched[i].push_back(cur_iter_output_tokens[i]);
	}

	auto context_stage_end_time = std::chrono::steady_clock::now();
	if (parallel_param.pipeline_para_size != 1 || parallel_param.tensor_para_size != 1)
		MPI_Barrier(MPI_COMM_WORLD);

	if (exit_after_context_stage) {
		return RuntimeUsage {
			.context_stage_time = CHRONO_TIMEPOINT_TO_MILLISECONDS(context_stage_end_time - context_stage_start_time),
			.decoding_stage_time = 0,
			.total_time = CHRONO_TIMEPOINT_TO_MILLISECONDS(context_stage_end_time - context_stage_start_time)
		};
	}

	if (print_debug_info) {
		printf("Running decoding stage...\n");
	}
	// Decoding stage
	// In each step, the input of the function `gpt.forward()` is:
	// [
	//	The last token from the first unfinished request,
	//	The last token from the second unfinished request,
	//	...
	//	The last token from the last unfinished request
	// ]
	// Pay attention that, the i-th token is NOT the last token of the i-th request,
	// but the last token of the i-th UNFINISHED request
	// So we maintain an array, request_id, to record the original request id of each input token
	int64_t* request_id = new int64_t[batch_size];
	int64_t cur_notdone_input_cnt = batch_size;
	std::vector<std::vector<int64_t>> next_iter_input_tokens_batched(batch_size);
	for (int64_t i = 0; i < batch_size; i++) {
		request_id[i] = i;
		next_iter_input_tokens_batched[i].push_back(cur_iter_output_tokens[i]);
		first_token_indexes[i] = input_tokens_batched[i].size();
	}

	auto decoding_stage_start_time = std::chrono::steady_clock::now();

	for (int64_t step = 0; step < max_decoding_step && cur_notdone_input_cnt != 0 ; ++step) {
		if (print_debug_info) {
			printf("Step %ld, Number of unfinished requests: %ld\n", step, cur_notdone_input_cnt);
		}
		// Allocate new blocks if needed
		for (int64_t i = 0; i < cur_notdone_input_cnt; ++i) {
			int64_t block_needed = (input_tokens_batched[request_id[i]].size()
									+ output_tokens_batched[request_id[i]].size()
									+ pagedattn_param.block_size - 1) 
									/ pagedattn_param.block_size;
			while (allocated_block_cnt[request_id[i]] < block_needed) {
				int64_t new_block = allocateNewBlock();
				CUDA_CHECK(cudaMemcpy(d_block_table + i*pagedattn_param.max_num_block_per_req + allocated_block_cnt[request_id[i]],
					&new_block, sizeof(int64_t), cudaMemcpyHostToDevice));
				allocated_block_cnt[request_id[i]] += 1;
			}
		}
		sync_check_cuda_error();

		auto cur_iter_output_tokens = gpt.forward(
			next_iter_input_tokens_batched,
			first_token_indexes,

			d_k_cache,
			d_v_cache,
			d_block_table	
		);
		// Decode generated token and put them into output_tokens_batched
		// Prepare input for the next round
		int64_t ptr = 0;
		int64_t new_notdone_input_cnt = cur_notdone_input_cnt;
		next_iter_input_tokens_batched.clear();
		first_token_indexes.clear();
		for (int64_t i = 0; i < cur_notdone_input_cnt; ++i) {
			int64_t result_token = cur_iter_output_tokens[i];
			output_tokens_batched[request_id[i]].push_back(result_token);
			if (result_token == end_token) {
				// The generation of this request is done
				--new_notdone_input_cnt;
			} else {
				next_iter_input_tokens_batched.push_back(std::vector<int64_t>{result_token});
				first_token_indexes.push_back(input_tokens_batched[request_id[i]].size() + output_tokens_batched[request_id[i]].size() - 1);
				// Copy k/v cache to the right place if necessary (can be optimized in the future)
				if (i != ptr) {
					request_id[ptr] = request_id[i];
					cudaMemcpyAsync(
						d_block_table + ptr*pagedattn_param.max_num_block_per_req,
						d_block_table + i*pagedattn_param.max_num_block_per_req,
						sizeof(int64_t) * pagedattn_param.max_num_block_per_req,
						cudaMemcpyDeviceToDevice
					);
					sync_check_cuda_error();
				}
				ptr += 1;
			}
		}
		cur_notdone_input_cnt = new_notdone_input_cnt;
	}

	sync_check_cuda_error_force();
	auto decoding_stage_end_time = std::chrono::steady_clock::now();

	// Calculate time usage
	RuntimeUsage runtime_usage = {
		.context_stage_time = CHRONO_TIMEPOINT_TO_MILLISECONDS(context_stage_end_time - context_stage_start_time),
		.decoding_stage_time = CHRONO_TIMEPOINT_TO_MILLISECONDS(decoding_stage_end_time - decoding_stage_start_time),
		.total_time = CHRONO_TIMEPOINT_TO_MILLISECONDS(decoding_stage_end_time - context_stage_start_time)
	};
	return runtime_usage;
}

// Instantiation
#define INSTANTIATE_RUN_BATCHED_INFERENCE(T) \
	template RuntimeUsage run_batched_inference<T>( \
		std::vector<std::vector<int64_t>> &output_tokens_batched, \
		const std::vector<std::vector<int64_t>> &input_tokens_batched, \
		st::model::Gpt<T> &gpt, \
		const st::model::GptHyperParam &hyper_param, \
		const st::model::GptPagedAttnParam &pagedattn_param, \
		const st::model::GptParallelismParam &parallel_param, \
		const int64_t max_decoding_step, \
		const int64_t end_token, \
		const int64_t num_total_blocks, \
		bool print_debug_info, \
		bool exit_after_context_stage, \
		std::optional<SimpleVocabDecoder> vocab_decoder \
	);

INSTANTIATE_RUN_BATCHED_INFERENCE(float)
INSTANTIATE_RUN_BATCHED_INFERENCE(half)
