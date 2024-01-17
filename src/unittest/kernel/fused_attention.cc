#include <random>
#include <cstring>
#include <vector>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/fused_context_stage_attention.h"
#include "kernel/fused_decoding_stage_attention.h"
#include "kernel/kvcache_mgmt.h"

#include "attention_ref.h"
#include "kvcache_mgmt_ref.h"

using torch::Tensor;

template<typename T>
class FusedAttentionTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(FusedAttentionTestSuite, SupportTypes);

struct TestcaseConfig {
	std::string name;
	int64_t num_layers;	// Only support 1 now
	int64_t num_q_heads;
	int64_t num_kv_heads;
	int64_t head_dim;
	int64_t num_reqs;
	int64_t max_input_len;
	std::vector<int64_t> block_sizes;
};

std::vector<TestcaseConfig> testcases = {
	// A tiny model for debugging
	{"Tiny model", 1, 1, 1, 64, 1, 32, {16}},
	// A tiny model with grouped query attention, for debugging
	{"Tiny model with GQA", 1, 4, 2, 64, 1, 32, {16}},
	// OPT-125M
	{"OPT-125M", 1, 12, 12, 64, 667, 123, {1, 2, 4, 8, 16, 32}},
	// OPT-1.3B
	{"OPT-1.3B", 1, 32, 32, 64, 667, 123, {1, 2, 4, 8, 16, 32}},
	// OPT-2.7B
	{"OPT-2.7B", 1, 32, 32, 80, 354, 123, {4, 8, 16, 32}},
	// OPT-6.7B
	{"OPT-6.7B", 1, 32, 32, 128, 354, 123, {1, 2, 4, 8, 16, 32}},
	// OPT-30B
	{"OPT-30B", 1, 56, 56, 128, 354, 123, {1, 2, 4, 8, 16, 32}},
	// LLaMA2-70B
	{"LLaMA2-70B", 1, 64, 8, 128, 354, 123, {1, 2, 4, 8, 16, 32}},
};

struct TestcaseData {
	Tensor qkvs;	// [num_tokens, 3, num_heads, head_dim]
	Tensor k_cache;	// [num_blocks, num_layers, num_heads, block_size, head_dim]
	Tensor v_cache;	// [num_blocks, num_layers, num_heads, block_size, head_dim]
	float qk_scale;
	std::vector<int64_t> block_table;	// [num_reqs, max_num_block_per_seq]
	std::vector<int64_t> input_lens;	// [num_reqs]
	int64_t num_layers;
	int64_t num_q_heads;
	int64_t num_kv_heads;
	int64_t head_dim;
	int64_t block_size;
	int64_t batch_size;	// num_reqs
	int64_t num_tokens;
	int64_t max_num_block_per_seq;
	int64_t max_input_len;
	std::vector<int64_t> is_context_stage;
	std::vector<int64_t> first_token_index;
};

TestcaseData GenerateTestcaseData(TestcaseConfig config, int64_t block_size, float decoding_stage_rate = 0.5, int random_seed = 0) {
	std::mt19937 gen(random_seed);
	TestcaseData result;
	assert(config.num_layers == 1);

	// Basic parameters
	result.qk_scale = 1.0f / std::sqrt((float)config.head_dim);
	result.num_layers = config.num_layers;
	result.num_q_heads = config.num_q_heads;
	result.num_kv_heads = config.num_kv_heads;
	result.head_dim = config.head_dim;
	result.block_size = block_size;
	result.batch_size = config.num_reqs;
	result.max_input_len = config.max_input_len;
	result.max_num_block_per_seq = (config.max_input_len+1 + block_size - 1) / block_size + randNumber(gen, 0, 10);

	// Resize vectors
	result.block_table.resize(config.num_reqs * result.max_num_block_per_seq);
	result.input_lens.resize(config.num_reqs);
	result.is_context_stage.resize(config.num_reqs);
	result.first_token_index.resize(config.num_reqs);

	// Generate token metadata
	int64_t num_tokens = 0;
	int64_t num_blocks = 0;
	for (int64_t req_index = 0; req_index < config.num_reqs; ++req_index) {
		// Generate input #input_index
		bool is_decoding_stage = randNumber(gen, 0, 1000) < 1000*decoding_stage_rate;
		auto input_len = randNumber(gen, 1l, (int64_t)config.max_input_len-1);
		result.input_lens[req_index] = input_len;
		result.is_context_stage[req_index] = !is_decoding_stage;
		result.first_token_index[req_index] = num_tokens;
		if (is_decoding_stage) {
			num_tokens += 1;
		} else {
			num_tokens += input_len;
		}
		num_blocks += (input_len+1+block_size-1) / block_size;
	}

	result.num_tokens = num_tokens;

	// Generate qkvs, k_cache and v_cache
	result.qkvs = torch::rand({num_tokens + 15, config.num_q_heads+2*config.num_kv_heads, config.head_dim}, torch::kCUDA) - 0.5;
	result.k_cache = torch::rand({num_blocks, config.num_layers, config.num_kv_heads, block_size, config.head_dim}, torch::kCUDA) - 0.5;
	result.v_cache = torch::rand({num_blocks, config.num_layers, config.num_kv_heads, block_size, config.head_dim}, torch::kCUDA) - 0.5;

	// Generate block_table
	std::vector<int64_t> all_available_blocks;
	for (int64_t block_index = 0; block_index < num_blocks; ++block_index) {
		all_available_blocks.push_back(block_index);
	}
	std::random_shuffle(all_available_blocks.begin(), all_available_blocks.end());
	for (int64_t req_index = 0; req_index < config.num_reqs; ++req_index) {
		int64_t input_len = result.input_lens[req_index];
		int64_t num_blocks = (input_len+1 + block_size - 1) / block_size;
		for (int64_t block_index = 0; block_index < num_blocks; ++block_index) {
			result.block_table[req_index * result.max_num_block_per_seq + block_index] = all_available_blocks.back();
			all_available_blocks.pop_back();
		}
		for (int64_t block_index = num_blocks; block_index < result.max_num_block_per_seq; ++block_index) {
			result.block_table[req_index * result.max_num_block_per_seq + block_index] = -100000;
		}
	}
	
	return result;
}

template<typename T>
void runFusedDecodingStageAttentionKernel(
	Tensor &result,		// [num_tokens, num_q_heads, head_dim]
	TestcaseData &data
) {
	GpuArray<int64_t> block_table(data.block_table);
	GpuArray<int64_t> input_lens(data.input_lens);
	GpuArray<int64_t> first_token_index(data.first_token_index);

	int64_t num_decoding_reqs = 0;
	std::vector<int64_t> ith_decoding_req_req_index;
	std::vector<int64_t> ith_decoding_req_token_index;
	int64_t cur_token_index = 0;
	for (int64_t i = 0; i < data.batch_size; ++i) {
		bool is_context_stage = data.is_context_stage[i];
		if (!is_context_stage) {
			num_decoding_reqs += 1;
			ith_decoding_req_req_index.push_back(i);
			ith_decoding_req_token_index.push_back(cur_token_index);
			cur_token_index += 1;
		} else {
			cur_token_index += data.input_lens[i];
		}
	}
	assert(cur_token_index == data.num_tokens);
	GpuArray<int64_t> ith_decoding_req_req_index_gpu(ith_decoding_req_req_index);
	GpuArray<int64_t> ith_decoding_req_token_index_gpu(ith_decoding_req_token_index);
	sync_check_cuda_error();

	if (num_decoding_reqs != 0) {
		st::kernel::fusedDecodingStageAttention<T>(
			(T*)result.data_ptr(),
			(T*)data.qkvs.data_ptr(),
			(T*)data.k_cache.data_ptr(),
			(T*)data.v_cache.data_ptr(),
			data.qk_scale,
			(int64_t*)block_table.data,
			(int64_t*)input_lens.data,
			(int64_t)num_decoding_reqs,
			(int64_t*)ith_decoding_req_req_index_gpu.data,
			(int64_t*)ith_decoding_req_token_index_gpu.data,
			(int64_t)data.max_input_len,
			(int64_t)data.num_layers,
			(int64_t)data.num_q_heads,
			(int64_t)data.num_kv_heads,
			(int64_t)data.head_dim,
			0ul,
			(int64_t)data.block_size,
			(int64_t)data.max_num_block_per_seq
		);
		sync_check_cuda_error_force();
	} else {
		printf("Warn: num_decoding_reqs == 0\n");
	}
}

template<typename T>
void runFusedContextStageAttentionKernel(
	Tensor &result,		// [num_tokens, num_q_heads, head_dim]
	TestcaseData &data
) {
	GpuArray<int64_t> block_table(data.block_table);
	GpuArray<int64_t> input_lens(data.input_lens);
	GpuArray<int64_t> first_token_index(data.first_token_index);

	int64_t num_context_reqs = 0;
	std::vector<int64_t> ith_context_req_req_index;
	std::vector<int64_t> ith_context_req_token_index;
	int64_t cur_token_index = 0;
	for (int64_t i = 0; i < data.batch_size; ++i) {
		bool is_context_stage = data.is_context_stage[i];
		if (is_context_stage) {
			num_context_reqs += 1;
			ith_context_req_req_index.push_back(i);
			ith_context_req_token_index.push_back(cur_token_index);
			cur_token_index += data.input_lens[i];
		} else {
			cur_token_index += 1;
		}
	}
	assert (cur_token_index == data.num_tokens);
	GpuArray<int64_t> ith_context_req_req_index_gpu(ith_context_req_req_index);
	GpuArray<int64_t> ith_context_req_token_index_gpu(ith_context_req_token_index);
	std::vector<float> m_buf(data.num_q_heads * data.num_tokens);
	std::vector<float> l_buf(data.num_q_heads * data.num_tokens);
	GpuArray<float> m_buf_gpu(m_buf);
	GpuArray<float> l_buf_gpu(l_buf);
	sync_check_cuda_error_force();

	if (num_context_reqs != 0) {
		st::kernel::fusedContextStageAttention(
			(T*)result.data_ptr(),
			(T*)data.qkvs.data_ptr(),
			data.qk_scale,
			(int64_t*)input_lens.data,
			(int64_t)num_context_reqs,
			(int64_t*)ith_context_req_req_index_gpu.data,
			(int64_t*)ith_context_req_token_index_gpu.data,
			(int64_t)data.num_q_heads,
			(int64_t)data.num_kv_heads,
			(int64_t)data.head_dim,
			(int64_t)data.num_tokens,
			m_buf_gpu.data,
			l_buf_gpu.data
		);
		sync_check_cuda_error_force();
	} else {
		printf("Warn: num_context_reqs == 0\n");
	}
}

template<typename T>
bool checkKVCacheCopyKernel(TestcaseData &data) {
	GpuArray<int64_t> block_table(data.block_table);
	GpuArray<int64_t> input_lens(data.input_lens);
	GpuArray<int64_t> first_token_index(data.first_token_index);

	int64_t num_context_reqs = 0;
	std::vector<int64_t> ith_context_req_req_index;
	std::vector<int64_t> ith_context_req_token_index;
	int64_t cur_token_index = 0;
	for (int64_t i = 0; i < data.batch_size; ++i) {
		bool is_context_stage = data.is_context_stage[i];
		if (is_context_stage) {
			num_context_reqs += 1;
			ith_context_req_req_index.push_back(i);
			ith_context_req_token_index.push_back(cur_token_index);
			cur_token_index += data.input_lens[i];
		} else {
			cur_token_index += 1;
		}
	}
	assert (cur_token_index == data.num_tokens);
	GpuArray<int64_t> ith_context_req_req_index_gpu(ith_context_req_req_index);
	GpuArray<int64_t> ith_context_req_token_index_gpu(ith_context_req_token_index);
	if (num_context_reqs == 0)
		return true;

	// Calculate reference k/v cache
	torch::Tensor ref_k_cache = data.k_cache.clone();	// [/, num_layers, num_kv_heads, block_size, head_dim]
	torch::Tensor ref_v_cache = data.v_cache.clone();
	st::reference::kernel::saveContextStageKVCacheKernelRef(
		ref_k_cache,
		ref_v_cache,
		data.qkvs,
		torch::from_blob(data.block_table.data(), {data.batch_size, data.max_num_block_per_seq}, torch::kInt64),
		torch::from_blob(data.input_lens.data(), {data.batch_size}, torch::kInt64),
		torch::from_blob(data.is_context_stage.data(), {data.batch_size}, torch::kInt64),
		0
	);

	// Launch the kernel
	st::kernel::saveContextStageKVCache(
		(T*)data.k_cache.data_ptr(),
		(T*)data.v_cache.data_ptr(),

		(T*)data.qkvs.data_ptr(),
		(int64_t*)block_table.data,

		(int64_t*)input_lens.data,
		(int64_t)num_context_reqs,
		(int64_t*)ith_context_req_req_index_gpu.data,
		(int64_t*)ith_context_req_token_index_gpu.data,

		(int64_t)data.block_size,
		(int64_t)data.max_num_block_per_seq,
		1ul,
		(int64_t)data.num_q_heads,
		(int64_t)data.num_kv_heads,
		(int64_t)data.head_dim,
		(int64_t)0
	);
	sync_check_cuda_error_force();

	// Check
	bool is_passed;
	is_passed = isArrayAlmostEqual(
		(T*)data.k_cache.data_ptr(),
		(T*)ref_k_cache.data_ptr(),
		data.k_cache.numel(),
		true,
		true
	);
	if (!is_passed) return false;
	is_passed = isArrayAlmostEqual(
		(T*)data.v_cache.data_ptr(),
		(T*)ref_v_cache.data_ptr(),
		data.v_cache.numel(),
		true,
		true
	);
	if (!is_passed) return false;
	return true;
}

template<typename T>
void runAllTestcases(double decoding_stage_rate, bool run_context_stage, bool run_decoding_stage, bool run_kvcache_copy) {
	printf("run_context_stage = %s\n", run_context_stage ? "true" : "false");
	printf("run_decoding_stage = %s\n", run_decoding_stage ? "true" : "false");
	for (TestcaseConfig config : testcases) {
		printf("Testing %s\n", config.name.c_str());
		printf("  num_q_heads: %ld\n", config.num_q_heads);
		printf("  num_kv_heads: %ld\n", config.num_kv_heads);
		printf("  head_dim: %ld\n", config.head_dim);
		printf("  num_reqs: %ld\n", config.num_reqs);
		printf("  max_input_len: %ld\n", config.max_input_len);
		for (auto block_size : config.block_sizes) {
			TestcaseData data = GenerateTestcaseData(config, block_size, decoding_stage_rate);
			printf("  Testing block_size = %ld\n", block_size);
			if (config.num_reqs <= 16) {
				printf("  req lengths: [");
				for (auto x : data.input_lens) {
					printf("%ld, ", x);
				}
				printf("\b\b]\n");
			}
			Tensor ref;
			st::reference::kernel::attentionKernelRef(
				ref,
				data.k_cache,
				data.v_cache,

				data.qkvs,
				data.qk_scale,
				torch::from_blob(data.block_table.data(), {data.batch_size, data.max_num_block_per_seq}, torch::kInt64),
				torch::from_blob(data.input_lens.data(), {data.batch_size}, torch::kInt64),
				torch::from_blob(data.is_context_stage.data(), {data.batch_size}, torch::kInt64),

				run_context_stage,
				run_decoding_stage
			);

			Tensor answer = torch::zeros({data.num_tokens, data.num_q_heads, data.head_dim}, torch::kCUDA);
			if (run_context_stage)
				runFusedContextStageAttentionKernel<T>(answer, data);
			if (run_decoding_stage)
				runFusedDecodingStageAttentionKernel<T>(answer, data);
			if (run_kvcache_copy) {
				bool is_success = checkKVCacheCopyKernel<T>(data);
				ASSERT_TRUE(is_success);
			}

			bool is_passed = isArrayAlmostEqual(
				(T*)answer.data_ptr(),
				(T*)ref.data_ptr(),
				data.num_tokens * data.num_q_heads * data.head_dim,
				true,
				true
			);
			ASSERT_TRUE(is_passed);
		}
	}
}

TYPED_TEST(FusedAttentionTestSuite, DecodingStageOnlyTest) {
	typedef TypeParam T;
	runAllTestcases<T>(0.99, false, true, false);
}

TYPED_TEST(FusedAttentionTestSuite, ContextStageOnlyTest) {
	typedef TypeParam T;
	runAllTestcases<T>(0.05, true, false, false);
}

TYPED_TEST(FusedAttentionTestSuite, BothStagesTest) {
	typedef TypeParam T;
	runAllTestcases<T>(0.9, true, true, false);
}

TYPED_TEST(FusedAttentionTestSuite, KVCacheCopyTest) {
	typedef TypeParam T;
	runAllTestcases<T>(0.9, false, false, true);
}
