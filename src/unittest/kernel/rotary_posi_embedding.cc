#include <random>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/rotary_posi_embedding.h"
#include "rotary_posi_embedding_ref.h"

template<typename T>
class RotaryPosiEmbeddingTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(RotaryPosiEmbeddingTestSuite, SupportTypes);

TYPED_TEST(RotaryPosiEmbeddingTestSuite, RotaryPosiEmbeddingTest) {
	typedef TypeParam T;
	std::mt19937 gen(0);

	const int64_t NUM_TOKENS = 243;
	const int64_t NUM_HEADS = 64;
	const int64_t HEAD_DIM = 128;

	torch::Tensor input = torch::rand({NUM_TOKENS, NUM_HEADS, HEAD_DIM}, torch::kCUDA);
	std::vector<int64_t> indexes(NUM_TOKENS);
	for (int64_t i = 0; i < NUM_TOKENS; ++i) {
		indexes[i] = gen() % NUM_TOKENS;
	}

	torch::Tensor ans_output = input.clone();
	GpuArray<int64_t> d_indexes(indexes);
	st::kernel::rotaryPosiEmbeddingBatched(
		(T*)ans_output.data_ptr(),
		d_indexes.data,
		NUM_TOKENS,
		NUM_HEADS,
		NUM_HEADS,
		HEAD_DIM
	);
	sync_check_cuda_error_force();

	torch::Tensor ref_output = input.clone();
	st::reference::kernel::rotaryPosiEmbeddingKernelRef(
		ref_output,
		indexes
	);
	sync_check_cuda_error_force();

	bool is_pass = isArrayAlmostEqual((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), NUM_TOKENS*NUM_HEADS*HEAD_DIM, true, true);
	sync_check_cuda_error();
	ASSERT_TRUE(is_pass);
}
