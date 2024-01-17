#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/findmax.h"

template<typename T>
class FindmaxTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(FindmaxTestSuite, SupportTypes);

TYPED_TEST(FindmaxTestSuite, FindmaxBatchedTest) {
	typedef TypeParam T;

	const int64_t BATCH_SIZE = 1892;
	const int64_t LENGTH = 2345;
	torch::Tensor input = torch::rand({BATCH_SIZE, LENGTH}, torch::kCUDA);
	torch::Tensor ref_output = torch::argmax(input, 1).cpu();
	torch::Tensor ans_output = torch::empty({BATCH_SIZE}, torch::kInt64).cuda();
	st::kernel::findmaxBatched(
		(int64_t*)ans_output.data_ptr(),
		(T*)input.data_ptr(),
		BATCH_SIZE,
		LENGTH
	);
	ans_output = ans_output.cpu();

	for (int64_t i = 0; i < BATCH_SIZE; ++i) {
		T ref_max = ((T*)input[i].cpu().data_ptr())[ref_output[i].item<int64_t>()];
		T ans_max = ((T*)input[i].cpu().data_ptr())[ans_output[i].item<int64_t>()];
		ASSERT_EQ(ref_max, ans_max);
	}
}
