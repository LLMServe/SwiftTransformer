#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/softmax.h"

template<typename T>
class SoftmaxTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(SoftmaxTestSuite, SupportTypes);

TYPED_TEST(SoftmaxTestSuite, ScaleMaskSoftmax) {
	typedef TypeParam T;

	const int64_t NUM_HEADS = 128;
	const int64_t INPUT_LEN = 556;
	const float scale = 1.0 / sqrt(INPUT_LEN);
	
	torch::Tensor input = getRandomTensor({NUM_HEADS, INPUT_LEN, INPUT_LEN});
	torch::Tensor ans_output = torch::zeros_like(input);
	torch::Tensor ref_output = input.to(torch::kFloat) * scale;

	st::kernel::scaleMaskSoftmax(
		(T*)ans_output.data_ptr(),
		(T*)input.data_ptr(),
		scale,
		NUM_HEADS,
		INPUT_LEN
	);

	torch::Tensor attn_mask = torch::zeros({INPUT_LEN, INPUT_LEN}, torch::kInt64);
	for (int64_t i = 0; i < INPUT_LEN; ++i) {
		for (int64_t j = i+1; j < INPUT_LEN; ++j) {
			attn_mask.accessor<int64_t, 2>()[i][j] = -10000;
		}
	}
	ref_output = ref_output + attn_mask.to(std::is_same<T, half>() ? at::kHalf : at::kFloat).to(torch::kCUDA);
	ref_output = torch::softmax(ref_output, 2);
	ref_output = ref_output.to(std::is_same<T, half>() ? at::kHalf : at::kFloat);

	bool is_passed = isArrayAlmostEqual((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), NUM_HEADS*INPUT_LEN*INPUT_LEN, true, true);
	ASSERT_TRUE(is_passed);
}

TYPED_TEST(SoftmaxTestSuite, ScaleSoftmax) {
	typedef TypeParam T;

	const int64_t NUM_HEADS = 128;
	const int64_t INPUT_LEN = 1950;
	const float scale = 1.0 / sqrt(INPUT_LEN);

	torch::Tensor input = getRandomTensor({NUM_HEADS, INPUT_LEN});
	torch::Tensor ans_output = torch::zeros_like(input);

	st::kernel::scaleSoftmax(
		(T*)ans_output.data_ptr(),
		(T*)input.data_ptr(),
		scale,
		NUM_HEADS,
		INPUT_LEN
	);

	torch::Tensor ref_output = input.to(torch::kFloat) * scale;
	ref_output = torch::softmax(ref_output, 1);
	ref_output = ref_output.to(std::is_same<T, half>() ? at::kHalf : at::kFloat);

	bool is_passed = isArrayAlmostEqual((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), NUM_HEADS*INPUT_LEN, true, true);
	ASSERT_TRUE(is_passed);
}