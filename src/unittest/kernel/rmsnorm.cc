#include <random>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/rmsnorm.h"

template<typename T>
class RmsnormTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(RmsnormTestSuite, SupportTypes);

TYPED_TEST(RmsnormTestSuite, RmsnormTest) {
	typedef TypeParam T;

	const int64_t BATCH_SIZE = 259;
	const int64_t HIDDEN_SIZE = 40761;

	torch::Tensor input = torch::rand({BATCH_SIZE, HIDDEN_SIZE}, torch::kCUDA);
	torch::Tensor weight = torch::rand({HIDDEN_SIZE}, torch::kCUDA);
	const T epsilon = 1e-4;

	torch::Tensor ans_output = torch::empty({BATCH_SIZE, HIDDEN_SIZE}, torch::kCUDA);
	st::kernel::rmsnorm(
		(T*)ans_output.data_ptr(),
		(T*)input.data_ptr(),

		(T*)weight.data_ptr(),
		epsilon,

		BATCH_SIZE,
		HIDDEN_SIZE
	);

	torch::Tensor t = input.pow(2).mean(-1, true) + (float)epsilon;
	torch::Tensor ref_output = input * torch::rsqrt(t) * weight;

	bool is_passed = isArrayAlmostEqual((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), BATCH_SIZE * HIDDEN_SIZE, true, true);
	ASSERT_TRUE(is_passed);
}
