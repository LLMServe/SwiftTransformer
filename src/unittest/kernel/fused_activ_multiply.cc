#include <random>
#include <vector>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/fused_activ_multiply.h"

template<typename T>
class FusedActivMultiplyTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(FusedActivMultiplyTestSuite, SupportTypes);

TYPED_TEST(FusedActivMultiplyTestSuite, FusedActivMultiplyTest) {
	typedef TypeParam T;
	using st::ActivationType;

	const int64_t SIZE = 1057362;

	torch::Tensor input1 = torch::rand({SIZE}, torch::kCUDA);
	torch::Tensor input2 = torch::rand({SIZE}, torch::kCUDA);

	for (ActivationType activation_type : std::vector<ActivationType>{
		ActivationType::RELU,
		ActivationType::SILU,
		ActivationType::GELU
	}) {
		torch::Tensor ref_output = (
			activation_type == ActivationType::RELU ? torch::relu(input1) :
			activation_type == ActivationType::SILU ? torch::silu(input1) :
			activation_type == ActivationType::GELU ? torch::gelu(input1) :
			input1 ) * input2;

		torch::Tensor ans_output = torch::empty({SIZE}, torch::kCUDA);
		st::kernel::fusedActivationMultiply(
			(T*)ans_output.data_ptr(),
			(T*)input1.data_ptr(),
			(T*)input2.data_ptr(),
			SIZE,
			activation_type
		);
		sync_check_cuda_error_force();

		bool is_pass = isArrayAlmostEqual((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), SIZE, true, true);
		ASSERT_TRUE(is_pass);
	}
}
