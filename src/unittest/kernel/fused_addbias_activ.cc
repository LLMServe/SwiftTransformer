#include <random>
#include <vector>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/fused_addbias_activ.h"

template<typename T>
class FusedAddbiasTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(FusedAddbiasTestSuite, SupportTypes);

TYPED_TEST(FusedAddbiasTestSuite, FusedAddbiasBatchedActivTest) {
	typedef TypeParam T;
	using st::ActivationType;

	const int64_t BATCH_SIZE = 243;
	const int64_t SIZE = 1890;

	torch::Tensor input = torch::rand({BATCH_SIZE, SIZE}, torch::kCUDA);
	torch::Tensor bias = torch::rand({SIZE}, torch::kCUDA);

	for (ActivationType activation_type : std::vector<ActivationType>{
		ActivationType::RELU,
		ActivationType::SILU,
		ActivationType::GELU
	}) {
		torch::Tensor ref_output = 
			activation_type == ActivationType::RELU ? torch::relu(input + bias) :
			activation_type == ActivationType::SILU ? torch::silu(input + bias) :
			activation_type == ActivationType::GELU ? torch::gelu(input + bias) :
			input + bias;
		torch::Tensor ans_output = torch::empty({BATCH_SIZE, SIZE}, torch::kCUDA);
		st::kernel::fusedAddbiasBatchedActivation(
			(T*)ans_output.data_ptr(),
			(T*)input.data_ptr(),
			(T*)bias.data_ptr(),
			BATCH_SIZE,
			SIZE,
			activation_type
		);
		sync_check_cuda_error_force();

		bool is_pass = isArrayAlmostEqual((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), BATCH_SIZE*SIZE, true, true);
		ASSERT_TRUE(is_pass);
	}
}
