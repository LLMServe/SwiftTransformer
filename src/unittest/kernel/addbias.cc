#include <random>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "kernel/addbias.h"

template<typename T>
class AddbiasTestSuite : public ::testing::Test {
public:
	void SetUp() override {
		setupTorch<T>();
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(AddbiasTestSuite, SupportTypes);

TYPED_TEST(AddbiasTestSuite, AddbiasTest) {
	typedef TypeParam T;

	const int64_t ROUND = 16;
	for (int64_t i = 0; i < ROUND; ++i) {
		const int64_t N = 1024 + 2*i;	// N need to be even, proposed by addbias.cu
		torch::Tensor input = torch::rand({N}, torch::kCUDA);
		torch::Tensor bias = torch::rand({N}, torch::kCUDA);
		torch::Tensor ref_output = input + bias;
		torch::Tensor ans_output = torch::empty({N}, torch::kCUDA);

		st::kernel::addbias<T>(
			(T*)ans_output.data_ptr(),
			(T*)input.data_ptr(),
			(T*)bias.data_ptr(),
			N
		);
		sync_check_cuda_error();
		
		bool is_pass = isArrayAlmostEqual<T>((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), N, true, true);
		sync_check_cuda_error();
		ASSERT_TRUE(is_pass);
	}
}

TYPED_TEST(AddbiasTestSuite, AddbiasBatchedTest) {
	typedef TypeParam T;

	const int64_t BATCH_SIZE = 14;
	const int64_t N = 1024;
	torch::Tensor input = torch::rand({BATCH_SIZE, N}, torch::kCUDA);
	torch::Tensor bias = torch::rand({N}, torch::kCUDA);
	torch::Tensor ref_output = input + bias;
	torch::Tensor ans_output = torch::empty({BATCH_SIZE, N}, torch::kCUDA);
	
	st::kernel::addbiasBatched<T>(
		(T*)ans_output.data_ptr(),
		(T*)input.data_ptr(),
		(T*)bias.data_ptr(),
		BATCH_SIZE,
		N
	);
	sync_check_cuda_error();

	bool is_pass = isArrayAlmostEqual((T*)ans_output.data_ptr(), (T*)ref_output.data_ptr(), N*BATCH_SIZE, true, true);
	sync_check_cuda_error();
	ASSERT_TRUE(is_pass);	
}