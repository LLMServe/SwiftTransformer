#include <iostream>
#include <random>

#include <torch/torch.h>
#include <gtest/gtest.h>
#include <nccl.h>

#include "../unittest_utils.h"
#include "../unittest_torch_utils.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "util/cublas_wrapper.h"
#include "layer/ffn.h"
#include "layer/gated_ffn.h"
#include "kernel/activation_types.h"

template<typename T>
class ParaFfnTestSuite : public ::testing::Test {
public:
	int mpi_init_flag;
	void SetUp() override {
		setupTorch<T>();
        // Init mpi
		MPI_Initialized(&mpi_init_flag);
		if(!mpi_init_flag){
			auto mpi_error = MPI_Init(nullptr, nullptr);
			if (mpi_error != MPI_SUCCESS) {
				std::cerr << "MPI_Init failed" << std::endl;
				exit(-1);
			}
		}
        
	}
	void TearDown() override {
		// TODO(sunyh): Correct way to finalize mpi
		// if (mpi_init_flag){
		// 	MPI_Finalize();
		// }
	}
};

TYPED_TEST_SUITE(ParaFfnTestSuite, SupportTypes);

TYPED_TEST(ParaFfnTestSuite, ParaFfnTest) {
	typedef TypeParam T;

	int rank, world_size;

	st::util::CublasWrapper cublas_wrapper;
    st::util::NcclComm nccl_comm;

	// Get rank and assign devicec
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	EXPECT_TRUE(world_size <= deviceCount);

	cudaSetDevice(rank);
	torch::Device device(torch::kCUDA, rank);

	if (nccl_comm.comm == nullptr) {
		ncclUniqueId uid;
		if (rank == 0) {
			st::util::stNcclGetUniqueId(uid);
		}
		MPI_Bcast(uid.internal, NCCL_UNIQUE_ID_BYTES, MPI_BYTE, 0, MPI_COMM_WORLD);
		nccl_comm = st::util::stNcclInit(world_size, rank, uid);
	}


	const int64_t BATCH_SIZE = 2413;
	const int64_t INPUT_DIM = 128;
	const int64_t INTER_DIM = 256;
	const int64_t OUTPUT_DIM = 64;
    
    ASSERT_EQ(INPUT_DIM % 8, 0);
    const int64_t LOCAL_INTER_DIM = INTER_DIM / nccl_comm.size;

	auto data_type = torch::kFloat32;
	if (std::is_same<T, half>::value) {
		data_type = torch::kFloat16;
	}

	auto cpu_config = torch::TensorOptions().dtype(data_type).device(torch::kCPU);
	auto gpu_config = torch::TensorOptions().dtype(data_type).device(torch::kCUDA);

	T* inter_buf1, *inter_buf2;
	CUDA_CHECK(cudaMalloc(&inter_buf1, BATCH_SIZE * LOCAL_INTER_DIM * sizeof(T)));
	CUDA_CHECK(cudaMalloc(&inter_buf2, BATCH_SIZE * LOCAL_INTER_DIM * sizeof(T)));

	torch::Tensor input, weight1, bias1, weight2 ,bias2, weight3;

	if (nccl_comm.rank == 0) {
		input = torch::rand({BATCH_SIZE, INPUT_DIM}, cpu_config)/INPUT_DIM;
		// Here we divide input by INPUT_DIM, since in practice we will feed the FFN
		// with data that are just normalized. Otherwise FFN may overflow.
		weight1 = torch::rand({INTER_DIM, INPUT_DIM}, cpu_config);
		bias1 = torch::rand({INTER_DIM}, cpu_config);
		weight2 = torch::rand({OUTPUT_DIM, INTER_DIM}, cpu_config);
		bias2 = torch::rand({OUTPUT_DIM}, cpu_config);
		weight3 = torch::rand({INTER_DIM, INPUT_DIM}, cpu_config);
	}
	else {
		input = torch::empty({BATCH_SIZE, INPUT_DIM}, cpu_config);
		weight1 = torch::empty({INTER_DIM, INPUT_DIM}, cpu_config);
		bias1 = torch::empty({INTER_DIM}, cpu_config);
		weight2 = torch::empty({OUTPUT_DIM, INTER_DIM}, cpu_config);
		bias2 = torch::empty({OUTPUT_DIM}, cpu_config);
		weight3 = torch::empty({INTER_DIM, INPUT_DIM}, cpu_config);
	}

	MPI_Bcast(input.data_ptr(), BATCH_SIZE * INPUT_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(weight1.data_ptr(), INTER_DIM * INPUT_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(bias1.data_ptr(), INTER_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(weight2.data_ptr(), INTER_DIM * OUTPUT_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(bias2.data_ptr(), OUTPUT_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(weight3.data_ptr(), INTER_DIM * INPUT_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);

	// split and get local data
	auto weight1_local = torch::split(weight1, LOCAL_INTER_DIM, 0)[nccl_comm.rank];
	auto bias1_local = torch::split(bias1, LOCAL_INTER_DIM, 0)[nccl_comm.rank];
	auto weight2_local = torch::split(weight2, LOCAL_INTER_DIM, 1)[nccl_comm.rank];
	auto weight3_local = torch::split(weight3, LOCAL_INTER_DIM, 0)[nccl_comm.rank];

	// Move data from CPU to GPU
	input = input.to(torch::kCUDA, data_type);
	weight1 = weight1.to(torch::kCUDA, data_type);
	bias1 = bias1.to(torch::kCUDA, data_type);
	weight2 = weight2.to(torch::kCUDA, data_type);
	weight3 = weight3.to(torch::kCUDA, data_type);
	bias2 = bias2.to(torch::kCUDA, data_type);
	weight1_local = weight1_local.to(torch::kCUDA, data_type);
	bias1_local = bias1_local.to(torch::kCUDA, data_type);
	weight2_local = weight2_local.to(torch::kCUDA, data_type);
	weight3_local = weight3_local.to(torch::kCUDA, data_type);
	
	// Test the correctness of ffn with activation type = relu
	torch::Tensor reluffn_output = torch::empty({BATCH_SIZE, OUTPUT_DIM}, gpu_config);
	st::layer::ffn(
		(T*)reluffn_output.data_ptr(),
		(T*)input.data_ptr(),
		(T*)weight1_local.data_ptr(),
		(T*)bias1_local.data_ptr(),
		(T*)weight2_local.data_ptr(),
		(T*)bias2.data_ptr(),

		BATCH_SIZE,
		INPUT_DIM,
		INTER_DIM,
		OUTPUT_DIM,
		st::ActivationType::RELU,

		inter_buf1,
		cublas_wrapper,
        nccl_comm
	);
	sync_check_cuda_error_force();

	torch::Tensor reluffn_ref_inter = torch::relu(torch::matmul(input, weight1.transpose(0, 1))+ bias1);	// (BATCH_SIZE, INTER_DIM)
	torch::Tensor reluffn_ref_output = torch::matmul(reluffn_ref_inter, weight2.transpose(0, 1)) + bias2;
	bool is_pass = isArrayAlmostEqual((T*)reluffn_output.data_ptr(), (T*)reluffn_ref_output.data_ptr(), BATCH_SIZE * OUTPUT_DIM, true, true);
	ASSERT_TRUE(is_pass);

	// Test the correctness of gatedFfn with activation type = silu
	torch::Tensor gatedSiluFfn_output = torch::empty({BATCH_SIZE, OUTPUT_DIM}, gpu_config);
	st::layer::gatedFfn(
		(T*)gatedSiluFfn_output.data_ptr(),
		(T*)input.data_ptr(),
		(T*)weight1_local.data_ptr(),
		(T*)weight2_local.data_ptr(),
		(T*)weight3_local.data_ptr(),

		BATCH_SIZE,
		INPUT_DIM,
		INTER_DIM,
		OUTPUT_DIM,
		st::ActivationType::SILU,

		inter_buf1,
		inter_buf2,

		cublas_wrapper,
		nccl_comm
	);
	sync_check_cuda_error_force();
	
	torch::Tensor gatedSiluFfn_ref_inter = torch::silu(torch::matmul(input, weight1.transpose(0, 1))) *
		torch::matmul(input, weight3.transpose(0, 1));
	torch::Tensor gatedSiluFfn_ref_output = torch::matmul(gatedSiluFfn_ref_inter, weight2.transpose(0, 1));
	is_pass = isArrayAlmostEqual((T*)gatedSiluFfn_output.data_ptr(), (T*)gatedSiluFfn_ref_output.data_ptr(), BATCH_SIZE * OUTPUT_DIM, true, true);
	ASSERT_TRUE(is_pass);

	cudaFree(inter_buf1);
	cudaFree(inter_buf2);
	st::util::stNcclDestroy(nccl_comm);
}