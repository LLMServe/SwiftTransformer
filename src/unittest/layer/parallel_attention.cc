#include <cassert>
#include <iostream>
#include <random>

#include <torch/torch.h>
#include <torch/all.h>
#include <gtest/gtest.h>

#include "attention_utils.h"
#include "util/nccl_utils.h"

#include "attention_ref.h"

#define print_line_number() std::cerr << __LINE__ << std::endl;

const int64_t BATCH_SIZE = 121;
const int64_t NUM_LAYERS = 3;
const int64_t LAYER_ID = 1;
const int64_t NUM_Q_HEADS = 8;
const int64_t NUM_KV_HEADS = 8;
const int64_t HEAD_DIM = 64;
const int64_t HIDDEN_SIZE = NUM_Q_HEADS*HEAD_DIM;
const int64_t MAX_INPUT_LEN = 64;
const int64_t BLOCK_SIZE = 16;

template<typename T>
class ParaAttentionTestSuite : public ::testing::Test {
public:
	st::util::CublasWrapper cublas_wrapper;

    int mpi_init_flag;

	const PagedAttnParam pagedattn_param = {
		.block_size = BLOCK_SIZE,
		.max_num_block_per_req = 8,
	};

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
        //     MPI_Finalize();
        // }
	}
};

TYPED_TEST_SUITE(ParaAttentionTestSuite, SupportTypes);

TYPED_TEST(ParaAttentionTestSuite, ParaAttentionTest) {
	typedef TypeParam T;
	std::mt19937 gen(0);

    int rank, world_size;

	st::util::CublasWrapper cublas_wrapper;
    st::util::NcclComm nccl_comm;

	// Get rank and assign devices
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

	assert (NUM_Q_HEADS % world_size == 0);
	assert (NUM_KV_HEADS % world_size == 0);
	assert (NUM_Q_HEADS % NUM_KV_HEADS == 0);
	const int64_t local_q_head_num = NUM_Q_HEADS / world_size;
	const int64_t local_kv_head_num = NUM_KV_HEADS / world_size;

    // Weights
	// Shared among all tests
	// We use +-1.0/HIDDEN_SIZE here in order to avoid overflow
	torch::Tensor qkv_weight_kernel = getRandomTensor({HIDDEN_SIZE, NUM_Q_HEADS+2*NUM_KV_HEADS, HEAD_DIM}, -1.0/HIDDEN_SIZE, 1.0/HIDDEN_SIZE, torch::kCPU);
	torch::Tensor qkv_weight_bias = getRandomTensor({NUM_Q_HEADS+2*NUM_KV_HEADS, HEAD_DIM}, -1, 1, torch::kCPU);
	torch::Tensor out_weight_kernel = getRandomTensor({NUM_Q_HEADS, HEAD_DIM, HIDDEN_SIZE}, -1.0/HIDDEN_SIZE, 1.0/HIDDEN_SIZE, torch::kCPU);
	torch::Tensor out_weight_bias = getRandomTensor({HIDDEN_SIZE}, -1, 1, torch::kCPU);
    MPI_Bcast(qkv_weight_kernel.data_ptr(), HIDDEN_SIZE * (NUM_Q_HEADS+2*NUM_KV_HEADS) * HEAD_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(qkv_weight_bias.data_ptr(), (NUM_Q_HEADS+2*NUM_KV_HEADS) * HEAD_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(out_weight_kernel.data_ptr(), HIDDEN_SIZE * HIDDEN_SIZE * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(out_weight_bias.data_ptr(), HIDDEN_SIZE * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);

	// qkv_weight: split by dim 2
	torch::Tensor qkv_weight_kernel_local = torch::cat({
		qkv_weight_kernel.slice(1, rank*local_q_head_num, (rank+1)*local_q_head_num),
		qkv_weight_kernel.slice(1, NUM_Q_HEADS+rank*local_kv_head_num, NUM_Q_HEADS+(rank+1)*local_kv_head_num),
		qkv_weight_kernel.slice(1, NUM_Q_HEADS+NUM_KV_HEADS+rank*local_kv_head_num, NUM_Q_HEADS+NUM_KV_HEADS+(rank+1)*local_kv_head_num)
	}, 1);
	torch::Tensor qkv_weight_bias_local = torch::cat({
		qkv_weight_bias.slice(0, rank*local_q_head_num, (rank+1)*local_q_head_num),
		qkv_weight_bias.slice(0, NUM_Q_HEADS+rank*local_kv_head_num, NUM_Q_HEADS+(rank+1)*local_kv_head_num),
		qkv_weight_bias.slice(0, NUM_Q_HEADS+NUM_KV_HEADS+rank*local_kv_head_num, NUM_Q_HEADS+NUM_KV_HEADS+(rank+1)*local_kv_head_num)
	}, 0);
	// out_weight: split by dim 1
	torch::Tensor out_weight_kernel_local = torch::split(out_weight_kernel, local_q_head_num, 0)[rank].clone();

    qkv_weight_kernel = qkv_weight_kernel.to(torch::kCUDA);
    qkv_weight_bias = qkv_weight_bias.to(torch::kCUDA);
    out_weight_kernel = out_weight_kernel.to(torch::kCUDA);
    out_weight_bias = out_weight_bias.to(torch::kCUDA);

	qkv_weight_kernel_local = qkv_weight_kernel_local.to(torch::kCUDA);
	qkv_weight_bias_local = qkv_weight_bias_local.to(torch::kCUDA);
	out_weight_kernel_local = out_weight_kernel_local.to(torch::kCUDA);

	// Generate input data
	torch::Tensor input_len_cpu = torch::zeros({BATCH_SIZE}, torch::kCPU).to(torch::kInt64);
    auto input_len_cpu_a = input_len_cpu.accessor<int64_t, 1>();
	int64_t num_tokens = 0;

    torch::Tensor input_len_gpu, is_context_stage_cpu, input;

    if (rank == 0){
        for (int64_t i = 0; i < BATCH_SIZE; ++i) {
		    input_len_cpu_a[i] = gen() % MAX_INPUT_LEN + 1;
		    num_tokens += input_len_cpu_a[i];
	    }
		// We use +-1.0/HIDDEN_SIZE here in order to avoid overflow
	    input = getRandomTensor({num_tokens, HIDDEN_SIZE}, -1.0/HIDDEN_SIZE, 1.0/HIDDEN_SIZE, torch::kCPU);
    }
    MPI_Bcast(&num_tokens, sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0){
        input = torch::zeros({num_tokens, HIDDEN_SIZE}, torch::kCPU);
    }

    MPI_Bcast(input_len_cpu.data_ptr(), BATCH_SIZE * sizeof(int64_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(input.data_ptr(), num_tokens * HIDDEN_SIZE * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);

    input = input.to(torch::kCUDA);
    input_len_gpu = input_len_cpu.to(torch::kCUDA);
	
	is_context_stage_cpu = torch::full({BATCH_SIZE}, true, torch::kCPU);
	if (rank == 0) {
		int num_context_reqs = gen() % (BATCH_SIZE-2) + 2;
		for (int i = 0; i < num_context_reqs; ++i) {
			is_context_stage_cpu[i] = true;
		}
	}
	MPI_Bcast(is_context_stage_cpu.data_ptr(), BATCH_SIZE * sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

	int64_t max_context_req_len = 0, max_decoding_req_len = 0, num_context_req = 0, num_decoding_req = 0;
	for (int i = 0; i < BATCH_SIZE; ++i) {
		bool is_context_stage = is_context_stage_cpu[i].item<bool>();
		if (!is_context_stage) {
			max_decoding_req_len = std::max(max_decoding_req_len, input_len_cpu_a[i]);
			num_decoding_req += 1;
		} else {
			max_context_req_len = std::max(max_context_req_len, input_len_cpu_a[i]);
			num_context_req += 1;
		}
	}

	// Buffers for st::layer::attention
	T *ans_qkv_buf, *ans_attn_out_buf;
    CUDA_CHECK(cudaMalloc(&ans_qkv_buf, (num_tokens + 15) * (NUM_Q_HEADS+2*NUM_KV_HEADS) * HEAD_DIM * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&ans_attn_out_buf, num_tokens * local_q_head_num * HEAD_DIM * sizeof(T)));

	// Allocate output tensor
	torch::Tensor ref_output = torch::zeros({num_tokens, HIDDEN_SIZE}, torch::kCUDA);
	torch::Tensor ref_k_cache = getRandomTensor({NUM_TOTAL_BLOCKS, NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM}, -1, 1, torch::kCPU);
	torch::Tensor ref_v_cache = getRandomTensor({NUM_TOTAL_BLOCKS, NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM}, -1, 1, torch::kCPU);
	
	// Broadcast ans_k_cache and ans_v_cache
	MPI_Bcast(ref_k_cache.data_ptr(), NUM_TOTAL_BLOCKS * NUM_LAYERS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(ref_v_cache.data_ptr(), NUM_TOTAL_BLOCKS * NUM_LAYERS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);

	ref_k_cache = ref_k_cache.to(torch::kCUDA);
	ref_v_cache = ref_v_cache.to(torch::kCUDA);
	
	torch::Tensor ans_output = torch::zeros({num_tokens, HIDDEN_SIZE}, torch::kCUDA);
	torch::Tensor ans_k_cache = torch::split(ref_k_cache, local_kv_head_num, 2)[rank].clone();
	torch::Tensor ans_v_cache = torch::split(ref_v_cache, local_kv_head_num, 2)[rank].clone();

	auto float_kernel_config = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

	torch::Tensor context_stage_kernel_m_buf_ans = torch::zeros({local_q_head_num, num_tokens}, float_kernel_config);
	torch::Tensor context_stage_kernel_l_buf_ans = torch::zeros({local_q_head_num, num_tokens}, float_kernel_config);

	// Build block table
	torch::Tensor block_table = torch::zeros({BATCH_SIZE, this->pagedattn_param.max_num_block_per_req}, torch::kInt64).to(torch::kCUDA);
	build_block_table(
		(int64_t*) block_table.data_ptr(),
	 	BATCH_SIZE,
		this->pagedattn_param,
		(int64_t*) input_len_cpu.data_ptr()
	);

	Indexes index = get_req_index(
		BATCH_SIZE,
		(int64_t*) input_len_cpu.data_ptr(),
		(bool*) is_context_stage_cpu.data_ptr()
	);
    index.toGPU();

	// generate ref_output
	st::reference::layer::attentionLayerRef(
		ref_output,
		ref_k_cache,
		ref_v_cache,
		input,
		input_len_cpu,
		is_context_stage_cpu,
		block_table.to(torch::kCPU),
		1.0f / sqrtf(HEAD_DIM),
		qkv_weight_kernel,
		qkv_weight_bias,
		out_weight_kernel,
		out_weight_bias,
		LAYER_ID
	);

	// Generate ans
	st::layer::attention(
        (T*)ans_output.data_ptr(),
		(T*)ans_k_cache.data_ptr(),
		(T*)ans_v_cache.data_ptr(),

		(T*)input.data_ptr(),
		(int64_t*)input_len_gpu.data_ptr(),
		(bool*)is_context_stage_cpu.data_ptr(),
		(int64_t*)block_table.data_ptr(),
		nullptr,

		num_context_req,
		num_decoding_req,
		index.ith_context_req_req_index,
		index.ith_context_req_token_index,
		index.ith_decoding_req_req_index,
		index.ith_decoding_req_token_index,
		max_context_req_len,
		max_decoding_req_len,

		(T*)qkv_weight_kernel_local.data_ptr(),
		(T*)qkv_weight_bias_local.data_ptr(),
		(T*)out_weight_kernel_local.data_ptr(),
		(T*)out_weight_bias.data_ptr(),

		BATCH_SIZE,
		num_tokens,
		HIDDEN_SIZE,
		NUM_LAYERS,
		NUM_Q_HEADS,
		NUM_KV_HEADS,
		HEAD_DIM,
		false,
		LAYER_ID,
		this->pagedattn_param.max_num_block_per_req,
		this->pagedattn_param.block_size,

		ans_qkv_buf,
		ans_attn_out_buf,

		(float*) context_stage_kernel_m_buf_ans.data_ptr(),
		(float*) context_stage_kernel_l_buf_ans.data_ptr(),
		this->cublas_wrapper,
        nccl_comm
    );
	
	auto splited_kcache = torch::split(ref_k_cache, local_kv_head_num, 2);
	torch::Tensor ref_k_cache_local = splited_kcache[rank].clone();
	torch::Tensor ref_v_cache_local = torch::split(ref_v_cache, local_kv_head_num, 2)[rank].clone();

	// Compare
	bool is_passed;
	
	is_passed = isArrayAlmostEqual(
		(T*)ans_k_cache.data_ptr(),
		(T*)ref_k_cache_local.data_ptr(),
		NUM_TOTAL_BLOCKS * NUM_LAYERS * local_kv_head_num * BLOCK_SIZE * HEAD_DIM,
		true,
		true
	);
	EXPECT_TRUE(is_passed);

	is_passed = isArrayAlmostEqual(
		(T*)ans_v_cache.data_ptr(),
		(T*)ref_v_cache_local.data_ptr(),
		NUM_TOTAL_BLOCKS * NUM_LAYERS * local_kv_head_num * BLOCK_SIZE * HEAD_DIM,
		true,
		true
	);
	EXPECT_TRUE(is_passed);
	
	is_passed = isArrayAlmostEqual(
		(T*)ans_output.data_ptr(),
		(T*)ref_output.data_ptr(),
		num_tokens * HIDDEN_SIZE,
		true,
		true
	);
	EXPECT_TRUE(is_passed);

	// Free up buffers
	CUDA_CHECK(cudaFree(ans_qkv_buf));
	CUDA_CHECK(cudaFree(ans_attn_out_buf));
}
