#include "ffn.h"

#include "util/cuda_utils.h"
#include "kernel/activation_types.h"

namespace st::layer {

// ffn - Feed forward network with tensor parallelism 
//
// Parallel parameters are passed by NcclComm struct
// 		- size: size of tensor parallel
// 		- rank: rank of current process
//		- comm: nccl communicator, initialized
//
// Weight addr should be prepared by the caller
//
// Architecture:
// 
// input [batch_size, input_dim]
// |
// | Linear 1
// | Weights:
// | 	- fc1_weight [inter_dim / tensor_para_size, input_dim]
// |	- fc1_bias [inter_dim / tensor_para_size]
// | inter = input * fc1_weight^T + fc1_bias
// | 
// V
// inter [batch_size, inter_dim / tensor_para_size]
// |
// | Activation
// | inter = max(inter, 0)
// |
// V
// inter [batch_size, inter_dim / tensor_para_size]
// |
// | Linear 2
// | Weights:
// |	- fc2_weight [output_dim, inter_dim / tensor_para_size]
// | ToReduce = inter * fc2_weight
// |
// V
// ToReduce [batch_size, output_dim]
// |
// V AllReduce
// | Weights:
// |	- fc2_bias [output_dim]
// | output = AllReduce(ToReduce) + fc2_bias

template<typename T>
void ffn(
	T* output,		// [batch_size, output_dim]
	T* input,		// [batch_size, input_dim]

	T* fc1_weight,	// [inter_dim / tensor_para_size, input_dim]
	T* fc1_bias,	// [inter_dim / tensor_para_size]
	T* fc2_weight,	// [output_dim, inter_dim / tensor_para_size]
	T* fc2_bias,	// [output_dim]

	int64_t batch_size,
	int64_t input_dim,
	int64_t inter_dim,
	int64_t output_dim,
	ActivationType activation_type,

	T* inter_buf,	// [batch_size, inter_dim / tensor_para_size]

	util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
){
	// Linear1
	cublas_wrapper.gemm(
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		batch_size,
		inter_dim / nccl_comm.size,
		input_dim,
		input,
		fc1_weight,
		inter_buf
	);
	sync_check_cuda_error();

	// Addbias & Relu
	// Use fused kernel to improve performance
	kernel::fusedAddbiasBatchedActivation(
		inter_buf,
		inter_buf,
		fc1_bias,
		batch_size,
		inter_dim / nccl_comm.size,
		activation_type
	);
	sync_check_cuda_error();

	// Linear2
	cublas_wrapper.gemm(
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		batch_size,
		output_dim,
		inter_dim / nccl_comm.size,
		inter_buf,
		fc2_weight,
		output
	);

	sync_check_cuda_error();

	if (nccl_comm.size != 1) {
		st::util::stNcclAllReduce(
			output,
			output,
			batch_size * output_dim,
			util::stNcclGetDataType<T>(),
			ncclSum,
			nccl_comm.comm,
			nccl_comm.stream
		);
	}

	sync_check_cuda_error();

	// Addbias
	kernel::addbiasBatched(output, output, fc2_bias, batch_size, output_dim);
	sync_check_cuda_error();
}

template void ffn(
	float* output, float* input,
	float* fc1_weight, float* fc1_bias,
	float* fc2_weight, float* fc2_bias,
	int64_t batch_size, int64_t input_dim, int64_t inter_dim, int64_t output_dim,
	ActivationType activation_type,
	float* inter_buf, util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
);

template void ffn(
	half* output, half* input,
	half* fc1_weight, half* fc1_bias,
	half* fc2_weight, half* fc2_bias,
	int64_t batch_size, int64_t input_dim, int64_t inter_dim, int64_t output_dim,
	ActivationType activation_type,
	half* inter_buf, util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
);

}	// namespace st::layer