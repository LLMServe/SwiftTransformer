#include "ffn.h"

#include <cassert>
#include <iostream>

#include "util/cuda_utils.h"
#include "kernel/fused_activ_multiply.h"
#include "kernel/activation_types.h"
namespace st::layer {

// gatedFfn - FFN_GeXXX (e.g. FFN_GeGLU) with tensor parallelism
//
// Parallel parameters are passed by NcclComm struct
// 		- size: size of tensor parallel
// 		- rank: rank of current process
//		- comm: nccl communicator, initialized
//
// Weight addr should be prepared by the caller
//
// This layer takes input of shape [num_tokens, input_dim] and weights including:
//  - w1: [inter_dim / tensor_para_size, input_dim]
//  - w2: [output_dim, inter_dim / tensor_para_size]
//  - w3: [inter_dim / tensor_para_size, input_dim]
// (w1, w2, and w3 corresponds to https://github.com/facebookresearch/llama/blob/main/llama/model.py#L212C4-L212C4)
//
// The output is of shape [num_tokens, output_dim]
// output = (activation(input•w1^T) * (input•w3^T)) • w2^T, where • is matrix multiplication and * is element-wise multiplication

template<typename T>
void gatedFfn(
	T* output,		// [num_tokens, output_dim]
	T* input,		// [num_tokens, output_dim]

	T* w1_weight,	// [inter_dim / tensor_para_size, input_dim]
	T* w2_weight,	// [output_dim, inter_dim / tensor_para_size]
	T* w3_weight,	// [inter_dim / tensor_para_size, input_dim]

	int64_t num_tokens,
	int64_t input_dim,
	int64_t inter_dim,
	int64_t output_dim,
	ActivationType activation_type,

	T* inter_buf1,	// [num_tokens, inter_dim / tensor_para_size]
	T* inter_buf2,	// [num_tokens, inter_dim / tensor_para_size]

	util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
) {
	assert (inter_dim % nccl_comm.size == 0);

	// Calculate input • w1_T
	cublas_wrapper.gemm(
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		num_tokens,
		inter_dim / nccl_comm.size,
		input_dim,
		input,
		w1_weight,
		inter_buf1
	);
	sync_check_cuda_error();

	// Calculate input • w3_T
	cublas_wrapper.gemm(
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		num_tokens,
		inter_dim / nccl_comm.size,
		input_dim,
		input,
		w3_weight,
		inter_buf2
	);
	sync_check_cuda_error();
	
	// Calculate silu(input • w1_T) * (input • w3_T)
	st::kernel::fusedActivationMultiply(
		inter_buf1,
		inter_buf1,
		inter_buf2,
		num_tokens * (inter_dim / nccl_comm.size),
		activation_type
	);
	sync_check_cuda_error();

	// Calculate (silu(input • w1_T) * (input • w3_T)) • w2_T
	cublas_wrapper.gemm(
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		num_tokens,
		output_dim,
		inter_dim / nccl_comm.size,
		inter_buf1,
		w2_weight,
		output
	);
	sync_check_cuda_error();

	if (nccl_comm.size != 1) {
		st::util::stNcclAllReduce(
			output,
			output,
			num_tokens * output_dim,
			util::stNcclGetDataType<T>(),
			ncclSum,
			nccl_comm.comm,
			nccl_comm.stream
		);
		sync_check_cuda_error();
	}
}

#define INSTANTIAL_GATED_FFN(T) \
	template void gatedFfn( \
		T* output, \
		T* input, \
		T* fc1_weight, \
		T* fc2_weight, \
		T* fc3_weight, \
		int64_t num_tokens, \
		int64_t input_dim, \
		int64_t inter_dim, \
		int64_t output_dim, \
		ActivationType activation_type, \
		T* inter_buf1, \
		T* inter_buf2, \
		util::CublasWrapper cublas_wrapper, \
		util::NcclComm nccl_comm \
	);

INSTANTIAL_GATED_FFN(half)
INSTANTIAL_GATED_FFN(float)

}	// namespace st::layer