#pragma once

#include "kernel/addbias.h"
#include "kernel/activation_types.h"
#include "util/cublas_wrapper.h"

#include "util/nccl_utils.h"
namespace st::layer {

template<typename T>
void gatedFfn(
	T* output,
	T* input,

	T* fc1_weight,
	T* fc2_weight,
	T* fc3_weight,

	int64_t num_tokens,
	int64_t input_dim,
	int64_t inter_dim,
	int64_t output_dim,
	ActivationType activation_type,

	T* inter_buf1,
	T* inter_buf2,

	util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
);

}	// namespace st::layer