#pragma once

#include "kernel/addbias.h"
#include "kernel/fused_addbias_activ.h"
#include "util/cublas_wrapper.h"

#include "util/nccl_utils.h"

namespace st::layer {

template<typename T>
void ffn(
	T* output,
	T* input,

	T* fc1_weight,
	T* fc1_bias,
	T* fc2_weight,
	T* fc2_bias,

	int64_t batch_size,
	int64_t input_dim,
	int64_t inter_dim,
	int64_t output_dim,
	ActivationType activation_type,

	T* inter_buf,

	util::CublasWrapper cublas_wrapper,
	util::NcclComm nccl_comm
);

}	// namespace st::layer