#pragma once
#include <cstdint>
#include <cinttypes>

#include "kernel/activation_types.h"
namespace st::kernel {

template<typename T>
void fusedActivationMultiply(
	T* output,
	const T* input1,
	const T* input2,
	int64_t n,
	ActivationType activationType
);

}	// namespace st::kernel