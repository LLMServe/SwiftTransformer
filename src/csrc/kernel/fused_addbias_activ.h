#pragma once

#include <cstdint>
#include <cinttypes>

#include "activation_types.h"

namespace st::kernel {

template<typename T>
void fusedAddbiasBatchedActivation(
	T* output,
	const T* input,
	const T* bias,
	const int64_t batch,
	const int64_t size,
	ActivationType activation_type
);

}	// namespace st::kernel