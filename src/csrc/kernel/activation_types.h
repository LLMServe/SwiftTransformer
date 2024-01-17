#pragma once

/*
	activation_types.h & activations.cuh

	Activation functions for neural networks.

	We put the definition of ActivationType in activtion_types.h and the
	implementation of the activation functions in activations.cuh. This is
	because we want the activation functions to be inlined into our kernels 
	(so we need to prepend `__forceinline__ __device__` to the function), while
	we want to be able to use the ActivationType enum in both the host and device.

	If you are writing a kernel that uses an activation function, you should
	include activation_types.h and activations.cuh in your kernel file. If
	you are writing host code that only use the ActivationType enum, you should
	only include activation_types.h.
*/

namespace st {
	namespace kernel {
		enum class ActivationType {
			RELU,
			SILU,
			GELU
		};
	}
	
	using ActivationType = kernel::ActivationType;
}
