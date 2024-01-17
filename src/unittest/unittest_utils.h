#pragma once

#include <random>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <map>

// Types for TYPED_TEST
typedef testing::Types<float, half> FloatAndHalfTypes;
#ifndef ENABLE_BF16
	typedef FloatAndHalfTypes SupportTypes;
#else
	typedef testing::Types<float, half, __nv_bfloat16> FloatHalfBf16Types;
	typedef FloatHalfBf16Types SupportTypes;
#endif


// Helpers for random number generating
// Generate a random integer from interval [min, max]
template<typename T>
inline T randNumber(std::mt19937 &rng, T min, T max) {
	std::uniform_int_distribution<T> dist(min, max);
	return dist(rng);
}

// isAlmostEqual - Check if two floats are equal under given precision
// It accomplishes this by checking whether fabs(answer-reference) <= ans_tol + rel_tol*fabs(reference)
// When both answer & reference are NaN, return true
inline bool isFloatAlmostEqual(float answer, float reference, const float abs_tol, const float rel_tol) {
	if (std::isnan(answer) && std::isnan(reference)) {
		return true;
	}
	if (std::isnan(answer) || std::isnan(reference)) {
		return false;
	}
	return fabs(answer-reference) <= abs_tol + rel_tol*fabs(reference);
}


// isArrayAlmostEqual - Check whether two float/half arrays are equal
// Precisions are:
// 		- abs_tol = 1e-4, rel_tol = 1e-2 for Float (FP32)
//		- abs_tol = 1e-3, rel_tol = 1e-1 for Half (FP16) and bfloat16
// If answer[] or reference[] is on device, please set is_answer_on_device or is_reference_on_device to true
template<typename T>
inline bool isArrayAlmostEqual(
	const T* answer_ptr, const T* reference_ptr, const int64_t n,
	const bool is_answer_on_device, const bool is_reference_on_device,
	const float max_allow_unmatch = -1,
	const bool record_pos = false
) {
	bool is_fp32 = std::is_same<T, float>::value;
	float abs_tol = is_fp32 ? 1e-4f : 1e-3f;
    float rel_tol = is_fp32 ? 1e-2f : 1e-1f;
	int64_t max_non_match = max_allow_unmatch != -1 ? max_allow_unmatch*n : (is_fp32 ? 0.002*n : 0.01*n);	// Allow up to 0.2% mismatch for FP32, 1% for FP16/bfloat16

	// Copy the array to host if necessary
	T* answer = (T*)answer_ptr;
	if (is_answer_on_device) {
		answer = (T*)malloc(n*sizeof(T));
		if (!answer) {
			printf("Failed to allocate memory for answer in isArrayAlmostEqual\n");
			assert(false);
		}
		cudaMemcpy(answer, answer_ptr, n*sizeof(T), cudaMemcpyDeviceToHost);
	}
	T* reference = (T*)reference_ptr;
	if (is_reference_on_device) {
		reference = (T*)malloc(n*sizeof(T));
		if (!reference) {
			printf("Failed to allocate memory for reference in isArrayAlmostEqual\n");
			assert(false);
		}
		cudaMemcpy(reference, reference_ptr, n*sizeof(T), cudaMemcpyDeviceToHost);
	}

	// Compare the two arrays, and output the difference
	std::vector<int64_t> error_pos;
	int64_t error_count = 0;
	int64_t first_error_pos = -1;
	for (int64_t i = 0; i < n; ++i) {
		bool ok = isFloatAlmostEqual(answer[i], reference[i], abs_tol, rel_tol);
		if (!ok) {
			if (record_pos){
				error_pos.push_back(i);
			}
			error_count += 1;
			if (error_count == 1) first_error_pos = i;
			if (error_count > max_non_match && error_count < max_non_match+4) {
				printf("Invalid result: answer[%ld] = %f, reference[%ld] = %f, abs_err = %f, rel_err = %f\n",
					i, (float)answer[i], i, (float)reference[i],
					fabs(answer[i]-reference[i]), fabs(answer[i]-reference[i])/fabs(reference[i]));
			}
		}
	}
	if (error_count != 0) {
		printf("Total %ld/%ld (%.2f%%) errors (1st error at pos #%ld)\n", error_count, n, 100.0*error_count/n, first_error_pos);
	}

	if (record_pos && error_count > max_non_match) {
		std::map<int64_t, int64_t> cnts;
		for (auto x: error_pos) {
			cnts[x] ++;
		}
		for (auto x: cnts) {
			printf("Error pos: %ld: %ld\n", x.first, x.second);
		}
		printf("\n");
	}

	// Free if necessary
	if (is_answer_on_device) {
		free(answer);
	}
	if (is_reference_on_device) {
		free(reference);
	}

	return error_count <= max_non_match;
}


// A tiny class for managing an array on GPU
// When being constructed, it allocates a space on GPU and copies the data to it.
// When being destructed, it frees the space on GPU.
// (Something likes std::unique_ptr)
template<typename T>
class GpuArray {
public:
	T* data;
	GpuArray(const std::vector<T> &host_data) {
		cudaMalloc(&data, host_data.size() * sizeof(T));
		cudaMemcpy(data, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice);
	}
	~GpuArray() {
		cudaFree(data);
	}
};
