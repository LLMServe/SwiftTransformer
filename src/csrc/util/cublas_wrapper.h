/*
	cublas_wrapper.h - cublas GEMM wrapper

	This file contains CublasWrapper, a wrapper for cuBLAS functions (mainly GEMM).

	We wrap cublas mainly for the following reasons:
		- C++ stores matrixes in row-major order, while cuBLAS accepts matrix in column-major order,
		  which is confusing and error-prone. So in this wrapper, every input matrix is ROW-MAJOR ORDER.
		- StridedBatchedGEMM in cuBLAS contains some seldom-used parameters, which makes the interface
		  of cuBLAS very complicated. So we wrap it to make it easier to use.
		- cuBLAS supports many algorithms for GEMM, and some of them are faster than the default one.
		  Currently it is using the default algo (CUBLAS_GEMM_DEFAULT) to do GEMM, in the
	      future we may run a small benchmark ahead of time, and then pick the fastest algo.
*/

#pragma once

#include <memory>
#include <stdexcept>

#include <cublas_v2.h>

#define checkCublasStatus(status) checkCublasStatus_line((status), __FILE__, __LINE__)

namespace st::util {

class CublasWrapper {
private:
	// Need to use a shared_ptr here, or when passing CublasWrapper to a function
	// and the function returns, the destructor of CublasWrapper will be called,
	// which destorys the handle, and then the handle in the function will be invalid.
	std::shared_ptr<cublasHandle_t> handle_;	
	cublasGemmAlgo_t algo_;

public:
	CublasWrapper():
		handle_(std::make_shared<cublasHandle_t>()) {
		cublasCreate(handle_.get());
		algo_ = CUBLAS_GEMM_DEFAULT;
	}

	~CublasWrapper() {
		if (handle_.use_count() == 1) {
			// I am the last one who uses the handle, so I should destroy it
			cublasDestroy(*handle_.get());
		}
	}

	/*
		gemmStridedBatched - Calculate C = A @ B

		PLEASE KEEP IN MIND THAT A, B AND C SHOULD BE STORED IN ROW MAJOR!

		The size of A is (m, k) (or (k, m) if transa is CUBLAS_OP_T)
		The size of B is (k, n) (or (n, k) if transb is CUBLAS_OP_T)
		The size of C is (m, n)

		stride: How much elements between two adjacent matrices in A, B and C
		In detail, we consider Aarray[0:m*k] as the first matrix in A,
		Aarray[stride_a+m*k : stride_a+2*m*k] as the second matrix in A, and so on.
		The same for stride_b and stride_c.

		batchCount: How many matrices in A, B and C

		alpha and beta: The same as cuBLAS. C = alpha*(A@B) + beta*C
	*/
	template<typename T>
	void gemmStridedBatched(
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const T alpha,
		const T* Aarray,
		long long int stride_a,
		const T* Barray,
		long long int stride_b,
		const T beta,
		T* Carray,
		long long int stride_c,
		int batchCount
	);

	/*
		gemmStridedBatched - A simplified version of gemmStridedBatched
		It omits alpha (set to 1) and beta (set to 0).
	*/
	template<typename T>
	void gemmStridedBatched(
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const T* Aarray,
		long long int stride_a,
		const T* Barray,
		long long int stride_b,
		T* Carray,
		long long int stride_c,
		int batchCount
	);
	
	/*
		gemmBatched - A simplified version of gemmStridedBatched
		It omits stride_a, stride_b and stride_c by assuming that
		A, B and C are stored continuously in memory.
	*/
	template<typename T>
	void gemmBatched(
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const T* Aarray,
		const T* Barray,
		T* Carray,
		int batchCount
	);

	/*
		gemm - Calculate the product of two matrixes
		Its function is exactly gemmBatched + batchCount=1
	*/
	template<typename T>
	void gemm(
		cublasOperation_t transa,
		cublasOperation_t transb,
		int m,
		int n,
		int k,
		const T* Aarray,
		const T* Barray,
		T* Carray
	);
};

template<typename T>
cublasComputeType_t getCublasComputeType() {
	if (std::is_same<T, half>::value) {
		return CUBLAS_COMPUTE_16F;
	}
	if (std::is_same<T, float>::value) {
		return CUBLAS_COMPUTE_32F; // TODO(sunyh): Maybe try CUBLAS_COMPUTE_32F_FAST_16F?
	}
	if (std::is_same<T, double>::value) {
		return CUBLAS_COMPUTE_64F;
	}
	if (std::is_same<T, int>::value) {
		return CUBLAS_COMPUTE_32I;
	}
	throw std::runtime_error("Cublas compute type: Unsupported type");
}

}	// namespace st::kernel