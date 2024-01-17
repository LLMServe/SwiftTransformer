#include "cublas_wrapper.h"

#include <cassert>
#include <iostream>

#include "util/cuda_utils.h"

namespace st::util {

template<typename T>
void CublasWrapper::gemmStridedBatched(
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
) {
	cudaDataType_t cuda_datatype = getCudaDataType<T>();
	cublasComputeType_t compute_type = getCublasComputeType<T>();
	cublasStatus_t status = cublasGemmStridedBatchedEx(
		*handle_.get(),
		transb,
		transa,
		n,
		m,
		k,
		&alpha,
		Barray,
		cuda_datatype,
		transb == CUBLAS_OP_N ? n : k,
		stride_b,
		Aarray,
		cuda_datatype,
		transa == CUBLAS_OP_N ? k : m,
		stride_a,
		&beta,
		Carray,
		cuda_datatype,
		n,
		stride_c,
		batchCount,
		compute_type,
		algo_
	);
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "CublasWrapper::gemmStridedBatched failed: " << status << std::endl;
		throw std::runtime_error("CublasWrapper::gemmStridedBatched failed");
	}
}

template void CublasWrapper::gemmStridedBatched(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const half alpha,
	const half* Aarray, long long int stride_a,
	const half* Barray, long long int stride_b,
	const half beta,
	half* Carray, long long int stride_c,
	int batchCount
);
template void CublasWrapper::gemmStridedBatched(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const float alpha,
	const float* Aarray, long long int stride_a,
	const float* Barray, long long int stride_b,
	const float beta,
	float* Carray, long long int stride_c,
	int batchCount
);

template<typename T>
void CublasWrapper::gemmStridedBatched(
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
) {
	gemmStridedBatched(
		transa,
		transb,
		m,
		n,
		k,
		(T)1.0,
		Aarray,
		stride_a,
		Barray,
		stride_b,
		(T)0.0,
		Carray,
		stride_c,
		batchCount
	);
}

template void CublasWrapper::gemmStridedBatched(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const half* Aarray, long long int stride_a,
	const half* Barray, long long int stride_b,
	half* Carray, long long int stride_c,
	int batchCount
);
template void CublasWrapper::gemmStridedBatched(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const float* Aarray, long long int stride_a,
	const float* Barray, long long int stride_b,
	float* Carray, long long int stride_c,
	int batchCount
);

template<typename T>
void CublasWrapper::gemmBatched(
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	const T* Aarray,
	const T* Barray,
	T* Carray,
	int batchCount
) {
	gemmStridedBatched(
		transa,
		transb,
		m,
		n,
		k,
		(T)1.0,
		Aarray,
		1LL*m*k,
		Barray,
		1LL*n*k,
		(T)0.0,
		Carray,
		1LL*m*n,
		batchCount
	);
}

template void CublasWrapper::gemmBatched(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const half* Aarray, const half* Barray, half* Carray,
	int batchCount
);
template void CublasWrapper::gemmBatched(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const float* Aarray, const float* Barray, float* Carray,
	int batchCount
);

template<typename T>
void CublasWrapper::gemm(
	cublasOperation_t transa,
	cublasOperation_t transb,
	int m,
	int n,
	int k,
	const T* Aarray,
	const T* Barray,
	T* Carray
) {
	gemmBatched(
		transa,
		transb,
		m,
		n,
		k,
		Aarray,
		Barray,
		Carray,
		1
	);
}

template void CublasWrapper::gemm(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const half* Aarray, const half* Barray, half* Carray
);
template void CublasWrapper::gemm(
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const float* Aarray, const float* Barray, float* Carray
);

inline void checkCublasStatus_line(cublasStatus_t status, const char* file, int line){
	if (status != CUBLAS_STATUS_SUCCESS) {
		std::cerr << "cublasLt failed: " << status << " " << file << ":" << line << std::endl;
		throw std::runtime_error("cublasLt failed:");
		abort();
	}
}

}