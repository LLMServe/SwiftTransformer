#include <functional>
#include <random>

#include <gtest/gtest.h>

#include "../unittest_utils.h"
#include "util/cublas_wrapper.h"
#include "util/cuda_utils.h"


// naiveGemmStridedBatched - Perform StridedBatchedGEMM on CPU
// When transa = transb = CUBLAS_OP_N, each matrix in A has a shape of m x k,
// and each matrix in B has a shape of k x n, and the result matrix C has a shape of m x n
// There are totally batchCount matrices in A, B and C
// A, B and C are stored in row major

template<typename T>
void naiveGemmStridedBatched(
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
	int lda = transa == CUBLAS_OP_N ? k : m;
	int ldb = transb == CUBLAS_OP_N ? n : k;
	int ldc = n;
	for (int batch = 0; batch < batchCount; batch++) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				// We use float here since
				//	- It is more accurate when accumulating a_{i, k} b_{k, j}
				//	- It is faster (2x speedup) on CPU, so our test can run faster
				float sum = 0.0;
				for (int l = 0; l < k; l++) {
					T a_elem = transa == CUBLAS_OP_N ? Aarray[batch * stride_a + i * lda + l] : Aarray[batch * stride_a + l * lda + i];
					T b_elem = transb == CUBLAS_OP_N ? Barray[batch * stride_b + l * ldb + j] : Barray[batch * stride_b + j * ldb + l];
					sum = sum + (float)a_elem*(float)b_elem;
				}
				Carray[batch * stride_c + i * ldc + j] = alpha * sum + beta * Carray[batch * stride_c + i * ldc + j];
			}
		}
	}
}

template<typename T>
class CublasWrapperTestSuite : public ::testing::Test {
protected:
	st::util::CublasWrapper wrapper;

public:
	void SetUp() override {
	}
	void TearDown() override {
	}
};

TYPED_TEST_SUITE(CublasWrapperTestSuite, SupportTypes);

TYPED_TEST(CublasWrapperTestSuite, gemmStridedBatched) {
	typedef TypeParam T;
	const int M = 61;
	const int N = 29;
	const int K = 124;
	const int BATCH_COUNT = 97;

	const int STRIDE_A = M*K + 12;
	const int STRIDE_B = K*N + 9;
	const int STRIDE_C = M*N + 13;

	// Alloc arrays on CPU
	T* Aarray = new T[STRIDE_A * BATCH_COUNT];
	T* Barray = new T[STRIDE_B * BATCH_COUNT];
	T* ref_Carray = new T[STRIDE_C * BATCH_COUNT];

	// Alloc arrays on GPU
	T* Aarray_gpu, *Barray_gpu, *ans_Carray_gpu;
	CUDA_CHECK(cudaMalloc(&Aarray_gpu, STRIDE_A * BATCH_COUNT * sizeof(T)));
	CUDA_CHECK(cudaMalloc(&Barray_gpu, STRIDE_B * BATCH_COUNT * sizeof(T)));
	CUDA_CHECK(cudaMalloc(&ans_Carray_gpu, STRIDE_C * BATCH_COUNT * sizeof(T)));

	for (int is_transa = 0; is_transa <= 1; ++is_transa) {
		for (int is_transb = 0; is_transb <= 1; ++is_transb) {
			cublasOperation_t transa = is_transa ? CUBLAS_OP_T : CUBLAS_OP_N;
			cublasOperation_t transb = is_transb ? CUBLAS_OP_T : CUBLAS_OP_N;
			
			// Fill in random numbers
			std::mt19937 gen(0);
			std::uniform_real_distribution<float> dist(-2, 2);
			for (int i = 0; i < STRIDE_A * BATCH_COUNT; i++) {
				Aarray[i] = dist(gen);
			}
			for (int i = 0; i < STRIDE_B * BATCH_COUNT; i++) {
				Barray[i] = dist(gen);
			}
			for (int i = 0; i < STRIDE_C * BATCH_COUNT; i++) {
				ref_Carray[i] = dist(gen);
			}
			T alpha = dist(gen);
			T beta = dist(gen);

			// Copy A, B and C to GPU
			CUDA_CHECK(cudaMemcpy(Aarray_gpu, Aarray, STRIDE_A * BATCH_COUNT * sizeof(T), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(Barray_gpu, Barray, STRIDE_B * BATCH_COUNT * sizeof(T), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(ans_Carray_gpu, ref_Carray, STRIDE_C * BATCH_COUNT * sizeof(T), cudaMemcpyHostToDevice));

			sync_check_cuda_error();

			// Calculate ref answers
			naiveGemmStridedBatched(
				transa,
				transb,
				M,
				N,
				K,
				alpha,
				Aarray,
				STRIDE_A,
				Barray,
				STRIDE_B,
				beta,
				ref_Carray,
				STRIDE_C,
				BATCH_COUNT
			);

			// Calculate ans
			this->wrapper.gemmStridedBatched(
				transa,
				transb,
				M,
				N,
				K,
				alpha,
				Aarray_gpu,
				STRIDE_A,
				Barray_gpu,
				STRIDE_B,
				beta,
				ans_Carray_gpu,
				STRIDE_C,
				BATCH_COUNT
			);
			sync_check_cuda_error();

			// Compare
			bool is_pass = isArrayAlmostEqual(ans_Carray_gpu, ref_Carray, STRIDE_C * BATCH_COUNT, true, false);
			ASSERT_TRUE(is_pass);
		}
	}

	// Free
	CUDA_CHECK(cudaFree(Aarray_gpu));
	CUDA_CHECK(cudaFree(Barray_gpu));
	CUDA_CHECK(cudaFree(ans_Carray_gpu));
	delete[] Aarray;
	delete[] Barray;
	delete[] ref_Carray;
	sync_check_cuda_error();
}