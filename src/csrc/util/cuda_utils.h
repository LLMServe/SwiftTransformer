#pragma once

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        printf("[ERROR] CUDA error %s:%d '%s': (%d) %s\n", __FILE__, __LINE__, #cmd, (int)result, cudaGetErrorString(result)); \
        exit(-1); \
    } \
} while(0)

inline void syncAndCheck(const char* const file, int const line, bool force_check = false) {
#ifdef DEBUG
    force_check = true;
#endif
    if (force_check) {
        cudaDeviceSynchronize();
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(std::string("[ST] CUDA runtime error: ") + cudaGetErrorString(result) + " "
                                    + file + ":" + std::to_string(line) + " \n");
        }
    }
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__, false)
#define sync_check_cuda_error_force() syncAndCheck(__FILE__, __LINE__, true)

// Some stuff for indexing into an 1-D array
#define INDEX_2D(dim1, dim2, index1, index2) \
    (((int64_t)index1) * (dim2) + (index2))
#define INDEX_3D(dim1, dim2, dim3, index1, index2, index3) \
    (((int64_t)index1) * (dim2) * (dim3) + ((int64_t)index2) * (dim3) + (index3))
#define INDEX_4D(dim1, dim2, dim3, dim4, index1, index2, index3, index4) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) + ((int64_t)index2) * (dim3) * (dim4) + ((int64_t)index3) * (dim4) + (index4))
#define INDEX_5D(dim1, dim2, dim3, dim4, dim5, index1, index2, index3, index4, index5) \
    (((int64_t)index1) * (dim2) * (dim3) * (dim4) * (dim5) + ((int64_t)index2) * (dim3) * (dim4) * (dim5) + ((int64_t)index3) * (dim4) * (dim5) + (index4) * (dim5) + (index5))

// A tiny stuff that supports remalloc on GPU
template<typename T>
struct RemallocableArray {
    T* ptr;
    int64_t size;

    RemallocableArray() {
        ptr = nullptr;
        size = 0;
    }

    ~RemallocableArray() {
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }

    void remalloc(int64_t target_size) {
        if (target_size > size) {
            int64_t new_size = size ? size*2 : 64;
            while (new_size < target_size) {
                new_size *= 2;
            }
            if (ptr != nullptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
            CUDA_CHECK(cudaMalloc(&ptr, new_size * sizeof(T)));
            size = new_size;
        }
    }
};

template<typename T>
inline void printGpuArrayHelper(const T* array, int64_t size, const char* arr_name) {
    T* array_cpu = new T[size];
    CUDA_CHECK(cudaMemcpy(array_cpu, array, sizeof(T) * size, cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < size; i++) {
        printf("%f ", (float)array_cpu[i]);
    }
    printf("\n");
    delete[] array_cpu;
}

#define printGpuArray(array, size) printGpuArrayHelper(array, size, #array)

// A util to check cuda memory usage
inline int64_t cuda_memory_size() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    return total_byte - free_byte;
}

// CUDAFreeAtReturn - A tiny macro to call cudaFree when the point goes out of scope
template<typename PTR_T>
class CUDAFreeAtReturnHelper {
private:
	PTR_T ptr;
	std::string pointer_name;
public:
	CUDAFreeAtReturnHelper(PTR_T ptr, std::string pointer_name):
		pointer_name(pointer_name) { this->ptr = ptr; }
	~CUDAFreeAtReturnHelper() {
		if (ptr != nullptr) {
			cudaFree(ptr);
			cudaDeviceSynchronize();
			cudaError_t result = cudaGetLastError();
			if (result) {
				fprintf(stderr, "Error occured when freeing pointer %s\n", pointer_name.c_str());
				fprintf(stderr, "%s\n", (std::string("[ST] CUDA runtime error: ") + cudaGetErrorString(result) + " "
										+ __FILE__ + ":" + std::to_string(__LINE__) + " \n").c_str());
				exit(1);
			}
		}
	}
};
#define CUDA_FREE_AT_RETURN(ptr) CUDAFreeAtReturnHelper<decltype(ptr)> ptr##_cuda_free_at_return(ptr, #ptr)

template<typename T>
cudaDataType_t getCudaDataType() {
    if (std::is_same<T, half>::value) {
        return CUDA_R_16F;
    }
#ifdef ENABLE_BF16
    else if (std::is_same<T, __nv_bfloat16>::value) {
        return CUDA_R_16BF;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return CUDA_R_32F;
    }
    else {
        throw std::runtime_error("Cuda data type: Unsupported type");
    }
}
