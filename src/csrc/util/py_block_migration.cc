#include "py_block_migration.h"

#include <cstdio>
#include <cstdlib>
#include <map>

#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>	// for at::cuda::getCurrentCUDAStream()

#include "cuda_utils.h"
#include "debug_utils.h"

namespace st::util {


/*
The following two functions convert cudaIpcMemHandle_t to/from bytes
We need this because we need to pass cudaIpcMemHandle_t to Python
*/

static std::vector<int64_t> cudaIpcMemHandle2Bytes(const cudaIpcMemHandle_t &handle) {
	std::vector<int64_t> result;
	for (size_t i = 0; i < sizeof(handle); ++i) {
		result.push_back(((uint8_t*) &handle)[i]);
	}
	return result;
}

static cudaIpcMemHandle_t bytes2CudaIpcMemHandle(const std::vector<int64_t> &bytes) {
	assert_whenever(bytes.size() == sizeof(cudaIpcMemHandle_t));
	cudaIpcMemHandle_t result;
	for (size_t i = 0; i < sizeof(result); ++i) {
		((uint8_t*) &result)[i] = bytes[i];
	}
	return result;
}


/*
get_ipc_mem_handle: Get the IPC memory handle of a tensor
The returned handle can be used to open the tensor in another process.
*/
std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor) {
	cudaIpcMemHandle_t handle;
	CUDA_CHECK(cudaIpcGetMemHandle(&handle, tensor.data_ptr()));
	return cudaIpcMemHandle2Bytes(handle);
}


/*
register_ipc_mem_handle: Register an IPC memory handle

This function receives a IPC memory handle and the context worker's info
(context_pp_rank and context_tp_rank) that the handle belongs to, then
it checks whether it needs to register the handle (i.e. whether the k/v range
it needs overlaps with the k/v range that the context worker calculates). If
the answer is YES, register it and note down its local address.

Return true if the handle is registered, false otherwise.
*/
static constexpr int64_t MAX_PARALLEL_HASH = 4096;	// Assume there are at most 64 pp stages and 64 tp stages
static void* context_worker_k_cache_addr[MAX_PARALLEL_HASH];
static void* context_worker_v_cache_addr[MAX_PARALLEL_HASH];
bool register_ipc_mem_handle(
	std::vector<int64_t> k_cache_handle_vec,
	std::vector<int64_t> v_cache_handle_vec,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,	// Generated via ParallelConfig.to_list()
	const std::vector<int64_t> &decoding_parallel_config
) {
	// Convert the handles to cudaIpcMemHandle_t
	const cudaIpcMemHandle_t k_cache_handle = bytes2CudaIpcMemHandle(k_cache_handle_vec);
	const cudaIpcMemHandle_t v_cache_handle = bytes2CudaIpcMemHandle(v_cache_handle_vec);

	// First we check whether the two k/v cache area overlaps
	const int64_t context_tp_size = context_parallel_config[0];
	const int64_t context_tp_rank = context_parallel_config[1];
	const int64_t context_pp_size = context_parallel_config[2];
	const int64_t context_pp_rank = context_parallel_config[3];
	const int64_t decoding_tp_size = decoding_parallel_config[0];
	const int64_t decoding_tp_rank = decoding_parallel_config[1];
	const int64_t decoding_pp_size = decoding_parallel_config[2];
	const int64_t decoding_pp_rank = decoding_parallel_config[3];

	const int64_t layers_per_context_worker = num_layers / context_pp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;
	const int64_t layers_per_decoding_worker = num_layers / decoding_pp_size;
	const int64_t heads_per_decoding_worker = num_heads / decoding_tp_size;

	const int64_t context_start_layer = context_pp_rank * layers_per_context_worker;
	const int64_t context_end_layer = context_start_layer + layers_per_context_worker;
	const int64_t context_start_head = context_tp_rank * heads_per_context_worker;
	const int64_t context_end_head = context_start_head + heads_per_context_worker;

	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_end_layer = decoding_start_layer + layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;
	const int64_t decoding_end_head = decoding_start_head + heads_per_decoding_worker;

	if (context_end_layer <= decoding_start_layer || context_start_layer >= decoding_end_layer ||
		context_end_head <= decoding_start_head || context_start_head >= decoding_end_head) {
		// No overlap
		return false;
	} else {
		// Overlap
		// Register the handle
		// On some platforms (e.g. non-nvlink platform) it's impossible to enable GPU p2p access, which 
		// leads to error when calling cudaIpcOpenMemHandle.
		const int64_t context_worker_hash = (context_pp_rank<<6) + context_tp_rank;
		cudaError_t err = cudaIpcOpenMemHandle(&context_worker_k_cache_addr[context_worker_hash], k_cache_handle, cudaIpcMemLazyEnablePeerAccess);
		if (err == cudaErrorPeerAccessUnsupported) {
			printf("Error: Peer-to-peer access is unsupported on this platform.\n");
			printf("In the current version of distserve, it is necessary to use a platform that supports GPU P2P access.\n");
			printf("Exiting...");
			exit(1);
		}
		CUDA_CHECK(cudaIpcOpenMemHandle(&context_worker_v_cache_addr[context_worker_hash], v_cache_handle, cudaIpcMemLazyEnablePeerAccess));
		return true;
	}
}


/*
migrate_blocks: Migrate blocks from the context stage engine to the decoding stage engine

This function is called by every decoding stage worker when the decoding
stage engine decides to migrate some blocks from the context stage engine
to the decoding stage engine.

In the following code, "pp" stands for "pipeline parallel", and "tp" stands
for "tensor parallel".

Here we do not pass a cudaStream to the function. Instead we use the current
stream indicated by at::cuda::getCurrentCUDAStream(). So it is python's
responsibility to set the current stream before calling this function.
*/

void migrate_blocks(
	// Parallelism parameters for the context stage engine
	const int64_t context_pp_size,
	const int64_t context_tp_size,

	// Block indexes of the context stage engine
	const std::vector<int64_t> &context_block_indexes,

	// Parallelism parameters for the decoding stage engine
	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	// Rank of the decoding stage worker that calls this function
	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	// Block indexes of the decoding stage engine
	const std::vector<int64_t> &decoding_block_indexes,

	// The decoding stage worker's KV cache
	torch::Tensor decoding_worker_k_cache,	// [num_blocks, layers_per_decoding_worker, heads_per_decoding_worker, block_size, head_dim]
	torch::Tensor decoding_worker_v_cache
) {
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	assert_whenever(decoding_worker_k_cache.is_contiguous());
	assert_whenever(decoding_worker_v_cache.is_contiguous());

	// Calculate some misc stuff
	const int64_t layers_per_decoding_worker = decoding_worker_k_cache.size(1);
	const int64_t heads_per_decoding_worker = decoding_worker_k_cache.size(2);
	const int64_t block_size = decoding_worker_k_cache.size(3);
	const int64_t head_dim = decoding_worker_k_cache.size(4);
	const int64_t num_layers = layers_per_decoding_worker * decoding_pp_size;
	const int64_t num_heads = heads_per_decoding_worker * decoding_tp_size;
	const int64_t layers_per_context_worker = num_layers / context_pp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;
	const int64_t num_blocks_to_copy = decoding_block_indexes.size();
	const int64_t dtype_size = decoding_worker_k_cache.dtype().itemsize();

	// The current decoding worker's region of the k/v cache
	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_end_layer = decoding_start_layer + layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;
	const int64_t decoding_end_head = decoding_start_head + heads_per_decoding_worker;

	for (int64_t context_pp_rank = 0; context_pp_rank < context_pp_size; ++context_pp_rank) {
		// First we iterate over every context pp stage
		const int64_t context_start_layer = context_pp_rank * layers_per_context_worker;
		const int64_t context_end_layer = context_start_layer + layers_per_context_worker;
		if (context_end_layer <= decoding_start_layer || context_start_layer >= decoding_end_layer) {
			continue;
		}
		for (int64_t context_tp_rank = 0; context_tp_rank < context_tp_size; ++context_tp_rank) {
			// Then we iterate over every context tp worker in the current pp stage
			const int64_t context_start_head = context_tp_rank * heads_per_context_worker;
			const int64_t context_end_head = context_start_head + heads_per_context_worker;
			if (context_end_head <= decoding_start_head || context_start_head >= decoding_end_head) {
				continue;
			}

			// The current context worker's region intersects with the current decoding worker's region
			// So we need to copy something from the context worker to the decoding worker
			// The context worker holds k/v cache of range [context_start_layer, context_end_layer) x [context_start_head, context_end_head)
			// The decoding worker holds k/v cache of range [decoding_start_layer, decoding_end_layer) x [decoding_start_head, decoding_end_head)
			// We then calculate the intersection of these two ranges
			const int64_t overlap_start_layer = std::max(context_start_layer, decoding_start_layer);
			const int64_t overlap_end_layer = std::min(context_end_layer, decoding_end_layer);
			const int64_t overlap_start_head = std::max(context_start_head, decoding_start_head);
			const int64_t overlap_end_head = std::min(context_end_head, decoding_end_head);
			assert_whenever(overlap_start_layer < overlap_end_layer);
			assert_whenever(overlap_start_head < overlap_end_head);

			// Note that this function is synchronous with respect to the host only if the source or destination of the transfer is host memory.
			// Note also that this copy is serialized with respect to all pending and future asynchronous work in to the current device, the copy's source device, and the copy's destination device (use cudaMemcpy3DPeerAsync to avoid this synchronization).

			// kv cache shape: [num_blocks, layers_per_worker, heads_per_worker, block_size, head_dim]
			for (int64_t block_id = 0; block_id < num_blocks_to_copy; ++block_id) {
				const int64_t context_block_index = context_block_indexes[block_id];
				const int64_t decoding_block_index = decoding_block_indexes[block_id];
				for (int is_value = 0; is_value < 2; ++is_value) {
					const int64_t context_worker_hash = (context_pp_rank<<6) + context_tp_rank;
					char* context_worker_base_ptr = (char*) (is_value ? context_worker_v_cache_addr[context_worker_hash] : context_worker_k_cache_addr[context_worker_hash]);
					if (!context_worker_base_ptr) {
						// This context worker has not registered. Panic
						fprintf(stderr, "Error: context worker %ld-%ld has not registered\n", context_pp_rank, context_tp_rank);
						exit(1);
					}
					CUDA_CHECK(cudaMemcpy2DAsync(
						(char*) (is_value ? decoding_worker_v_cache.data_ptr() : decoding_worker_k_cache.data_ptr())
							+ INDEX_5D(0, layers_per_decoding_worker, heads_per_decoding_worker, block_size, head_dim,
								decoding_block_index,
								overlap_start_layer - decoding_start_layer,
								overlap_start_head - decoding_start_head,
								0, 0) * dtype_size,
						(uint64_t) ((block_size * head_dim * dtype_size) * heads_per_decoding_worker),
						context_worker_base_ptr
							+ INDEX_5D(0, layers_per_context_worker, heads_per_context_worker, block_size, head_dim,
								context_block_index,
								overlap_start_layer - context_start_layer,
								overlap_start_head - context_start_head,
								0, 0) * dtype_size,
						(uint64_t) ((block_size * head_dim * dtype_size) * heads_per_context_worker),
						(size_t) ((overlap_end_head - overlap_start_head) * block_size * head_dim * dtype_size),
						(size_t) (overlap_end_layer - overlap_start_layer),
						cudaMemcpyDeviceToDevice,
						stream
					));
				}
			}
		}
	}
}

}