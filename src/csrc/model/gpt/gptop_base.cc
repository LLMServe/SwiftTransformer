#include "gptop_base.h"

#include "util/torch_utils.h"
#include "util/cuda_utils.h"
#include "util/nccl_utils.h"

namespace st::model {

GptOpBase::GptOpBase(
	std::string inference_dtype,
	GptHyperParam hyper_param,
	GptPagedAttnParam pagedattn_param,
	GptParallelismParam parallelism_param
) {
	if (inference_dtype == "fp32") {
		gpt = new Gpt<float>(hyper_param, pagedattn_param, parallelism_param);
	} else if (inference_dtype == "fp16") {
		gpt = new Gpt<__half>(hyper_param, pagedattn_param, parallelism_param);
	} else {
		throw std::runtime_error("Unsupported inference_dtype: " + inference_dtype);
	}

	weight_loaded = false;
}

GptOpBase::~GptOpBase() {
	delete gpt;
}

void GptOpBase::loadWeight(const std::string& weight_path) {
    this->gpt->loadWeight(weight_path);
    this->weight_loaded = true;
};

void GptOpBase::initDummyWeight() {
    this->gpt->initDummyWeight();
    this->weight_loaded = true;
};

std::vector<int64_t> GptOpBase::forward(
    const std::vector<std::vector<int64_t>> &input_tokens_batched,
	const std::vector<int64_t> &first_token_indexes, // [batchsize]
    torch::Tensor &k_cache, // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor &v_cache, // [num_blocks, num_heads, block_size, head_dim]
    const std::vector<std::vector<int64_t>> &block_table)
{
    if (!this->weight_loaded) {
        throw std::runtime_error("Please load the weight before inference.");
    }

    int64_t batch_size = input_tokens_batched.size();
    if (batch_size == 0) {
		return std::vector<int64_t>();
    }

    // Prepare block_table
    int64_t* h_block_table = new int64_t[batch_size * this->gpt->pagedattn_param.max_num_block_per_req];
    for (int64_t i = 0; i < batch_size; i++) {
        memcpy(h_block_table + i * this->gpt->pagedattn_param.max_num_block_per_req, block_table[i].data(), block_table[i].size() * sizeof(int64_t));
    }
    d_block_table.remalloc(batch_size * this->gpt->pagedattn_param.max_num_block_per_req);
    cudaMemcpy(d_block_table.ptr, h_block_table, sizeof(int64_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice);
    delete[] h_block_table;
    sync_check_cuda_error();

    auto result = this->gpt->forward(input_tokens_batched,
                              first_token_indexes,
                              st::util::convertTensorToRawPtr(k_cache),
                              st::util::convertTensorToRawPtr(v_cache),
                              d_block_table.ptr);
    return result;
}

void GptOpBase::init_communicator(const std::vector<int64_t> tp_id, const std::vector<int64_t> pp_id){
    ncclUniqueId tp_uid, pp_uid;
	memcpy(tp_uid.internal, &tp_id[0], NCCL_UNIQUE_ID_BYTES);
	memcpy(pp_uid.internal, &pp_id[0], NCCL_UNIQUE_ID_BYTES);
    this->gpt->init_communicator(tp_uid, pp_uid);
}

}
