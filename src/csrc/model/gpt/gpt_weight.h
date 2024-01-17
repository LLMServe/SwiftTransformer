#pragma once

#include <string>
#include <vector>

#include <torch/script.h>

#include "gpt_hyper_param.h"
#include "gpt_parallelism_param.h"

namespace st::model {

// GptLayerWeight - weights of a single GPT layer
// This struct contains the weights of a single GPT layer. The weights are
// loaded from a pt file.
// All pointers are owned (allocated and freed) by GptWeight<T>.
template<typename T>
struct GptLayerWeight {
	T* attn_qkv_kernel = nullptr;		// [hidden_size, local_q_head_num+2*local_kv_head_num, head_dim]
	T* attn_qkv_bias = nullptr;			// [local_q_head_num+2*local_kv_head_num, head_dim]
	T* attn_out_kernel = nullptr;		// [local_q_head_num, head_dim, hidden_size]
	T* attn_out_bias = nullptr;			// [hidden_size]

	T* attn_layernorm_weight = nullptr;	// [hidden_size]
	T* attn_layernorm_bias = nullptr;	// [hidden_size]

	T* ffn_fc1_weight = nullptr;		// [inter_dim / tensor_para_size, hidden_size]
	T* ffn_fc1_bias = nullptr;			// [inter_dim / tensor_para_size], will be used only when is_gated_ffn = false
	T* ffn_fc2_weight = nullptr;		// [hidden_size, inter_dim / tensor_para_size]
	T* ffn_fc2_bias = nullptr;			// [hidden_size], will be used only when is_gated_ffn = false
	T* ffn_fc3_weight = nullptr;		// [inter_dim / tensor_para_size, hidden_size], will be used only when is_gated_ffn = true

	T* final_layernorm_weight = nullptr;// [hidden_size]
	T* final_layernorm_bias = nullptr;	// [hidden_size]
};


// GptWeight - weights of the GPT model
// All pointers are owned (allocated and freed) by itself
template<typename T>
class GptWeight {
private:
	void allocateWeightArray();
	void freeWeightArray();
	void loadTensor_qkv_weight_kernel_or_bias(const uint32_t dim, T* to_ptr, const std::string model_dir, const std::string key, const int64_t expect_size);
	void loadTensor_tp(const uint32_t dim, T* to_ptr, const std::string model_dir, const std::string key,  const int64_t expect_size);
	void loadTensor_all(T* to_ptr, const std::string model_dir, const std::string key, const int64_t expect_size);
	bool contain_embedding_layer = false;

public:
	GptHyperParam hyper_param;
	GptParallelismParam parallelism_param;

	T* embed_tokens_weight;		// [vocab_size, hidden_size]
	T* embed_positions_weight;	// [max_position_embeddings, hidden_size], will be used only when is_rotary_embedding = false

	std::vector<GptLayerWeight<T>> layer_weights;

	T* final_layernorm_weight;	// [hidden_size]
	T* final_layernorm_bias;	// [hidden_size]
	T* output_proj_weight;		// [vocab_size, hidden_size]

	T layernorm_epsilon = (T)1e-5;

	bool initialized = false;

	GptWeight();
	~GptWeight();

	void init(const GptHyperParam& hyper_param, GptParallelismParam& parallelism_param = GptParallelismParam());

	void loadWeight(const std::string& weight_path);
	void initDummyWeight();
};


}	// namespace st::model
