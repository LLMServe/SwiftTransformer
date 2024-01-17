#pragma once

#include <iostream>

#include "kernel/activation_types.h"
namespace st::model {

struct GptHyperParam {
	// Hyper-parameters
	int64_t vocab_size;		// The size of the vocabulary
	int64_t max_position_embeddings;	// The maximum length of the input sequence
	int64_t hidden_size;	// The length of the embedded vector
	int64_t num_layers;		// The number of layers (transformer blocks)
	int64_t num_q_heads;	// The number of query heads in the multi-head attention
	int64_t num_kv_heads;	// The number of key/value heads in the multi-head attention.
							// If the model does not use GQA (Grouped Query Attention), just
							// set num_kv_heads = num_q_heads
	int64_t head_dim;		// The dimension of each head (length of the key, query, and value vectors)
	int64_t ffn_inter_dim;	// The intermediate dimension of the feed-forward network

	// Model configurations
	bool is_pre_layernorm;	// Perform layernorm before/after the self-attention and feed-forward network
	bool is_rotary_posi_embedding;	// Use rotary position embedding instead of absolute position embedding
	bool is_gated_ffn;		// Use gated feed-forward network
	ActivationType ffn_activation_type;	// The activation function of the feed-forward network
	bool is_rmsnorm;		// Use RMSNorm instead of LayerNorm
	bool is_attn_qkv_biased;
	bool is_attn_out_biased;
	
	friend std::ostream& operator<<(std::ostream& os, const GptHyperParam& params) {
		os << "GptHyperParam {\n"
			<< "\tvocab_size = " << params.vocab_size << "\n"
			<< "\tmax_position_embeddings = " << params.max_position_embeddings << "\n"
			<< "\thidden_size = " << params.hidden_size << "\n"
			<< "\tnum_layers = " << params.num_layers << "\n"
			<< "\tnum_q_heads = " << params.num_q_heads << "\n"
			<< "\tnum_kv_heads = " << params.num_kv_heads << "\n"
			<< "\thead_dim = " << params.head_dim << "\n"
			<< "\tffn_inter_dim = " << params.ffn_inter_dim << "\n"
			<< "\tis_pre_layernorm = " << params.is_pre_layernorm << "\n"
			<< "\tis_rotary_posi_embedding = " << params.is_rotary_posi_embedding << "\n"
			<< "\tis_gated_ffn = " << params.is_gated_ffn << "\n"
			<< "\tffn_activation_type = " << static_cast<int>(params.ffn_activation_type) << "\n"
			<< "\tis_rmsnorm = " << params.is_rmsnorm << "\n"
			<< "\tis_attn_qkv_biased = " << params.is_attn_qkv_biased << "\n"
			<< "\tis_attn_out_bias = " << params.is_attn_out_biased << "\n"
			<< "}";
		return os;
	}

	static GptHyperParam GetOptHyperParam (
		int64_t vocab_size,
		int64_t max_position_embeddings,
		int64_t hidden_size,
		int64_t num_layers,
		int64_t num_heads,
		int64_t head_dim,
		int64_t ffn_inter_dim
	) {
		return GptHyperParam{
			.vocab_size = vocab_size,
			.max_position_embeddings = max_position_embeddings,
			.hidden_size = hidden_size,
			.num_layers = num_layers,
			.num_q_heads = num_heads,
			.num_kv_heads = num_heads,
			.head_dim = head_dim,
			.ffn_inter_dim = ffn_inter_dim,
			.is_pre_layernorm = true,
			.is_rotary_posi_embedding = false,
			.is_gated_ffn = false,
			.ffn_activation_type = ActivationType::RELU,
			.is_rmsnorm = false,
			.is_attn_qkv_biased = true,
			.is_attn_out_biased = true
		};
	}

	static GptHyperParam GetLlama2HyperParam (
		int64_t vocab_size,
		int64_t max_position_embeddings,
		int64_t hidden_size,
		int64_t num_layers,
		int64_t num_q_heads,
		int64_t num_kv_heads,
		int64_t head_dim,
		int64_t ffn_inter_dim
	) {
		return GptHyperParam{
			.vocab_size = vocab_size,
			.max_position_embeddings = max_position_embeddings,
			.hidden_size = hidden_size,
			.num_layers = num_layers,
			.num_q_heads = num_q_heads,
			.num_kv_heads = num_kv_heads,
			.head_dim = head_dim,
			.ffn_inter_dim = ffn_inter_dim,
			.is_pre_layernorm = true,
			.is_rotary_posi_embedding = true,
			.is_gated_ffn = true,
			.ffn_activation_type = ActivationType::SILU,
			.is_rmsnorm = true,
			.is_attn_qkv_biased = false,
			.is_attn_out_biased = false
		};
	}

	static GptHyperParam GetGpt2HyperParam (
		int64_t vocab_size,
		int64_t max_position_embeddings,
		int64_t hidden_size,
		int64_t num_layers,
		int64_t num_heads,
		int64_t head_dim,
		int64_t ffn_inter_dim
	) {
		return GptHyperParam{
			.vocab_size = vocab_size,
			.max_position_embeddings = max_position_embeddings,
			.hidden_size = hidden_size,
			.num_layers = num_layers,
			.num_q_heads = num_heads,
			.num_kv_heads = num_heads,
			.head_dim = head_dim,
			.ffn_inter_dim = ffn_inter_dim,
			.is_pre_layernorm = true,
			.is_rotary_posi_embedding = false,
			.is_gated_ffn = false,
			.ffn_activation_type = ActivationType::GELU,
			.is_rmsnorm = false,
			.is_attn_qkv_biased = true,
			.is_attn_out_biased = true
		};
	}
};

}	// namespace st::model
