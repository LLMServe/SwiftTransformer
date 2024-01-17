#pragma once

#include <vector>
#include <optional>

#include "model/gpt/opt/optop.h"
#include "simple_vocab_decoder.h"

struct RuntimeUsage {
	double context_stage_time;
	double decoding_stage_time;
	double total_time;
};

template<typename T>
RuntimeUsage run_batched_inference(
	std::vector<std::vector<int64_t>> &output_tokens_batched,
	const std::vector<std::vector<int64_t>> &input_tokens_batched,
	st::model::Gpt<T> &gpt,
	const st::model::GptHyperParam &hyper_param,
	const st::model::GptPagedAttnParam &pagedattn_param,
	const st::model::GptParallelismParam &parallel_param,
	const int64_t max_decoding_step,
	const int64_t end_token,
	const int64_t num_total_blocks,
	bool print_debug_info,
	bool exit_after_context_stage = false,
	std::optional<SimpleVocabDecoder> vocab_decoder = std::nullopt
);
