#pragma once

#include <cstdio>
#include <vector>

#include "simple_vocab_decoder.h"

inline void print_prompt_and_output(
	const std::vector<int64_t> &prompt_tokens,
	const std::vector<int64_t> &output_tokens,
	const SimpleVocabDecoder &decoder) {
	printf("(");
	for (auto token : prompt_tokens)
		printf("%s ", decoder.decode(token).c_str());
	printf(") ");
	for (auto token : output_tokens)
		printf("%s ", decoder.decode(token).c_str());
	printf("  (prompt len = %ld, %ld tokens generated)", prompt_tokens.size(), output_tokens.size());
	printf("\n");
}