#pragma once

#include <unistd.h>
#include <unordered_map>

#include "model/gpt/gpt_hyper_param.h"
#include "kernel/activation_types.h"

// https://huggingface.co/facebook/opt-125m/blob/main/config.json
const st::model::GptHyperParam HYPERPARAM_OPT_125M = st::model::GptHyperParam::GetOptHyperParam(	// opt-125m
	50272,
	2048,
	768,
	12,
	12,
	64,
	3072
);

const st::model::GptHyperParam HYPERPARAM_OPT_1P3B = st::model::GptHyperParam::GetOptHyperParam(	// opt-1.3b
	50272,
	2048,
	2048,
	24,
	32,
	64,
	8192
);

const st::model::GptHyperParam HYPERPARAM_OPT_2P7B = st::model::GptHyperParam::GetOptHyperParam( // opt-2.7b
	50272,
	2048,
	2560,
	32,
	32,
	80,
	10240
);

const st::model::GptHyperParam HYPERPARAM_OPT_6P7B = st::model::GptHyperParam::GetOptHyperParam(	// opt-6.7b
	50272,
	2048,
	4096,
	32,
	32,
	128,
	16384
);

const st::model::GptHyperParam HYPERPARAM_OPT_13B = st::model::GptHyperParam::GetOptHyperParam(	// opt-13b
	50272,
	2048,
	5120,
	40,
	40,
	128,
	20480
);

const st::model::GptHyperParam HYPERPARAM_OPT_30B = st::model::GptHyperParam::GetOptHyperParam(	// opt-30b
	50272,
	2048,
	7168,
	48,
	56,
	128,
	28672
);

const st::model::GptHyperParam HYPERPARAM_LLAMA2_7B = st::model::GptHyperParam::GetLlama2HyperParam(	// llama2-7b
	32000,
	4096,
	4096,
	32,
	32,
	32,
	128,
	11008
);

const st::model::GptHyperParam HYPERPARAM_LLAMA2_13B = st::model::GptHyperParam::GetLlama2HyperParam(	// llama2-13b
	32000,
	4096,
	5120,
	40,
	40,
	40,
	128,
	13824
);

const st::model::GptHyperParam HYPERPARAM_LLAMA2_70B = st::model::GptHyperParam::GetLlama2HyperParam(	// llama2-70b
	32000,
	4096,
	8192,
	80,
	64,
	8,
	128,
	28672
);

// str2hyperparam - Return the correct hyperparam based on the string.
// If the string is invalid, print the valid hyperparam and return a hyperparam with vocab_size = -1.
inline st::model::GptHyperParam str2hyperparam(const std::string &str) {
	static const std::unordered_map<std::string, st::model::GptHyperParam> hyper_param_map = {
		{"opt_125m", HYPERPARAM_OPT_125M},
		{"opt_1.3b", HYPERPARAM_OPT_1P3B},
		{"opt_2.7b", HYPERPARAM_OPT_2P7B},
		{"opt_6.7b", HYPERPARAM_OPT_6P7B},
		{"opt_13b", HYPERPARAM_OPT_13B},
		{"opt_30b", HYPERPARAM_OPT_30B},
		{"llama2_7b", HYPERPARAM_LLAMA2_7B},
		{"llama2_13b", HYPERPARAM_LLAMA2_13B},
		{"llama2_70b", HYPERPARAM_LLAMA2_70B}
	};

	if (hyper_param_map.find(str) == hyper_param_map.end()) {
		printf("Invalid number of parameters: %s\n", str.c_str());
		printf("Valid number of parameters: ");
		for (auto it = hyper_param_map.begin(); it != hyper_param_map.end(); ++it) {
			printf("%s ", it->first.c_str());
		}
		exit(1);
		st::model::GptHyperParam res = HYPERPARAM_OPT_125M;
		res.vocab_size = -1;
		return res;
	}

	return hyper_param_map.at(str);
}
