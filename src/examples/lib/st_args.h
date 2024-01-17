#pragma once

#include <unordered_map>
#include <string>

#include "model/gpt/gpt_hyper_param.h"

namespace st::example {

enum class Precision {
    FP32,
    FP16,
    INVALID
};

const std::unordered_map<std::string, Precision> precision_map = {
    {"fp32", Precision::FP32},
    {"fp16", Precision::FP16},
    {"FP32", Precision::FP32},
    {"FP16", Precision::FP16},
    {"", Precision::INVALID}
};

Precision precision_from_string(const std::string& precision_str);
std::string precision_to_string(Precision precision);

struct RunArgs {
    std::string model_weight_path, vocab_json_path, input_path;
    st::model::GptHyperParam hyper_param;
    Precision precision;
    bool is_debug = false;

    RunArgs() = default;
    RunArgs(
        const std::string& model_weight_path, const std::string& vocab_json_path, const std::string& input_path,
        const std::string& str_hyper_param,  const std::string& str_precision,
        bool is_debug = false
    );
};

} // namespace st::example