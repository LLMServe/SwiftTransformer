#include "st_args.h"

#include <cstdio>

#include "common_gpt_hyper_params.h"

namespace st::example {

Precision precision_from_string(const std::string& precision_str){
    if (precision_map.find(precision_str) == precision_map.end()){
        std::cerr << "Invalid precision string: " + precision_str << std::endl;
        return Precision::INVALID;
    }
    return precision_map.at(precision_str);
}

std::string precision_to_string(Precision precision){
    for (auto it = precision_map.begin(); it != precision_map.end(); ++it){
        if (it->second == precision){
            return it->first;
        }
    }
    std::cerr << "Invalid precision: " + std::to_string(static_cast<int>(precision)) << std::endl;
    return "";
}

RunArgs::RunArgs(
        const std::string& model_weight_path, const std::string& vocab_json_path, const std::string& input_path,
        const std::string& str_hyper_param,  const std::string& str_precision,
        bool is_debug
    )
    : model_weight_path(model_weight_path), vocab_json_path(vocab_json_path), input_path(input_path),
      hyper_param(str2hyperparam(str_hyper_param)), precision(precision_from_string(str_precision)),
      is_debug(is_debug)
    {
    }

}