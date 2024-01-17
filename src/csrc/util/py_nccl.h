#pragma once

#include <torch/extension.h>
#include <nccl.h>
#include <vector>

namespace st::util {

// torch function to generate nccl_id
std::vector<int64_t> generate_nccl_id();

} // namespace st::util