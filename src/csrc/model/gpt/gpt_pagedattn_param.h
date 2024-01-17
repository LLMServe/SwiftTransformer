#pragma once

#include <iostream>

namespace st::model {

struct GptPagedAttnParam {
	// Hyperparameters related to PagedAttention
	int64_t block_size;
	int64_t max_num_block_per_req;

	friend std::ostream& operator<<(std::ostream& os, const GptPagedAttnParam& params) {
		os << "GptPagedAttnParam {\n"
			<< "\tblock_size = " << params.block_size << "\n"
			<< "\tmax_num_block_per_req = " << params.max_num_block_per_req << "\n"
			<< "}";
		return os;
	}
};

}	// namespace st::model
