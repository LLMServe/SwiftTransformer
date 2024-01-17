#pragma once

namespace st::kernel {

template<typename T>
void findmaxBatched(
	int64_t* max_indices,
	const T* input,
	const int64_t batch_size,
	const int64_t length
);

}	// namespace st::kernel