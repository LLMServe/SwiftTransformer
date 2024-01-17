#pragma once

#include <string>
#include <vector>

#include <nccl.h>

#include "gpt_hyper_param.h"
#include "gpt_pagedattn_param.h"
#include "gpt_parallelism_param.h"

namespace st::model {

/*
GptBase: Abstract base class for GPT. The actual GPT class is based on this class.

We have this class because PyTorch binding requires a non-template class, so
our XXXop (e.g. OptOp) classes cannot have template parameters. So the following
implementation won't work:

```
template <typename T>
class OptOp {
		Gpt<T> gpt;
};
```

Instead, we have to use a non-template base class, and leverage virtual function
and polymorphism in C++ to implement the actual GPT and XXXop class. For example:

```
class GptBase {
	// Virtual functions like loadWeight and forward
};

template<typename T>
class Gpt : GPTBase {
	// Implementations of virtual functions
};

class GptOpBase {
	GptBase* gpt;	// A pointer to GPTBase, which can be Gpt<T> for any T.
	// Implementations of loadWeight() and forward(), which takes torch::Tensor
	// as input and calls Gpt::loadWeight() and Gpt::forward().
};

class OptOp : GptOpBase {
	// Implementation of constructor
};
```
*/

class GptBase {
public: 
	GptHyperParam hyper_param;
	GptPagedAttnParam pagedattn_param;
	GptParallelismParam parallelism_param;

	virtual ~GptBase() {}
	virtual void loadWeight(const std::string&) = 0;
	virtual void initDummyWeight() = 0;

	// Forward function for GPT.
	// Args: 
	//	input_tokens_batched: a batch of requests, where each element is vector of tokens for corresponding request.
	//						  input_tokens_batched may contain requests in context or decoding phase.
	//	first_token_indexes: the index of the first token in each request's input_tokens. For example, if request i is
	// 						  in decoding phase, and it has generated 5 tokens, then first_token_indexes[i] = 5.
	//						  if first_token_indexes[j] == 0, then request j is in context phase.
	//	d_k_cache: the overall key cache. [num_blocks, num_layers, num_local_heads, block_size, head_dim]
	//	d_v_cache: the overall value cache. [num_blocks, num_layers, num_local_heads, block_size, head_dim] 
	//	block_table: block_table[i][j] = k means the j-th logical block for request i is the k-th physical block in the overall key/value cache.
	// Note: here request i is the i-th request in input_tokens_batched, not the i-th request in the whole dataset.
    virtual std::vector<int64_t> forward(
		// input data && metadata
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const std::vector<int64_t> &first_token_indexes,

		// key-value management
		void* d_k_cache,	   
		void* d_v_cache,
		int64_t* d_block_table
    ) = 0;

	virtual void init_communicator(const ncclUniqueId& tp_id, const ncclUniqueId& pp_id) = 0;

};

}  // namespace st::model