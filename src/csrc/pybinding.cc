#include <torch/script.h>

#include "util/py_nccl.h"
#include "util/py_swapping.h"
#include "util/py_block_migration.h"

#include "model/gpt/opt/optop.h"
#include "model/gpt/llama2/llama2op.h"
#include "model/gpt/gpt2/gpt2op.h"

/*
The two function wrappers below are needed to avoid the following error:

RuntimeError: Tried to convert an IValue of type __torch__.torch.classes.gpt_ops.OptOp (of Python compilation unit at: 0) to custom class type __torch__.torch.classes.gpt_ops.GptOpBase (of Python compilation unit at: 0)

We encounter the error above because of that, when we create a OptOp class in python and
call its load_weight() method, we are actually calling the load_weight() method of the
GptOpBase class, which is the base class of OptOp. However, PyTorch thinks it needs to
convert the OptOp object to a GptOpBase object (since the first argument of loadWeight
is GptOpBase*), which is not possible because we didn't defined that.

The solution is to define a wrapper function that takes a OptOp object as the first
argument and calls the loadWeight() method of the OptOp object, which avoid type
conversion.
*/
template<typename T>
void loadWeightWrapper(const c10::intrusive_ptr<T>& self, const std::string& path) {
  self->loadWeight(path);
}

template<typename T>
void initDummyWeightWrapper(const c10::intrusive_ptr<T>& self) {
  self->initDummyWeight();
}

template<typename T>
std::vector<int64_t> forwardWrapper(const c10::intrusive_ptr<T>& self,
                                    const std::vector<std::vector<int64_t>> &input_tokens_batched,
                                    const std::vector<int64_t> &first_token_indexes,
                                    torch::Tensor &k_cache,
                                    torch::Tensor &v_cache,
                                    const std::vector<std::vector<int64_t>> &block_table) {
  return self->forward(input_tokens_batched, first_token_indexes, k_cache, v_cache, block_table);
}

template<typename T>
void initCommunicatorWrapper(const c10::intrusive_ptr<T>& self,
                             const std::vector<int64_t> tp_id,
                             const std::vector<int64_t> pp_id) {
  self->init_communicator(tp_id, pp_id);
}

TORCH_LIBRARY(gpt_ops, m) {
  m.class_<st::model::GptOpBase>("GptOpBase");  // Must add this class or will get error: "c10::intrusive_ptr<...> could not be converted to any of the known types."
  m.class_<st::model::OptOp>("OptOp")
    .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t, std::string, int64_t, int64_t, std::vector<int64_t> >())
    .def("load_weight", &loadWeightWrapper<st::model::OptOp>)
    .def("init_dummy_weights", &initDummyWeightWrapper<st::model::OptOp>)
    .def("forward", &forwardWrapper<st::model::OptOp>)
    .def("init_communicator", &initCommunicatorWrapper<st::model::OptOp>)

  ;
  m.class_<st::model::Llama2Op>("Llama2Op")
    .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t, std::string, int64_t, int64_t, std::vector<int64_t> >())
    .def("load_weight", &loadWeightWrapper<st::model::Llama2Op>)
    .def("init_dummy_weights", &initDummyWeightWrapper<st::model::Llama2Op>)
    .def("forward", &forwardWrapper<st::model::Llama2Op>)
    .def("init_communicator", &initCommunicatorWrapper<st::model::Llama2Op>)
  ;
  m.class_<st::model::Gpt2Op>("Gpt2Op")
    .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t, std::string, int64_t, int64_t, std::vector<int64_t> >())
    .def("load_weight", &loadWeightWrapper<st::model::Gpt2Op>)
    .def("init_dummy_weights", &initDummyWeightWrapper<st::model::Gpt2Op>)
    .def("forward", &forwardWrapper<st::model::Gpt2Op>)
    .def("init_communicator", &initCommunicatorWrapper<st::model::Gpt2Op>)

  ;
}

TORCH_LIBRARY(nccl_ops, m)
{
    m.def("generate_nccl_id", &st::util::generate_nccl_id);
}

TORCH_LIBRARY(swapping_ops, m) {
  m.def("swap", &st::util::swap);
}

TORCH_LIBRARY(block_migration_ops, m) {
  m.def("get_ipc_mem_handle", &st::util::get_ipc_mem_handle);
  m.def("register_ipc_mem_handle", &st::util::register_ipc_mem_handle);
  m.def("migrate_blocks", &st::util::migrate_blocks);
}