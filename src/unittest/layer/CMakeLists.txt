add_library(unittest_layer_ref
	attention_ref.cc
)
target_link_libraries(unittest_layer_ref PUBLIC unittest_kernel_ref)

add_executable(unittest_layer_para
	parallel_ffn.cc
	parallel_attention.cc
)
target_link_libraries(unittest_layer_para PUBLIC unittest_layer_ref layer util kernel gtest_main nccl_utils)