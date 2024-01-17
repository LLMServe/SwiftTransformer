add_library(unittest_kernel_ref
	attention_ref.cc
	kvcache_mgmt_ref.cc
	rotary_posi_embedding_ref.cc
)
target_link_libraries(unittest_kernel_ref PUBLIC kernel)

add_executable(unittest_kernel
	addbias.cc
	findmax.cc
	fused_activ_multiply.cc
	fused_addbias_activ.cc
	layernorm.cc
	rmsnorm.cc
	rotary_posi_embedding.cc
	softmax.cc
)
target_link_libraries(unittest_kernel PUBLIC kernel unittest_kernel_ref gtest_main)
