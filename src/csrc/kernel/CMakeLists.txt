add_library(kernel STATIC
	addbias.cu
	count_nan.cu
	embedding.cu
	findmax.cu
	fused_activ_multiply.cu
	fused_addbias_activ.cu
	fused_context_stage_attention.cu
	fused_decoding_stage_attention.cu
	fused_decoding_stage_attention_mha.cu
	gather_last_tokens.cu
	kvcache_mgmt.cu
	layernorm.cu
	rmsnorm.cu
	rotary_posi_embedding.cu
	softmax.cu
	unfused_attention.cu
)
target_link_libraries(kernel PUBLIC util)
set_property(TARGET kernel PROPERTY POSITION_INDEPENDENT_CODE ON)

file(GLOB xformers_autogen_impl_files ${CMAKE_CURRENT_SOURCE_DIR}/xformers/xformers/csrc/attention/cuda/fmha/autogen/impl/*.cu)
add_library(xformers_autogen_impl STATIC ${xformers_autogen_impl_files})
target_include_directories(xformers_autogen_impl PUBLIC xformers/third_party/cutlass/include)
set_property(TARGET xformers_autogen_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(xformers_kernel STATIC
	xformers_attention.cu
)
target_include_directories(xformers_kernel PUBLIC xformers/third_party/cutlass/include)
target_link_libraries(xformers_kernel PUBLIC util xformers_autogen_impl)
set_property(TARGET xformers_kernel PROPERTY POSITION_INDEPENDENT_CODE ON)

