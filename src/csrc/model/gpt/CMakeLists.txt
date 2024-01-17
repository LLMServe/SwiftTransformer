add_library(model_gpt gpt_weight.cc gpt.cc gptop_base.cc)
target_link_libraries(model_gpt layer kernel util)
set_property(TARGET model_gpt PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(opt)
add_subdirectory(llama2)
add_subdirectory(gpt2)
