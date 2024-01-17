add_library(model_llama2 llama2op.cc)
target_link_libraries(model_llama2 layer kernel util model_gpt "${TORCH_LIBRARIES}")
set_property(TARGET model_llama2 PROPERTY POSITION_INDEPENDENT_CODE ON)
