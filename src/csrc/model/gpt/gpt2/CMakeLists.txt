add_library(model_gpt2 gpt2op.cc)
target_link_libraries(model_gpt2 layer kernel util model_gpt "${TORCH_LIBRARIES}")
set_property(TARGET model_gpt2 PROPERTY POSITION_INDEPENDENT_CODE ON)
