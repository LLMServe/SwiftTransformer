add_library(model_opt optop.cc)
target_link_libraries(model_opt layer kernel util model_gpt "${TORCH_LIBRARIES}")
set_property(TARGET model_opt PROPERTY POSITION_INDEPENDENT_CODE ON)
