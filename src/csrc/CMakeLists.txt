add_subdirectory(kernel)
add_subdirectory(layer)
add_subdirectory(model)
add_subdirectory(util)

add_library(st_pybinding SHARED pybinding.cc)

target_link_libraries(st_pybinding
    model_gpt
    model_opt
    model_llama2
    model_gpt2
    py_nccl_utils
    py_swapping
    py_block_migration
)

# Set the output directory for the shared library
set_target_properties(st_pybinding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)