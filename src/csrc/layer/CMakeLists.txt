add_library(layer STATIC
	attention.cc
	ffn.cc
	gated_ffn.cc
)
set_property(TARGET layer PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(layer PUBLIC kernel util xformers_kernel)

if (MPI_FOUND AND NCCL_FOUND)
	target_link_libraries(layer PUBLIC nccl_utils)
endif()
