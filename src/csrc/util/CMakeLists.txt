add_library(util STATIC cublas_wrapper.cc)
target_link_libraries(util PUBLIC CUDA::cublas)
set_property(TARGET util PROPERTY POSITION_INDEPENDENT_CODE  ON)

add_library(nccl_utils STATIC nccl_utils.cc)
set_property(TARGET nccl_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET nccl_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(nccl_utils PUBLIC ${MPI_CXX_LIBRARIES} ${NCCL_LIBRARIES})
target_include_directories(nccl_utils PUBLIC ${MPI_CXX_INCLUDE_DIRS} ${NCCL_INCLUDE_DIRS})

add_library(py_nccl_utils STATIC py_nccl.cc)
set_property(TARGET py_nccl_utils PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET py_nccl_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(py_nccl_utils PUBLIC ${MPI_CXX_LIBRARIES} ${NCCL_LIBRARIES})

add_library(py_swapping STATIC py_swapping.cc)
target_link_libraries(py_swapping PUBLIC util)
set_property(TARGET py_swapping PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(py_block_migration STATIC py_block_migration.cc)
target_link_libraries(py_block_migration PUBLIC util)
set_property(TARGET py_block_migration PROPERTY POSITION_INDEPENDENT_CODE ON)
