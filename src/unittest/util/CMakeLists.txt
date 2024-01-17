add_executable(unittest_util cublas_wrapper.cc)
target_link_libraries(unittest_util PUBLIC util gtest_main)
