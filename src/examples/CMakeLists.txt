add_library(st_args STATIC lib/st_args.cc)
target_link_libraries(st_args PRIVATE)

add_executable(run_gpt run_gpt.cc lib/inference_batch.cc)
target_link_libraries(run_gpt model_gpt nlohmann_json::nlohmann_json argparse st_args)

add_executable(benchmark_all_input_same benchmark_all_input_same.cc lib/inference_batch.cc)
target_link_libraries(benchmark_all_input_same model_gpt nlohmann_json::nlohmann_json argparse)
