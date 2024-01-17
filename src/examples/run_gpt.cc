#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <mpi.h>

#include <argparse/argparse.hpp>

#include "cuda_profiler_api.h"

#include "model/gpt/gpt_hyper_param.h"
#include "model/gpt/gpt_pagedattn_param.h"
#include "model/gpt/gpt.h"

#include "lib/common_gpt_hyper_params.h"
#include "lib/inference_batch.h"
#include "lib/simple_vocab_decoder.h"
#include "lib/st_args.h"

constexpr int64_t NUM_TOTAL_BLOCKS = 1024;	// Allocate this number of blocks at the beginning
// constexpr int64_t END_TOKEN = 2;
constexpr int64_t END_TOKEN = 50118;	// TODO WHY NOT 2?
constexpr int64_t MAX_DECODING_STEP = 128;

// fake_main - the main function
// We need this since we want to use T (the inference datatype, FP16 or FP32) during initialization
template<typename T>
int fake_main(st::example::RunArgs args) {

#ifdef DEBUG
	args.is_debug = true;
#endif

	// Get the rank and size
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("Running with %d GPUs\n", size);
	
	// Identify the tensor parallel rank and size
	int tp_rank, tp_size;
	MPI_Comm tp_comm;
	MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &tp_comm);
	MPI_Comm_rank(tp_comm, &tp_rank);
	MPI_Comm_size(tp_comm, &tp_size);

	int pp_rank, pp_size;
	MPI_Comm pp_comm;
	MPI_Comm_split(MPI_COMM_WORLD, tp_rank, rank, &pp_comm);
	MPI_Comm_rank(pp_comm, &pp_rank);
	MPI_Comm_size(pp_comm, &pp_size);

	// Init parallel config
	st::model::GptParallelismParam parallel_param(tp_size, tp_rank, pp_size, pp_rank);
	parallel_param.init_by_hyper_param(args.hyper_param);

	// Init NCCL
	ncclUniqueId tp_id, pp_id; //TODO(sunyh): set up pp
	if (rank == 0) {
		ncclGetUniqueId(&tp_id);
		ncclGetUniqueId(&pp_id);
	}
	MPI_Bcast(&tp_id, sizeof(tp_id), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pp_id, sizeof(pp_id), MPI_BYTE, 0, MPI_COMM_WORLD);

	// Init CUDA
	cudaSetDevice(tp_rank);
	
	// Read input tokens
	printf("Reading input tokens...\n");
	std::vector<std::vector<int64_t>> input_tokens_batched;
	std::ifstream input_file(args.input_path);
	if (!input_file.is_open()) {
		printf("Failed to open input file: %s\n", args.input_path.c_str());
		exit(1);
	}
	std::string line;
	while (std::getline(input_file, line)) {
		if (line.empty()) {
			continue;
		}
		std::vector<int64_t> tokens;
		std::istringstream iss(line);
		int64_t token;
		while (iss >> token) {
			tokens.push_back(token);
		}
		input_tokens_batched.push_back(tokens);
	}

	// Init pagedattention config
	st::model::GptPagedAttnParam pagedattn_param = {
		.block_size = 16,
		.max_num_block_per_req = 1024
	};

	// Load the model
	printf("Loading model...\n");

	st::model::Gpt<T> model(args.hyper_param, pagedattn_param, parallel_param);
	model.init_communicator(tp_id, pp_id);
	model.loadWeight(args.model_weight_path);
	sync_check_cuda_error();

	// Generate!
	std::vector<std::vector<int64_t>> output_tokens_batched;
	RuntimeUsage runtime_usage = run_batched_inference(
		output_tokens_batched,
		input_tokens_batched,
		model,
		args.hyper_param,
		pagedattn_param,
		parallel_param,
		MAX_DECODING_STEP,
		END_TOKEN,
		NUM_TOTAL_BLOCKS,
		args.is_debug
	);

	// Print result
	int64_t batch_size = input_tokens_batched.size();
	SimpleVocabDecoder* vocab_decoder_ptr = args.vocab_json_path == "" ? nullptr : new SimpleVocabDecoder(args.vocab_json_path);
	for (int64_t i = 0; i < batch_size; ++i) {
		printf("[%2ld] ", i);
		printf("(");
		for (auto input_token : input_tokens_batched[i]) {
			if (vocab_decoder_ptr)
				printf("%s", vocab_decoder_ptr->decode(input_token).c_str());
			else 
				printf("%ld ", input_token);
		}
		printf(")");
		for (auto output_token : output_tokens_batched[i]) {
			if (vocab_decoder_ptr)
				printf("%s", vocab_decoder_ptr->decode(output_token).c_str());
			else
				printf("%ld ", output_token);
		}
		printf("  (%ld tokens generated)", output_tokens_batched[i].size());
		printf("\n");
	}

	// Print time usage
	printf("Time usage:\n");
	printf("\tContext stage: %.2f ms\n", runtime_usage.context_stage_time);
	printf("\tRegression stage: %.2f ms\n", runtime_usage.decoding_stage_time);
	printf("\tTotal: %.2f ms\n", runtime_usage.total_time);
	printf("Done!\n");

	return 0;
}

// main - Dispatch to fake_main based on precision (argv[4])
int main(int argc, char* argv[]) {
#ifdef DEBUG
	// Show a warning when debug mode is enabled
	#warning "DEBUG mode is enabled"
	#warning "This will cause a significant slowdown"
	fprintf(stderr, "[WARNING] DEBUG mode is enabled\n");
	fprintf(stderr, "[WARNING] This will cause a significant slowdown\n");
#endif
	argparse::ArgumentParser program("run_gpt", "0.1");

	program.add_description("This programs loads a model, feeds "
		   "it with given input sequences, and prints the generated tokens.\n"
		   "example: "
		   );

	program.add_argument("model_path")
		.help("Path to the model weight file")
		.required();
	program.add_argument("model_name")
		.help("Name of the model")
		.required();
	program.add_argument("-v", "--vocab_json_path")
		.help("Path to the vocab json file")
		.default_value(std::string(""));
	program.add_argument("precision")
		.help("Precision used for inference (fp16 or fp32)")
		.required();
	program.add_argument("input_path")
		.help("Path to the input file")
		.required();
	program.add_argument("-d", "--debug")
		.help("Enable debug mode")
		.default_value(false)
		.implicit_value(true);

	try {
		program.parse_args(argc, argv);
	} catch (const std::runtime_error& err) {
  		std::cerr << err.what() << std::endl;
  		std::cerr << program;
  		std::exit(1);
	}

	// Init MPI and NCCL
	int stat = MPI_Init(&argc, &argv);
	if (stat != MPI_SUCCESS) {
		printf("Failed to init MPI\n");
		return 1;
	}

	st::example::RunArgs args(
		program.get<std::string>("model_path") + "/", // NOTE(sunyh): NO WINDOWS SUPPORT FOR NOW
		program.get<std::string>("vocab_json_path"),
		program.get<std::string>("input_path"),
		program.get<std::string>("model_name"),
		program.get<std::string>("precision"),
		program.get<bool>("debug")
	);
	
	switch (args.precision){
		case st::example::Precision::FP16:
			fake_main<half>(args);
			break;
		case st::example::Precision::FP32:
			fake_main<float>(args);
			break;
		default:
			printf("Invalid precision: %s\n", st::example::precision_to_string(args.precision).c_str());
			printf("Valid precision: fp16, fp32\n");
			return 1;
	}

	MPI_Finalize();
	
	return 0;
}
