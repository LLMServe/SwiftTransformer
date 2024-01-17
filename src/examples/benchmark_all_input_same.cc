#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

#include "cuda_profiler_api.h"
#include <argparse/argparse.hpp>

#include "model/gpt/gpt_hyper_param.h"
#include "model/gpt/gpt_pagedattn_param.h"
#include "model/gpt/gpt_parallelism_param.h"
#include "model/gpt/gpt.h"
#include "util/nccl_utils.h"

#include "lib/common_gpt_hyper_params.h"
#include "lib/inference_batch.h"
#include "lib/simple_vocab_decoder.h"
#include "lib/utils.h"

constexpr int64_t NUM_TOTAL_BLOCKS = 1024;	// Allocate this number of blocks at the beginning

// The following tokens will be used to generate input sequences
// It corresponds to the following input sequence:
// "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty, twenty-one, twenty-two, twenty-three, twenty-four, twenty-five, twenty-six, twenty-seven, twenty-eight, twenty-nine, thirty,"...
// Input sequence = [TOKENS_IN_EVERY_INPUT[:input_len]] * batch_size
const std::vector<int64_t> TOKENS_IN_EVERY_INPUT = {3762, 6, 80, 6, 130, 6, 237, 6, 292, 6, 411, 6, 707, 6, 799, 6, 1117, 6, 2724, 6, 19353, 6, 11971, 6, 30361, 6, 31925, 6, 23843, 6, 32382, 6, 37100, 6, 34851, 6, 40126, 6, 10328, 6, 10328, 12, 1264, 6, 10328, 12, 7109, 6, 10328, 12, 9983, 6, 10328, 12, 10231, 6, 10328, 12, 9579, 6, 10328, 12, 13664, 6, 10328, 12, 17723, 6, 10328, 12, 19491, 6, 10328, 12, 22255, 6, 16984, 6, 16984, 12, 1264, 6, 16984, 12, 7109, 6, 16984, 12, 9983, 6, 16984, 12, 10231, 6, 16984, 12, 9579, 6, 16984, 12, 13664, 6, 16984, 12, 17723, 6, 16984, 12, 19491, 6, 16984, 12, 22255, 6, 24503, 6, 24503, 12, 1264, 6, 24503, 12, 7109, 6, 24503, 12, 9983, 6, 24503, 12, 10231, 6, 24503, 12, 9579, 6, 24503, 12, 13664, 6, 24503, 12, 17723, 6, 24503, 12, 19491, 6, 24503, 12, 22255, 6, 14865, 6, 14865, 12, 1264, 6, 14865, 12, 7109, 6, 14865, 12, 9983, 6, 14865, 12, 10231, 6, 14865, 12, 9579, 6, 14865, 12, 13664, 6, 14865, 12, 17723, 6, 14865, 12, 19491, 6, 14865, 12, 22255, 6, 33910, 6, 33910, 12, 1264, 6, 33910, 12, 7109, 6, 33910, 12, 9983, 6, 33910, 12, 10231, 6, 33910, 12, 9579, 6, 33910, 12, 13664, 6, 33910, 12, 17723, 6, 33910, 12, 19491, 6, 33910, 12, 22255, 6, 39676, 6, 39676, 12, 1264, 6, 39676, 12, 7109, 6, 39676, 12, 9983, 6, 39676, 12, 10231, 6, 39676, 12, 9579, 6, 39676, 12, 13664, 6, 39676, 12, 17723, 6, 39676, 12, 19491, 6, 39676, 12, 22255, 6, 42991, 6, 42991, 12, 1264, 6, 42991, 12, 7109, 6, 42991, 12, 9983, 6, 42991, 12, 10231, 6, 42991, 12, 9579, 6, 42991, 12, 13664, 6, 42991, 12, 17723, 6, 42991, 12, 19491, 6, 42991, 12, 22255, 6, 33035, 6, 33035, 12, 1264, 6, 33035, 12, 7109, 6, 33035, 12, 9983, 6, 33035, 12, 10231, 6, 33035, 12, 9579, 6, 33035, 12, 13664, 6, 33035, 12, 17723, 6, 33035, 12, 19491, 6, 33035, 12, 22255, 6, 65, 6317, 6, 65, 6317, 65, 6, 65, 6317, 80, 6, 65, 6317, 130, 6, 65, 6317, 237, 6, 65, 6317, 292, 6, 65, 6317, 411, 6, 65, 6317, 707, 6, 65, 6317, 799, 6, 65, 6317, 1117, 6, 65, 6317, 2724, 6, 65, 6317, 19353, 6, 65, 6317, 11971, 6, 65, 6317, 30361, 6, 65, 6317, 31925, 6, 65, 6317, 23843, 6, 65, 6317, 32382, 6, 65, 6317, 37100, 6, 65, 6317, 34851, 6, 65, 6317, 40126, 6, 65, 6317, 10328, 6, 65, 6317, 10328, 12, 1264, 6, 65, 6317, 10328, 12, 7109, 6, 65, 6317, 10328, 12, 9983, 6, 65, 6317, 10328, 12, 10231, 6, 65, 6317, 10328, 12, 9579, 6, 65, 6317, 10328, 12, 13664, 6, 65, 6317, 10328, 12, 17723, 6, 65, 6317, 10328, 12, 19491, 6, 65, 6317, 10328, 12, 22255, 6, 65, 6317, 16984, 6, 65, 6317, 16984, 12, 1264, 6, 65, 6317, 16984, 12, 7109, 6, 65, 6317, 16984, 12, 9983, 6, 65, 6317, 16984, 12, 10231, 6, 65, 6317, 16984, 12, 9579, 6, 65, 6317, 16984, 12, 13664, 6, 65, 6317, 16984, 12, 17723, 6, 65, 6317, 16984, 12, 19491, 6, 65, 6317, 16984, 12, 22255, 6, 65, 6317, 24503, 6, 65, 6317, 24503, 12, 1264, 6, 65, 6317, 24503, 12, 7109, 6, 65, 6317, 24503, 12, 9983, 6, 65, 6317, 24503, 12, 10231, 6, 65, 6317, 24503, 12, 9579, 6, 65, 6317, 24503, 12, 13664, 6, 65, 6317, 24503, 12, 17723, 6, 65, 6317, 24503, 12, 19491, 6, 65, 6317, 24503, 12, 22255, 6, 65, 6317, 14865, 6, 65, 6317, 14865, 12, 1264, 6, 65, 6317, 14865, 12, 7109, 6, 65, 6317, 14865, 12, 9983, 6, 65, 6317, 14865, 12, 10231, 6, 65, 6317, 14865, 12, 9579, 6, 65, 6317, 14865, 12, 13664, 6, 65, 6317, 14865, 12, 17723, 6, 65, 6317, 14865, 12, 19491, 6, 65, 6317, 14865, 12, 22255, 6, 65, 6317, 33910, 6, 65, 6317, 33910, 12, 1264, 6, 65, 6317, 33910, 12, 7109, 6, 65, 6317, 33910, 12, 9983, 6, 65, 6317, 33910, 12, 10231, 6, 65, 6317, 33910, 12, 9579, 6, 65, 6317, 33910, 12, 13664, 6, 65, 6317, 33910, 12, 17723, 6, 65, 6317, 33910, 12, 19491, 6, 65, 6317, 33910, 12, 22255, 6, 65, 6317, 39676, 6, 65, 6317, 39676, 12, 1264, 6, 65, 6317, 39676, 12, 7109, 6, 65, 6317, 39676, 12, 9983, 6, 65, 6317, 39676, 12, 10231, 6, 65, 6317, 39676, 12, 9579, 6, 65, 6317, 39676, 12, 13664, 6, 65, 6317, 39676, 12, 17723, 6, 65, 6317, 39676, 12, 19491, 6, 65, 6317, 39676, 12, 22255, 6, 65, 6317, 42991, 6, 65, 6317, 42991, 12, 1264, 6, 65, 6317, 42991, 12, 7109, 6, 65, 6317, 42991, 12, 9983, 6, 65, 6317, 42991, 12, 10231, 6, 65, 6317, 42991, 12, 9579, 6, 65, 6317, 42991, 12, 13664, 6, 65, 6317, 42991, 12, 17723, 6, 65, 6317, 42991, 12, 19491, 6, 65, 6317, 42991, 12, 22255, 6, 65, 6317, 33035, 6, 65, 6317, 33035, 12, 1264, 6, 65, 6317, 33035, 12, 7109, 6, 65, 6317, 33035, 12, 9983, 6, 65, 6317, 33035, 12, 10231, 6, 65, 6317, 33035, 12, 9579, 6, 65, 6317, 33035, 12, 13664, 6, 65, 6317, 33035, 12, 17723, 6, 65, 6317, 33035, 12, 19491, 6, 65, 6317, 33035, 12, 22255, 6, 80, 6317, 4};

// Variables related to the experiment
struct ExperimentParameters {
	int64_t input_len;	// The length (number of tokens) in a input sequence
	int64_t batch_size;	// The number of input sequences
	int64_t block_size;
	int64_t num_decoding_step;
};

// The following unordered_map stores parameters for different experiments
// The key is the name of the experiment
// while the value is a vector of ExperimentParameters, which are the parameters for the experiment
std::unordered_map<std::string, std::vector<ExperimentParameters>> experiment_params_map = {
	// For testing different block_sizes on small input len, small batch_size, and small decoding step
	{"diff_block_size_on_small_input_len_small_batch_size_small_decoding_step", {
		{32, 4, 1, 32},
		{32, 4, 2, 32},
		{32, 4, 4, 32},
		{32, 4, 8, 32},
		{32, 4, 16, 32},
		{32, 4, 32, 32}
	}},
	// For testing different block_sizes on small input len, large batch_size, and small decoding step
	{"diff_block_size_on_small_input_len_large_batch_size_small_decoding_step", {
		{32, 128, 1, 32},
		{32, 128, 2, 32},
		{32, 128, 4, 32},
		{32, 128, 8, 32},
		{32, 128, 16, 32},
		{32, 128, 32, 32}
	}},
	// For testing different block_sizes on large input len, small batch_size, and small decoding step
	{"diff_block_size_on_large_input_len_small_batch_size_small_decoding_step", {
		{512, 4, 1, 16},
		{512, 4, 2, 16},
		{512, 4, 4, 16},
		{512, 4, 8, 16},
		{512, 4, 16, 16},
		{512, 4, 32, 16}
	}},
	// For testing different block_sizes on large input len, large batch_size, and small decoding step
	{"diff_block_size_on_large_input_len_large_batch_size_small_decoding_step", {
		{512, 32, 1, 16},
		{512, 32, 2, 16},
		{512, 32, 4, 16},
		{512, 32, 8, 16},
		{512, 32, 16, 16},
		{512, 32, 32, 16}
	}},
	// For testing different block_sizes on small input len, small batch_size, and large decoding step
	{"diff_block_size_on_small_input_len_small_batch_size_large_decoding_step", {
		{32, 4, 1, 256},
		{32, 4, 2, 256},
		{32, 4, 4, 256},
		{32, 4, 8, 256},
		{32, 4, 16, 256},
		{32, 4, 32, 256}
	}},
	// For testing different block_sizes on small input len, large batch_size, and large decoding step
	{"diff_block_size_on_small_input_len_large_batch_size_large_decoding_step", {
		{32, 128, 1, 256},
		{32, 128, 2, 256},
		{32, 128, 4, 256},
		{32, 128, 8, 256},
		{32, 128, 16, 256},
		{32, 128, 32, 256}
	}},
	// For testing different batch_size on large input len and large decoding step on opt-1.3b
	{"diff_batch_size_1.3b", {
		{256, 1, 16, 16},
		{256, 2, 16, 16},
		{256, 4, 16, 16},
		{256, 8, 16, 16},
		{256, 16, 16, 16},
		{256, 32, 16, 16},
		{256, 48, 16, 16},
		{256, 64, 16, 16},
		{256, 80, 16, 16},
		{256, 96, 16, 16},
		{256, 112, 16, 16},
		{256, 128, 16, 16}
	}},
	// For testing different batch_size on large input len and large decoding step on opt-6.7b
	{"diff_batch_size_6.7b", {
		{128, 1, 16, 16},
		{128, 2, 16, 16},
		{128, 4, 16, 16},
		{128, 8, 16, 16},
		{128, 16, 16, 16},
		{128, 32, 16, 16},
		{128, 48, 16, 16},
		{128, 64, 16, 16},
		{128, 80, 16, 16},
		{128, 96, 16, 16},
		// {128, 112, 16, 16},
	}},
	{"diff_batch_size_13b", {
		{256, 1, 16, 16},
		{256, 2, 16, 16},
		{256, 4, 16, 16},
		{256, 8, 16, 16},
		{256, 16, 16, 16},
		{256, 32, 16, 16},
		{256, 48, 16, 16},
		{256, 64, 16, 16},
	}},
	// A testcase for benchmarking context attention kernel
	{"benchmark_context_kernel", {
		{256, 128, 16, 16}
	}},
	// A testcase for benchmarking decoding attention kernel
	{"benchmark_decoding_kernel", {
		{16, 128, 16, 256}
	}}
};

// fake_main - the main function
// We need this since we want to use T (the inference datatype, FP16 or FP32) during initialization
template<typename T>
int fake_main(
	int argc, char* argv[],
	std::string model_weight_path,
	std::string num_of_params,
	std::string vocab_json_path,
	std::string precision,
	std::string exp_params_name
) {
	// Initialize MPI
	int rank = 0, world_size = 1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	int64_t num_tensor_parallel = world_size;
	printf("Running with %d GPU\n", world_size);

	// Check if the number of tensor parallel is valid
	if (num_tensor_parallel != 1 && num_tensor_parallel != 2 && num_tensor_parallel != 4 && num_tensor_parallel != 8) {
		printf("Invalid number of tensor parallel: %ld\n", num_tensor_parallel);
		printf("Valid choices: 1, 2, 4, 8\n");
		return 1;
	}

	bool is_debug = false;
#ifdef DEBUG
	is_debug = true;
#endif

	// Choose experiment parameters
	if (!experiment_params_map.count(exp_params_name)) {
		printf("Invalid experiment parameters name: %s\n", exp_params_name.c_str());
		printf("Valid experiment parameters name:\n");
		for (auto it = experiment_params_map.begin(); it != experiment_params_map.end(); ++it) {
			printf("\t%s\n", it->first.c_str());
		}
		return 1;
	}
	std::vector<ExperimentParameters> exp_params = experiment_params_map[exp_params_name];

	// Choose hyper parameters
	st::model::GptHyperParam hyper_param = str2hyperparam(num_of_params);
	if (hyper_param.vocab_size == -1) return 1;

	st::model::GptParallelismParam parallel_param;
	ncclUniqueId nccl_id, pp_nccl_id; // TODO(sunyh): support tp here

	// Set parallel mode
	if (num_tensor_parallel != 1) {
		// get device count
		int device_count;
		CUDA_CHECK(cudaGetDeviceCount(&device_count));
		if (device_count < num_tensor_parallel) {
			printf("num_tensor_parallel is set to %ld, but only %d GPUs are available\n", num_tensor_parallel, device_count);
			return 1;
		}

		if (world_size != num_tensor_parallel) {
			printf("num_tensor_parallel is set to %ld, but %d MPI ranks are running\n", num_tensor_parallel, world_size);
			return 1;
		}

		parallel_param.tensor_para_size = world_size;
		parallel_param.tensor_para_rank = rank;
		parallel_param.init_by_hyper_param(hyper_param);

		// Init NCCL
		if (rank == 0) {
			ncclGetUniqueId(&nccl_id);
			ncclGetUniqueId(&pp_nccl_id);// TODO(sunyh): delete this hack
		}
		MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
		
		cudaSetDevice(rank);
	}

	// Load the model
	printf("Loading model...\n");
	st::model::Gpt<T> model(
		hyper_param,
		{.block_size = 0, .max_num_block_per_req = 0},	// The PagedAttnParam here will be overwritten later
		parallel_param
	);

	if (num_tensor_parallel != 1) {
		model.init_communicator(nccl_id, pp_nccl_id);
	}

	if (model_weight_path != "dummy") {
		model.loadWeight(model_weight_path.c_str());
	} else {
		printf("No model path specified, using dummy weights\n");
		model.initDummyWeight();
	}

	// Load the decoder
	SimpleVocabDecoder* vocab_decoder_ptr = vocab_json_path == "" ? nullptr : new SimpleVocabDecoder(vocab_json_path);

	std::vector<std::pair<ExperimentParameters, RuntimeUsage>> results;
	for (ExperimentParameters exp_param : exp_params) {
		std::vector<int64_t> input_seq = TOKENS_IN_EVERY_INPUT;
		while ((int64_t)input_seq.size() > exp_param.input_len) input_seq.pop_back();
		while ((int64_t)input_seq.size() < exp_param.input_len) input_seq.push_back(input_seq.back());
		std::vector<std::vector<int64_t>> input_tokens_batched;
		for (int64_t i = 0; i < exp_param.batch_size; ++i) {
			input_tokens_batched.push_back(input_seq);
		}
		st::model::GptPagedAttnParam pagedattn_param = {
			.block_size = exp_param.block_size,
			.max_num_block_per_req = (exp_param.input_len + exp_param.num_decoding_step + exp_param.block_size - 1) / exp_param.block_size + 1
		};
		model.setPagedattnParam(pagedattn_param);
		int64_t num_total_blocks = pagedattn_param.max_num_block_per_req * exp_param.batch_size + 1;
		std::vector<std::vector<int64_t>> output_tokens_batched;
		if (rank == 0) {
			printf("--------\n");
			printf("Exp params:\n");
			printf("\tinput_len: %ld\n", exp_param.input_len);
			printf("\tbatch_size: %ld\n", exp_param.batch_size);
			printf("\tblock_size: %ld\n", exp_param.block_size);
			printf("\tnum_decoding_step: %ld\n", exp_param.num_decoding_step);
			printf("Warming up...\n");
		}
		run_batched_inference(
			output_tokens_batched,
			input_tokens_batched,
			model,
			hyper_param,
			pagedattn_param,
			parallel_param,
			exp_param.num_decoding_step,
			-1,
			num_total_blocks,
			is_debug,
			false
		);
		if (rank == 0) {
			printf("Running...\n");
		}
		RuntimeUsage runtime_usage = run_batched_inference(
			output_tokens_batched,
			input_tokens_batched,
			model,
			hyper_param,
			pagedattn_param,
			parallel_param,
			exp_param.num_decoding_step,
			-1,
			num_total_blocks,
			is_debug,
			false
		);
		if (rank == 0) {
			printf("Time usage:\n");
			printf("\tContext stage: %.2f ms\n", runtime_usage.context_stage_time);
			printf("\tRegression stage: %.2f ms\n", runtime_usage.decoding_stage_time);
			printf("\tTotal: %.2f ms\n", runtime_usage.total_time);
			printf("Model output:\n");
			printf("(");
			for (auto input_token : input_tokens_batched[0]) {
				if (vocab_decoder_ptr)
					printf("%s", vocab_decoder_ptr->decode(input_token).c_str());
				else 
					printf("%ld ", input_token);
			}
			printf(")");
			for (auto output_token : output_tokens_batched[0]) {
				if (vocab_decoder_ptr)
					printf("%s", vocab_decoder_ptr->decode(output_token).c_str());
				else
					printf("%ld ", output_token);
			}
			printf("  (%ld tokens generated)", output_tokens_batched[0].size());
			printf("\n");
		}
		results.push_back({exp_param, runtime_usage});
	}

	// Print results
	if (rank == 0) {
		printf("Experiment parameters: %s\n", exp_params_name.c_str());
		printf("Results:\n");
		printf("  block_size   input_len  batch_size decode_step context_t(ms)  decode_t(ms)   total_t(ms)\n");
		for (auto result : results) {
			printf("%12ld%12ld%12ld%12ld%14.2f%14.2f%14.2f\n",
				result.first.block_size,
				result.first.input_len,
				result.first.batch_size,
				result.first.num_decoding_step,
				result.second.context_stage_time,
				result.second.decoding_stage_time,
				result.second.total_time
			);
		}

		// Format the result with \t for easy copy-paste to spreadsheet
		printf("\nThe following table is formatted with tabs(\\t) for easy copy-paste to Excel.\n");
		for (auto result : results) {
			printf("%ld\t%ld\t%ld\t%ld\t%.2f\t%.2f\t%.2f\n",
				result.first.block_size,
				result.first.input_len,
				result.first.batch_size,
				result.first.num_decoding_step,
				result.second.context_stage_time,
				result.second.decoding_stage_time,
				result.second.total_time
			);
		}
	}

	if(num_tensor_parallel != 1) {
		MPI_Finalize();
	}

	return 0;
}

// main - Dispatch to fake_main based on precision (argv[4])
int main(int argc, char* argv[]) {
#ifdef DEBUG
	// Show a warning when debug mode is enabled - We are profiling now!
	#warning "DEBUG mode is enabled"
	#warning "This will cause a significant slowdown"
	fprintf(stderr, "[WARNING] DEBUG mode is enabled\n");
	fprintf(stderr, "[WARNING] This will cause a significant slowdown\n");
#endif
	argparse::ArgumentParser parser("benchmark_all_input_same", "0.1");
	parser.add_argument("model_name")
		.help("Name of this model")
		.required();
	parser.add_argument("precision")
		.help("Precision used for inference (fp16 or fp32)")
		.required();
	parser.add_argument("exp_params_name")
		.help("Name of the experiment parameters")
		.required();
	parser.add_argument("-m", "--model_path")
		.help("Path to the model weight file (type 'dummy' to use dummy weights)")
		.default_value(std::string("dummy"));
	parser.add_argument("-v", "--vocab_json_path")
		.help("Path to the vocab json file")
		.default_value(std::string(""));
	
	try {
		parser.parse_args(argc, argv);
	} catch (const std::runtime_error& err) {
  		std::cerr << err.what() << std::endl;
  		std::cerr << parser;
  		std::exit(1);
	}

	std::string precision = parser.get("precision");
	if (precision == "fp16") {
		return fake_main<half>(
			argc, argv,
			parser.get("model_path"),
			parser.get("model_name"),
			parser.get("vocab_json_path"),
			parser.get("precision"),
			parser.get("exp_params_name")
		);
	} else if (precision == "fp32") {
		return fake_main<float>(
			argc, argv,
			parser.get("model_path"),
			parser.get("model_name"),
			parser.get("vocab_json_path"),
			parser.get("precision"),
			parser.get("exp_params_name")
		);
	} else {
		printf("Invalid precision: %s\n", precision.c_str());
		printf("Valid precision: fp16, fp32\n");
		return 1;
	}
	return 0;
}
