#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// SimpleVocabDecoder - A tiny (inaccurate) vocab decoder
// It just maps token ids to tokens according to the vocab json file
// Mainly for debugging
class SimpleVocabDecoder {
private:
	std::unordered_map<int64_t, std::string> vocab_map;
public:
	SimpleVocabDecoder(const std::string &vocab_json_path) {
		std::ifstream vocab_json_file(vocab_json_path);
		if (!vocab_json_file.is_open()) {
			printf("Failed to open vocab json file: %s\n", vocab_json_path.c_str());
			exit(1);
		}
		json vocab_json;
		vocab_json_file >> vocab_json;
		for (auto it = vocab_json.begin(); it != vocab_json.end(); it++) {
			vocab_map[it.value().get<int64_t>()] = it.key();
		}
	}

	inline std::string decode(int64_t token) const {
		if (vocab_map.find(token) == vocab_map.end()) {
			return "<UNK>";
		}
		std::string result = vocab_map.at(token);
		if ((int)result[0] == -60 && (int)result[1] == -96) {	// Ġ
			result = " " + result.substr(2);
		}
		return result;
	}
};
