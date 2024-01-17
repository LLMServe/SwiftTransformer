import os, sys
import lib.gpt_token_encoder as encoder

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage: python3 encode_input.py <vocab_file> <bpe_file>")
		sys.exit(1)

	vocab_file = sys.argv[1]
	bpe_file = sys.argv[2]
	enc = encoder.get_encoder(vocab_file, bpe_file)
	for line in sys.stdin:
		line = line.strip()
		print(" ".join(map(str, enc.encode(line))))
		