from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter

class bpetokenizer:
    '''A simple BPE tokenizer that uses a vocabulary file to encode text.'''
    def __init__(self, input_path: str, vocab_size: int, special_token: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_token = special_token
        self.pre_token_counts = Counter()

    def pre_tokenize(self):
        with open(self.input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                self.pre_token_counts.update(re.findall(PAT, chunk))
            print(f"Total pre-token counts: {self.pre_token_counts}")

    def compute_bpe_merges(self):
        # Implement the BPE merge algorithm here
        for token, freq in self.pre_token_counts.most_common(self.vocab_size):
            print(f"Token: {token}, Frequency: {freq}")

    

if __name__ == "__main__":
    tokenizer = bpetokenizer(input_path="data/debug_dataset.txt", vocab_size=1000, special_token=["<|endoftext|>"])
    tokenizer.pre_tokenize()