from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter

class bpetokenizer:
    '''A simple BPE tokenizer that uses a vocabulary file to encode text.'''
    def __init__(self, input_path: str, vocab_size: int, special_token: list[str]):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_token = special_token
        self.split_bytes_counts = {}
        
    def pre_tokenize(self):
        '''Pre-tokenize the input text file and count the frequency of each pre-token.'''
        pre_token_counts = Counter()

        with open(self.input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                pre_token_counts.update(re.findall(PAT, chunk))
            print(f"Total pre-token counts: {pre_token_counts}")

        for word, freq in pre_token_counts.items():
            self.split_bytes_counts[self._split_bytes(word)] = freq

        print(f"Total split bytes: {self.split_bytes_counts}")

    def _split_bytes(self, word: str) -> tuple[bytes,...]:
        '''Split a word into its constituent bytes.'''
        return tuple(bytes([b]) for b in word.encode("utf-8"))

    def merge_token(self):
        '''Merge the most frequent pairs of bytes into new tokens until the vocabulary size is reached.'''
        ...

    

if __name__ == "__main__":
    tokenizer = bpetokenizer(input_path="data/debug_dataset.txt", vocab_size=100, special_token=["<|endoftext|>"])
    tokenizer.pre_tokenize()