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
        self.vocab = {bytes([i]): i for i in range(256)} # token -> token_id
        
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

        while len(self.vocab) < self.vocab_size:
            # 1. Count the frequency of each pair of bytes
            pair_counts = self._count_pairs()

            if not pair_counts:
                break

            # 2. Find the most frequent pair of bytes
            best_pair = max(pair_counts, key=lambda pair : (pair_counts[pair], pair))

            # 3. Merge the most frequent pair of bytes into a new token
            self.split_bytes_counts = self._merge_pair(self.split_bytes_counts, best_pair)

            # 4. Add the new token to the vocabulary
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

        print(f"Final vocabulary size: {self.split_bytes_counts}")
        
    def _count_pairs(self) -> Counter:
        '''Count the frequncy of each pair of bytes'''
        pair_counts = Counter()

        for token, freq in self.split_bytes_counts.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += freq

        return pair_counts

    def _merge_pair(self, split_bytes_counts: dict[tuple[bytes,...], int],
                    pair: tuple[bytes, bytes]) -> dict[tuple[bytes,...], int]:
        new_split_bytes_counts = {}

        for token, freq in split_bytes_counts.items():
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                    new_token.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_split_bytes_counts[tuple(new_token)] = new_split_bytes_counts.get(tuple(new_token), 0) + freq

        return new_split_bytes_counts


if __name__ == "__main__":
    tokenizer = bpetokenizer(input_path="data/debug_dataset.txt", vocab_size=300, special_token=["<|endoftext|>"])
    tokenizer.pre_tokenize()
    tokenizer.merge_token()