"""
ModelForge AI - ML Engine: Byte-Pair Encoding (BPE) Subword Tokenizer
Implements Sennrich et al. Neural Machine Translation with Subword Units (BPE)
from first principles with vocabulary generation, frequency merge rules, and regex pre-tokenization.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re
from collections import Counter, defaultdict


class BytePairEncoder:
    """Byte-Pair Encoding subword tokenizer."""

    def __init__(self, vocab_size: int = 1000):
        self.target_vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<mask>": 4,
        }

    def _get_stats(self, word_freqs: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
        """Count frequencies of adjacent symbol pairs."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: Tuple[bytes, bytes],
        word_freqs: Dict[Tuple[bytes, ...], int],
    ) -> Dict[Tuple[bytes, ...], int]:
        """Replace all occurrences of symbol pair with merged symbol."""
        new_word_freqs = {}
        p0, p1 = pair
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(p0 + p1)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, corpus: List[str]):
        """Train BPE vocabulary merges on raw string corpus."""
        # 1. Initialize base vocabulary with 256 individual UTF-8 bytes + special tokens
        self.vocab = {idx: name.encode("utf-8") for name, idx in self.special_tokens.items()}
        next_idx = len(self.special_tokens)

        for b in range(256):
            self.vocab[next_idx] = bytes([b])
            next_idx += 1

        # 2. Pre-tokenize corpus into words and character byte sequences
        word_counts = Counter()
        for text in corpus:
            words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            for w in words:
                byte_seq = tuple(bytes([b]) for b in w.encode("utf-8"))
                word_counts[byte_seq] += 1

        # 3. Iteratively merge most frequent symbol pairs
        word_freqs = dict(word_counts)
        num_merges = self.target_vocab_size - len(self.vocab)

        for i in range(max(0, num_merges)):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break

            word_freqs = self._merge_vocab(best_pair, word_freqs)
            merged_bytes = best_pair[0] + best_pair[1]

            self.merges[best_pair] = next_idx
            self.vocab[next_idx] = merged_bytes
            next_idx += 1

        return self

    def encode(self, text: str) -> List[int]:
        """Tokenize text into subword token IDs."""
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        inv_vocab = {v: k for k, v in self.vocab.items()}
        token_ids = []

        for w in words:
            symbols = list(tuple(bytes([b]) for b in w.encode("utf-8")))

            while len(symbols) > 1:
                # Find best matching merge
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                valid_pairs = [(self.merges[p], p) for p in pairs if p in self.merges]

                if not valid_pairs:
                    break

                _, best_pair = min(valid_pairs)
                p0, p1 = best_pair

                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == p0 and symbols[i + 1] == p1:
                        new_symbols.append(p0 + p1)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            for sym in symbols:
                token_ids.append(inv_vocab.get(sym, self.special_tokens["<unk>"]))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to human-readable UTF-8 string."""
        byte_chunks = []
        for tid in token_ids:
            if tid in self.vocab:
                byte_chunks.append(self.vocab[tid])

        raw_bytes = b"".join(byte_chunks)
        return raw_bytes.decode("utf-8", errors="replace")
