"""
Byte-Level Byte-Pair Encoding (BPE) Tokenizer for URA-Shree.
Implemented completely from scratch with zero external LLM dependencies.
Supports special token preservation, regex pre-tokenization, merge tables, and lossless UTF-8 decoding.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Set, Optional, Union

from tokenizer.vocabulary import Vocabulary, SPECIAL_TOKENS, bytes_to_unicode


class BPETokenizer:
    """
    Byte-Level Byte-Pair Encoding (BPE) Tokenizer.
    Guarantees zero out-of-vocabulary (OOV) errors by falling back to base byte representations.
    """

    # GPT-style pre-tokenization regex pattern: contractions, words with underscores/numbers, punctuation, whitespace
    PAT = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens = special_tokens if special_tokens is not None else list(SPECIAL_TOKENS)
        self.special_tokens_set = set(self.special_tokens)

        # Byte to unicode mappings for lossless UTF-8 character conversion
        self.byte_encoder: Dict[int, str] = bytes_to_unicode()
        self.byte_decoder: Dict[str, int] = {v: k for k, v in self.byte_encoder.items()}

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary()
            self._init_base_vocab()

        # Merges stored as a list of (token_a, token_b) tuples
        self.merges: List[Tuple[str, str]] = merges if merges is not None else []
        # Fast lookup for merge priority: (token_a, token_b) -> rank
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        # Regex pattern to match any special token
        self._compile_special_pattern()

    def _init_base_vocab(self) -> None:
        """Initialize vocabulary with special tokens and all 256 byte tokens."""
        # 1. Register special tokens
        for token in self.special_tokens:
            self.vocab.add_token(token)

        # 2. Register all 256 byte tokens via byte_encoder
        for b in range(256):
            self.vocab.add_token(self.byte_encoder[b])

    def _compile_special_pattern(self) -> None:
        """Compile regex pattern to locate special tokens within raw text."""
        if self.special_tokens:
            escaped = [re.escape(tok) for tok in self.special_tokens]
            self.special_pattern = re.compile(f"({'|'.join(escaped)})")
        else:
            self.special_pattern = None

    @property
    def pad_id(self) -> int:
        return self.vocab.get_id("<|pad|>")

    @property
    def bos_id(self) -> int:
        return self.vocab.get_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self.vocab.get_id("<|eos|>")

    @property
    def unk_id(self) -> int:
        return self.vocab.get_id("<|unk|>")

    def __len__(self) -> int:
        return len(self.vocab)

    @staticmethod
    def _get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """Return set of adjacent symbol pairs in a word tuple."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token_str: str) -> List[str]:
        """
        Applies learned BPE merge operations to a sequence of characters.
        Merges the highest-ranked (lowest rank index) pair iteratively.
        """
        word = tuple(token_str)
        pairs = self._get_pairs(word)

        if not pairs:
            return list(word)

        while True:
            # Find candidate pair with the lowest rank (earliest merged during training)
            min_pair = None
            min_rank = float("inf")
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            # If no pairs match any merge rule, we are done
            if min_pair is None or min_pair not in self.bpe_ranks:
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        return list(word)

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """
        Encodes input string into a list of integer token IDs.
        Special tokens in `allowed_special` (or all special tokens by default) are preserved intact.
        """
        if allowed_special is None:
            allowed_special = self.special_tokens_set

        if not text:
            return []

        token_ids: List[int] = []

        # Split text around special tokens if any exist
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            # If this chunk is a recognized special token
            if part in self.special_tokens_set:
                if part in allowed_special:
                    token_ids.append(self.vocab.get_id(part))
                else:
                    # Treat as normal text if not allowed
                    part_bytes = part.encode("utf-8")
                    for b in part_bytes:
                        token_ids.append(self.vocab.get_id(self.byte_encoder[b]))
                continue

            # Otherwise, run standard pre-tokenization regex on normal text
            matches = self.PAT.findall(part)
            for match in matches:
                # Convert match to byte string, then to unicode character string
                encoded_bytes = match.encode("utf-8")
                byte_chars = "".join(self.byte_encoder[b] for b in encoded_bytes)

                # Apply BPE merge rules
                bpe_tokens = self._bpe(byte_chars)
                for bpe_token in bpe_tokens:
                    if bpe_token in self.vocab:
                        token_ids.append(self.vocab.get_id(bpe_token))
                    else:
                        # Byte-level fallback guarantees no uncaught tokens
                        for char in bpe_token:
                            token_ids.append(self.vocab.get_id(char, default=self.unk_id))

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes a list of token IDs back into a UTF-8 string.
        """
        byte_list: List[int] = []
        result_chars: List[str] = []

        for tid in token_ids:
            try:
                token_str = self.vocab.get_token(tid)
            except KeyError:
                token_str = "<|unk|>"

            if token_str in self.special_tokens_set:
                if not skip_special_tokens:
                    # Flush any accumulated bytes before outputting special token
                    if byte_list:
                        result_chars.append(bytes(byte_list).decode("utf-8", errors="replace"))
                        byte_list = []
                    result_chars.append(token_str)
                continue

            # Convert token string characters back to bytes
            for char in token_str:
                if char in self.byte_decoder:
                    byte_list.append(self.byte_decoder[char])
                else:
                    # Fallback for unexpected unicode
                    if byte_list:
                        result_chars.append(bytes(byte_list).decode("utf-8", errors="replace"))
                        byte_list = []
                    result_chars.append(char)

        if byte_list:
            result_chars.append(bytes(byte_list).decode("utf-8", errors="replace"))

        return "".join(result_chars)

    def save(self, directory: str) -> str:
        """
        Saves the tokenizer vocabulary, merges, and special tokens to a directory.
        Returns the path to the saved tokenizer.json file.
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, "tokenizer.json")

        data = {
            "special_tokens": self.special_tokens,
            "vocab": self.vocab.token_to_id,
            "merges": [f"{p[0]} {p[1]}" for p in self.merges],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath_or_dir: str) -> "BPETokenizer":
        """
        Loads a saved tokenizer from a directory or tokenizer.json file.
        """
        if os.path.isdir(filepath_or_dir):
            filepath = os.path.join(filepath_or_dir, "tokenizer.json")
        else:
            filepath = filepath_or_dir

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found at: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = Vocabulary(token_to_id=data["vocab"])
        merges = []
        for merge_str in data["merges"]:
            parts = merge_str.split(" ")
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

        special_tokens = data.get("special_tokens", SPECIAL_TOKENS)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
