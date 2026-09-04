"""
Vocabulary Management & Byte Encoding for URA-Shree BPE Tokenizer.
Provides bi-directional token-ID mappings, special token definitions,
and byte-level mappings ensuring 100% lossless UTF-8 representation.
"""

from typing import Dict, List, Optional, Tuple, Set
import json
import os

# Special control tokens for language model and autonomous coding agent
SPECIAL_TOKENS: List[str] = [
    "<|pad|>",         # Padding token (ID: 0)
    "<|bos|>",         # Beginning of sequence (ID: 1)
    "<|eos|>",         # End of sequence (ID: 2)
    "<|unk|>",         # Fallback unknown token (ID: 3)
    "<|user|>",        # User prompt turn marker
    "<|assistant|>",   # Assistant response turn marker
    "<|system|>",      # System prompt marker
    "<|tool_call|>",   # Beginning of tool invocation call
    "<|tool_result|>", # Output of executed tool
]


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns a dictionary mapping every 8-bit byte (0..255) to a unique printable Unicode character.
    This prevents control characters and whitespace from causing parsing/serialization issues,
    following the byte-level BPE design (similar to GPT-2/GPT-4).
    """
    # Printable ASCII and Latin-1 supplement ranges:
    # 33 ('!') to 126 ('~'), 161 ('¡') to 172 ('¬'), 174 ('®') to 255 ('ÿ')
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # Map remaining non-printable bytes to Unicode characters starting at 256
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class Vocabulary:
    """
    Bi-directional mapping between token strings and integer IDs.
    Supports serialization to and from JSON format.
    """
    def __init__(self, token_to_id: Optional[Dict[str, int]] = None):
        if token_to_id is not None:
            self.token_to_id: Dict[str, int] = dict(token_to_id)
            self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}
        else:
            self.token_to_id = {}
            self.id_to_token = {}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary if not already present. Returns its ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def get_id(self, token: str, default: Optional[int] = None) -> int:
        """Get the integer ID for a token."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        if default is not None:
            return default
        raise KeyError(f"Token '{token}' not found in vocabulary.")

    def get_token(self, token_id: int) -> str:
        """Get the string token for an integer ID."""
        if token_id in self.id_to_token:
            return self.id_to_token[token_id]
        raise KeyError(f"Token ID '{token_id}' not found in vocabulary.")

    def save(self, filepath: str) -> None:
        """Serialize vocabulary to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"token_to_id": self.token_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(token_to_id=data["token_to_id"])
