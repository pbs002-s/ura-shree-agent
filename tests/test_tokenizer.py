"""
Comprehensive Unit Tests for URA-Shree Byte-Level BPE Tokenizer.
Verifies lossless encoding/decoding, special token preservation,
training loop mechanics, and JSON serialization.
"""

import os
import sys
import tempfile
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.vocabulary import Vocabulary, SPECIAL_TOKENS, bytes_to_unicode
from tokenizer.tokenizer import BPETokenizer
from tokenizer.train_tokenizer import train_bpe, SAMPLE_TRAINING_CORPUS


def test_vocabulary_basic():
    """Verify vocabulary bi-directional mapping and containment."""
    vocab = Vocabulary()
    idx1 = vocab.add_token("<|pad|>")
    idx2 = vocab.add_token("hello")

    assert idx1 == 0
    assert idx2 == 1
    assert "<|pad|>" in vocab
    assert "hello" in vocab
    assert "missing" not in vocab

    assert vocab.get_id("hello") == 1
    assert vocab.get_token(1) == "hello"

    with pytest.raises(KeyError):
        vocab.get_id("unknown_token")

    with pytest.raises(KeyError):
        vocab.get_token(9999)


def test_bytes_to_unicode_mapping():
    """Verify all 256 bytes have unique single-character mappings."""
    b2u = bytes_to_unicode()
    assert len(b2u) == 256
    assert len(set(b2u.values())) == 256

    for b in range(256):
        assert b in b2u
        assert isinstance(b2u[b], str)
        assert len(b2u[b]) == 1


def test_base_byte_roundtrip_lossless():
    """Verify base tokenizer (0 merges) provides 100% lossless UTF-8 roundtrip."""
    tokenizer = BPETokenizer()
    test_cases = [
        "Hello, World!",
        "def add(a: int, b: int) -> int:\n    return a + b",
        "Emoji test: 🚀 🤖 ⚡ ✨",
        "Special punctuation: !@#$%^&*()_+~`-={}|[]\\:\";'<>?,./",
        "Indentation test:\n\t\t\tFour spaces:    End.",
        "Non-ASCII: Café, España, 東京, 서울, München",
    ]

    for text in test_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text, f"Failed roundtrip for: {text}"


def test_special_tokens_preservation():
    """Verify special control tokens are recognized as single IDs and not split."""
    tokenizer = BPETokenizer()

    text = "<|user|>\nWrite a Python function.\n<|assistant|>\n<|tool_call|>\nfilesystem.list\n<|tool_result|>"
    ids = tokenizer.encode(text)

    # Check that each special token maps to its exact known ID
    assert tokenizer.vocab.get_id("<|user|>") in ids
    assert tokenizer.vocab.get_id("<|assistant|>") in ids
    assert tokenizer.vocab.get_id("<|tool_call|>") in ids
    assert tokenizer.vocab.get_id("<|tool_result|>") in ids

    # Roundtrip with special tokens included
    decoded_with_special = tokenizer.decode(ids, skip_special_tokens=False)
    assert decoded_with_special == text

    # Decoding with skip_special_tokens=True strips them
    decoded_stripped = tokenizer.decode(ids, skip_special_tokens=True)
    assert "<|user|>" not in decoded_stripped
    assert "<|tool_call|>" not in decoded_stripped
    assert "Write a Python function." in decoded_stripped


def test_bpe_training_and_compression():
    """Verify training learns merges and reduces sequence length (compression)."""
    # Base tokenizer length on corpus
    base_tokenizer = BPETokenizer()
    sample_text = "def binary_search(arr, target):\n    return target"
    base_token_count = len(base_tokenizer.encode(sample_text))

    # Train tokenizer with target vocab size 400 (base ~265 + merges)
    trained_tokenizer = train_bpe(
        SAMPLE_TRAINING_CORPUS,
        target_vocab_size=350,
        verbose=False
    )
    assert len(trained_tokenizer) > 265
    assert len(trained_tokenizer.merges) > 0

    trained_token_count = len(trained_tokenizer.encode(sample_text))
    # Trained tokenizer should compress repetitive code tokens into fewer IDs
    assert trained_token_count < base_token_count

    # Verify lossless decode on trained tokenizer
    decoded = trained_tokenizer.decode(trained_tokenizer.encode(sample_text))
    assert decoded == sample_text


def test_tokenizer_save_and_load():
    """Verify saving to JSON and loading back reproduces identical state."""
    trained_tokenizer = train_bpe(
        SAMPLE_TRAINING_CORPUS[:3],
        target_vocab_size=300,
        verbose=False
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = trained_tokenizer.save(tmp_dir)
        assert os.path.exists(save_path)

        loaded_tokenizer = BPETokenizer.load(save_path)
        assert len(loaded_tokenizer) == len(trained_tokenizer)
        assert len(loaded_tokenizer.merges) == len(trained_tokenizer.merges)

        test_code = "import torch\nimport torch.nn as nn\nprint('Testing URA-Shree!')"
        orig_ids = trained_tokenizer.encode(test_code)
        loaded_ids = loaded_tokenizer.encode(test_code)

        assert orig_ids == loaded_ids
        assert loaded_tokenizer.decode(loaded_ids) == test_code
