"""
Unit Tests for URA-Shree Dataset Preprocessing, Causal Dataset, and DataLoader.
Verifies text cleaning, deduplication, causal (x, y) token shifting, and DataLoader batches.
"""

import os
import sys
import tempfile
import numpy as np
import torch
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocessing import clean_text, deduplicate_corpus, compute_content_hash, DataFilter
from data.dataset import CausalLMDataset, create_dataloader
from data.prepare_dataset import prepare_dataset
from tokenizer.tokenizer import BPETokenizer


def test_text_cleaning_preserves_indentation():
    """Verify that clean_text standardizes newlines but keeps code indentation intact."""
    raw_code = "def foo():\r\n    x = 10\r\n\treturn x\x00\x07\r\n\r\n\r\n\r\n"
    cleaned = clean_text(raw_code)

    assert "def foo():" in cleaned
    assert "    x = 10" in cleaned  # 4 spaces preserved
    assert "\treturn x" in cleaned   # Tab preserved
    assert "\x00" not in cleaned      # Null byte removed
    assert "\x07" not in cleaned      # Bell control char removed
    assert "\r" not in cleaned        # Carriage return removed
    assert "\n\n\n" not in cleaned    # Triple newlines collapsed


def test_deduplication():
    """Verify exact and near-duplicate removal."""
    corpus = [
        "def add(a, b): return a + b",
        "def sub(a, b): return a - b",
        "def add(a, b): return a + b", # Exact duplicate
        "def   add(a,   b):   return   a   +   b", # Near duplicate
    ]

    exact_dedup = deduplicate_corpus(corpus, normalize_whitespace=False)
    assert len(exact_dedup) == 3

    near_dedup = deduplicate_corpus(corpus, normalize_whitespace=True)
    assert len(near_dedup) == 2


def test_data_filter():
    """Verify quality heuristic filtering."""
    data_filter = DataFilter(min_chars=20, max_chars=1000, min_alpha_ratio=0.3, max_line_length=100)

    # Valid code
    valid, reason = data_filter.is_valid("def calculate():\n    return sum([1, 2, 3, 4, 5])")
    assert valid, f"Expected valid, got: {reason}"

    # Too short
    valid, reason = data_filter.is_valid("x = 1")
    assert not valid
    assert "Too short" in reason

    # Low alphanumeric ratio (e.g. noise)
    valid, reason = data_filter.is_valid(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    assert not valid
    assert "Alphanumeric ratio" in reason

    # Line too long
    long_line = "a" * 150
    valid, reason = data_filter.is_valid(long_line)
    assert not valid
    assert "Line exceeds" in reason


def test_causal_lm_dataset_shifting():
    """
    CRITICAL TEST: Verify causal next-token alignment.
    y[t] must be exactly x[t + 1] across the entire sequence.
    """
    # Create known sequence of 100 integers: [0, 1, 2, ..., 99]
    seq_len = 16
    tokens = np.arange(100, dtype=np.uint16)

    dataset = CausalLMDataset(tokens, seq_len=seq_len)
    assert len(dataset) == (100 - 1) // 16 # 6 full sequences

    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)

    # x = [0, 1, ..., 15]
    # y = [1, 2, ..., 16]
    assert torch.equal(x, torch.arange(0, 16))
    assert torch.equal(y, torch.arange(1, 17))
    # Next-token prediction invariant:
    assert torch.equal(x[1:], y[:-1])


def test_create_dataloader():
    """Verify DataLoader produces batched tensors of shape [B, T]."""
    tokens = np.arange(1000, dtype=np.uint16)
    seq_len = 32
    batch_size = 4

    dataset = CausalLMDataset(tokens, seq_len=seq_len)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (batch_size, seq_len)
    assert batch_y.shape == (batch_size, seq_len)
    assert batch_x.dtype == torch.int64
    assert batch_y.dtype == torch.int64


def test_binary_dataset_prepare_and_memmap():
    """Verify prepare_dataset generates valid binary files loadable via memory mapping."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path, val_path, n_train, n_val = prepare_dataset(
            tokenizer_path="checkpoints/tokenizer.json",
            output_dir=tmp_dir,
            val_ratio=0.2,
            min_copies=10,
        )

        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert n_train > 0
        assert n_val > 0

        # Verify CausalLMDataset can load the memmap file
        seq_len = 64
        train_ds = CausalLMDataset(train_path, seq_len=seq_len)
        assert len(train_ds) > 0

        sample_x, sample_y = train_ds[0]
        assert sample_x.shape == (seq_len,)
        assert sample_y.shape == (seq_len,)
        assert torch.equal(sample_x[1:], sample_y[:-1])

        # Release Windows memory-map file handle before tempdir cleanup
        train_ds.close()
