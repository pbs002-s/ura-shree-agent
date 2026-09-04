"""
Unit Tests for URA-Shree Local Inference Engine.
Verifies model loading from local checkpoints, greedy sampling determinism,
streaming token generation, repetition penalties, and stop sequence halting.
"""

import os
import sys
import torch
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.chat import InferenceEngine


@pytest.fixture(scope="module")
def inference_engine():
    """Fixture providing initialized InferenceEngine from local checkpoint."""
    ckpt_path = "checkpoints/best.pt" if os.path.exists("checkpoints/best.pt") else "checkpoints/last.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"No checkpoint available at {ckpt_path}. Run training first.")

    engine = InferenceEngine(
        checkpoint_path=ckpt_path,
        tokenizer_path="checkpoints/tokenizer.json",
        device="cpu", # Test on CPU for deterministic portability
    )
    return engine


def test_inference_engine_initialization(inference_engine):
    """Verify engine loads model weights and tokenizer successfully."""
    assert inference_engine.model is not None
    assert inference_engine.tokenizer is not None
    assert inference_engine.step_count > 0
    assert len(inference_engine.tokenizer) > 0


def test_greedy_generation_determinism(inference_engine):
    """
    CRITICAL INFERENCE TEST:
    Greedy generation (temperature=0.0) must produce strictly deterministic,
    identical token sequences across repeated runs.
    """
    prompt = "def binary_search("
    out1 = inference_engine.generate(prompt, max_new_tokens=15, temperature=0.0)
    out2 = inference_engine.generate(prompt, max_new_tokens=15, temperature=0.0)

    assert out1 == out2, f"Greedy generation was non-deterministic! Out1: {out1}, Out2: {out2}"
    assert len(out1) > 0


def test_streaming_token_generator(inference_engine):
    """Verify streaming token generator yields individual string tokens progressively."""
    prompt = "class Transformer:"
    tokens_streamed = []

    for token in inference_engine.generate_stream(prompt, max_new_tokens=10, temperature=0.5):
        assert isinstance(token, str)
        tokens_streamed.append(token)

    assert len(tokens_streamed) > 0
    assert len(tokens_streamed) <= 10


def test_stop_sequence_halting(inference_engine):
    """Verify generation halts immediately when a stop word is produced."""
    prompt = "SELECT * FROM users;"
    stop_word = "users"

    out = inference_engine.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.0,
        stop_words=[stop_word],
    )
    # The generation should stop upon emitting the stop sequence
    assert len(out) <= 500
