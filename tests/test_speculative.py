"""
Unit tests for speculative decoding (inference/speculative.py).

Verifies exact output equivalence with greedy decoding, that the
rejection/resampling path runs and produces valid shapes when target and
draft actually disagree, and that acceptance-rate bookkeeping is sane.
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import ModelConfig
from model.model import ShreeTransformerLM
from inference.speculative import speculative_generate, SpeculativeStats


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        name="test-tiny",
        vocab_size=128,
        max_seq_len=64,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        intermediate_dim=64,
        dropout=0.0,
    )
    model = ShreeTransformerLM(cfg, verbose=False)
    model.eval()
    return model


def test_speculative_matches_greedy_decode_exactly(tiny_model):
    """
    CRITICAL EQUIVALENCE TEST:
    With the target model used as its own draft, speculative decoding must
    reproduce the target's plain greedy generation byte-for-byte - speculation
    changes only how many forward passes it costs, never the output.
    """
    torch.manual_seed(0)
    prompt = torch.randint(0, 128, (1, 6))

    baseline = tiny_model.generate(
        prompt.clone(), max_new_tokens=16, temperature=0.0, repetition_penalty=1.0, use_cache=True
    )
    spec_out, stats = speculative_generate(
        tiny_model, tiny_model, prompt.clone(), max_new_tokens=16, num_speculative_tokens=4, temperature=0.0
    )

    assert torch.equal(baseline, spec_out)
    assert stats.rounds > 0
    # Draft == target, so every proposal is accepted in the greedy special case.
    assert stats.acceptance_rate == 1.0


def test_speculative_respects_max_new_tokens_exactly(tiny_model):
    """The bonus/correction token must never push output past the requested budget."""
    prompt = torch.randint(0, 128, (1, 5))
    for n in (1, 2, 3, 7):
        out, _stats = speculative_generate(
            tiny_model, tiny_model, prompt.clone(), max_new_tokens=n, num_speculative_tokens=4, temperature=0.0
        )
        assert out.shape == (1, 5 + n)


def test_speculative_rejection_path_with_mismatched_draft():
    """A weaker, differently-shaped draft model exercises the reject + resample branch."""
    torch.manual_seed(1)
    target_cfg = ModelConfig(
        vocab_size=128, max_seq_len=64, embed_dim=48, num_layers=3, num_heads=4, intermediate_dim=96, dropout=0.0
    )
    draft_cfg = ModelConfig(
        vocab_size=128, max_seq_len=64, embed_dim=16, num_layers=1, num_heads=2, intermediate_dim=32, dropout=0.0
    )
    target = ShreeTransformerLM(target_cfg, verbose=False).eval()
    draft = ShreeTransformerLM(draft_cfg, verbose=False).eval()

    prompt = torch.randint(0, 128, (1, 6))
    out, stats = speculative_generate(
        target, draft, prompt, max_new_tokens=32, num_speculative_tokens=4, temperature=0.8
    )

    assert out.shape == (1, 6 + 32)
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert stats.proposed > 0
    assert stats.rounds > 0


def test_speculative_stops_at_eos():
    """Generation must stop as soon as the eos token appears, including inside a draft block."""
    torch.manual_seed(2)
    cfg = ModelConfig(vocab_size=32, max_seq_len=64, embed_dim=16, num_layers=1, num_heads=2, intermediate_dim=32, dropout=0.0)
    model = ShreeTransformerLM(cfg, verbose=False).eval()

    prompt = torch.randint(0, 32, (1, 4))
    out, _stats = speculative_generate(
        model, model, prompt, max_new_tokens=50, num_speculative_tokens=4, temperature=0.0, eos_id=5
    )
    generated = out[0, 4:].tolist()
    if 5 in generated:
        assert generated.index(5) == len(generated) - 1


def test_speculative_stats_to_dict():
    stats = SpeculativeStats(proposed=10, accepted=7, rounds=3)
    d = stats.to_dict()
    assert d["proposed"] == 10
    assert d["accepted"] == 7
    assert d["acceptance_rate"] == 0.7
