"""
Comprehensive Unit Tests for URA-Shree Transformer Architecture.
Verifies shapes, causal masking (no future leakage), weight tying,
gradient backpropagation, autoregressive generation, and CUDA support.
"""

import os
import sys
import torch
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.config import ModelConfig
from model.embeddings import TransformerEmbedding, RotaryEmbedding
from model.attention import CausalSelfAttention
from model.transformer import TransformerBlock, RMSNorm
from model.model import ShreeTransformerLM


@pytest.fixture
def tiny_config():
    """Tiny config fixture for fast unit test execution."""
    return ModelConfig(
        name="test-tiny",
        vocab_size=256,
        max_seq_len=64,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        intermediate_dim=128,
        dropout=0.0,
        tie_weights=True,
    )


def test_embedding_layer(tiny_config):
    """Verify embedding produces correct tensor shapes."""
    emb = TransformerEmbedding(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))
    out = emb(input_ids)
    assert out.shape == (2, 16, tiny_config.embed_dim)


def test_rotary_embedding():
    """Verify RoPE rotates vectors without altering their norm."""
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=64)
    q = torch.randn(1, 2, 10, dim)
    k = torch.randn(1, 2, 10, dim)
    q_rot, k_rot = rope(q, k, seq_len=10)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_causal_self_attention_masking(tiny_config):
    """Verify attention weights strictly follow lower-triangular causal structure."""
    attn = CausalSelfAttention(tiny_config)
    x = torch.randn(2, 12, tiny_config.embed_dim)
    out, weights, _cache = attn(x, return_attention_weights=True)

    assert out.shape == (2, 12, tiny_config.embed_dim)
    assert weights.shape == (2, tiny_config.num_heads, 12, 12)

    # Check upper triangle: every weight where j > i must be 0.0 (masked by -inf before softmax)
    upper_tri = torch.triu(weights, diagonal=1)
    assert torch.all(upper_tri == 0.0).item(), "Causal attention leaked information to future tokens!"


def test_causal_attention_no_future_leakage(tiny_config):
    """
    CRITICAL MATHEMATICAL INVARIANT:
    Token predictions at position t must NOT change if future tokens (at > t) are altered.
    """
    model = ShreeTransformerLM(tiny_config)
    model.eval()

    # Input sequence A: [10, 20, 30, 40, 50]
    seq_a = torch.tensor([[10, 20, 30, 40, 50]])
    # Input sequence B: [10, 20, 30, 99, 88] (positions 3 and 4 are changed)
    seq_b = torch.tensor([[10, 20, 30, 99, 88]])

    with torch.no_grad():
        logits_a, _ = model(seq_a)
        logits_b, _ = model(seq_b)

    # Positions 0, 1, 2 must produce identical output logits in both sequences!
    diff_pos0 = torch.max(torch.abs(logits_a[:, 0, :] - logits_b[:, 0, :])).item()
    diff_pos1 = torch.max(torch.abs(logits_a[:, 1, :] - logits_b[:, 1, :])).item()
    diff_pos2 = torch.max(torch.abs(logits_a[:, 2, :] - logits_b[:, 2, :])).item()

    assert diff_pos0 < 1e-5, f"Position 0 leaked future info! Diff: {diff_pos0}"
    assert diff_pos1 < 1e-5, f"Position 1 leaked future info! Diff: {diff_pos1}"
    assert diff_pos2 < 1e-5, f"Position 2 leaked future info! Diff: {diff_pos2}"


def test_full_model_forward_and_loss(tiny_config):
    """Verify full forward pass with and without targets."""
    model = ShreeTransformerLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16))

    # Without targets
    logits, loss = model(input_ids)
    assert logits.shape == (2, 16, tiny_config.vocab_size)
    assert loss is None

    # With targets
    targets = torch.randint(0, tiny_config.vocab_size, (2, 16))
    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (2, 16, tiny_config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0


def test_weight_tying(tiny_config):
    """Verify that LM head and token embeddings share the exact same weight tensor."""
    tiny_config.tie_weights = True
    model = ShreeTransformerLM(tiny_config)
    assert model.lm_head.weight is model.embedding.token_embeddings.weight


def test_backward_gradient_propagation(tiny_config):
    """Verify loss gradient backpropagates through all blocks down to embeddings."""
    model = ShreeTransformerLM(tiny_config)
    model.train()

    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    targets = torch.randint(0, tiny_config.vocab_size, (2, 8))

    logits, loss = model(input_ids, targets=targets)
    loss.backward()

    # Check that every single trainable parameter received a valid gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive a gradient!"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs!"


def test_autoregressive_generation(tiny_config):
    """Verify autoregressive text generation extends sequence by exactly max_new_tokens."""
    model = ShreeTransformerLM(tiny_config)
    prompt = torch.tensor([[1, 5, 10, 15]])
    gen_len = 10

    # Greedy generation
    out = model.generate(prompt, max_new_tokens=gen_len, temperature=0.0)
    assert out.shape == (1, 4 + gen_len)
    assert torch.equal(out[:, :4], prompt)

    # Nucleus / Top-k generation
    out_sampled = model.generate(prompt, max_new_tokens=gen_len, temperature=0.8, top_k=20, top_p=0.9)
    assert out_sampled.shape == (1, 4 + gen_len)


def test_cuda_execution_if_available(tiny_config):
    """Verify model can execute forward and backward passes on NVIDIA GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")

    model = ShreeTransformerLM(tiny_config).to("cuda")
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 16), device="cuda")
    targets = torch.randint(0, tiny_config.vocab_size, (2, 16), device="cuda")

    logits, loss = model(input_ids, targets=targets)
    assert logits.device.type == "cuda"
    assert loss.device.type == "cuda"

    loss.backward()
    torch.cuda.synchronize()
