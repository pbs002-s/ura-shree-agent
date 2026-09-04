"""
The CUDA-graph decoder must be a pure speed optimisation: identical tokens out,
or it is not worth having. These tests skip on machines without CUDA.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.fast_decode import FastDecoder, StaticKVCache, decode_step
from model.config import ModelConfig
from model.model import ShreeTransformerLM

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def build(**overrides) -> ShreeTransformerLM:
    config = ModelConfig(
        vocab_size=256, max_seq_len=64, embed_dim=64, num_layers=2,
        num_heads=4, intermediate_dim=128, dropout=0.0, **overrides,
    )
    torch.manual_seed(7)
    return ShreeTransformerLM(config, verbose=False).eval()


@pytest.mark.parametrize("overrides", [
    {},
    {"pos_encoding": "rope"},
    {"pos_encoding": "rope", "num_kv_heads": 2, "ffn": "swiglu"},
])
@cuda_only
def test_graph_decode_matches_the_eager_path(overrides):
    device = torch.device("cuda")
    # float32 keeps the comparison about the decoder rather than about precision.
    model = build(**overrides).to(device=device, dtype=torch.float32)
    prompt = torch.randint(0, 256, (1, 16), device=device)

    with torch.inference_mode():
        logits, cache = model.step(prompt)
        eager, current, state = [], logits, cache
        for _ in range(12):
            token = int(torch.argmax(current, dim=-1).item())
            eager.append(token)
            current, state = model.step(torch.tensor([[token]], device=device), past_key_values=state)

        logits, prefill = model.step(prompt)
        decoder = FastDecoder(model, device, torch.float32)
        decoder.reset(prefill)
        fast, current = [], logits
        for _ in range(12):
            token = int(torch.argmax(current, dim=-1).item())
            fast.append(token)
            current = decoder.step(token)

    assert fast == eager, f"{overrides}: graph decode diverged"
    assert decoder.info()["captured"], decoder.info()["error"]


@cuda_only
def test_replayed_logits_are_copied_not_aliased():
    """Each step must return its own tensor, or the previous value mutates."""
    device = torch.device("cuda")
    model = build().to(device=device, dtype=torch.float32)
    prompt = torch.randint(0, 256, (1, 8), device=device)

    with torch.inference_mode():
        _, prefill = model.step(prompt)
        decoder = FastDecoder(model, device, torch.float32)
        decoder.reset(prefill)
        first = decoder.step(3)
        snapshot = first.clone()
        decoder.step(9)

    assert torch.equal(first, snapshot), "the graph overwrote logits the caller still held"


@cuda_only
def test_static_cache_is_sized_from_the_config():
    device = torch.device("cuda")
    model = build(num_kv_heads=2).to(device=device, dtype=torch.float32)
    cache = StaticKVCache(model, batch_size=1, device=device, dtype=torch.float32)

    assert len(cache.keys) == model.config.num_layers
    assert cache.keys[0].shape == (1, 2, 64, model.config.head_dim)
    # 2 layers x (k + v) x 1 x 2 heads x 64 positions x 16 dims x 4 bytes
    assert cache.bytes() == 2 * 2 * 2 * 64 * model.config.head_dim * 4


def test_availability_is_false_without_cuda():
    assert FastDecoder.available(torch.device("cpu")) is False


@cuda_only
def test_engine_streams_with_the_graph_decoder():
    """A full generation through the engine must run and report the graph state."""
    import os

    if not (os.path.exists("checkpoints/coding_best.pt") and os.path.exists("checkpoints/tokenizer.json")):
        pytest.skip("no local checkpoint available")

    from inference.engine import InferenceEngine

    engine = InferenceEngine(
        checkpoint_path="checkpoints/coding_best.pt",
        tokenizer_path="checkpoints/tokenizer.json",
    )
    assert engine.use_graph_decode

    text = engine.generate("<|user|>\nhello\n<|assistant|>\n", max_new_tokens=24, temperature=0.0)
    assert isinstance(text, str)
    assert engine.last_stats.completion_tokens > 0
    assert engine.describe()["graph_decode"]["captured"]
