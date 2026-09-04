"""
CUDA-graph accelerated decoding.

A profile of the eager decode loop on a small model tells an unambiguous story:
roughly 180 kernel launches per token, about 0.5 ms of actual GPU work, and
about 10 ms of wall time. The arithmetic is not the problem - the CPU cost of
telling the GPU what to do is. A bigger model would hide that overhead behind
real work; a 14M-parameter one cannot.

The fix has two halves, and both are needed:

  * A *static* KV cache. The eager cache grows by one column per step, so every
    step has new tensor shapes, which makes graph capture impossible and defeats
    torch.compile. Here the cache is allocated once at full context length and
    new keys and values are written in place at a position held in a device
    tensor. Shapes never change.

  * A captured CUDA graph. With static shapes and static buffers, one decode
    step is recorded once and then replayed, which submits the whole step to the
    GPU as a single operation instead of ~180 individual launches.

Attention then runs over the whole allocated buffer with a mask derived from the
position tensor. That is more arithmetic than strictly necessary, and it is
worth it: the GPU work was never the bottleneck.

Everything here is optional. `FastDecoder.available()` says whether this path
can be used, and the engine falls back to the eager loop when it cannot.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from model.model import ShreeTransformerLM


class StaticKVCache:
    """
    Fixed-size key/value buffers, one pair per layer.

    `position` lives on the device so it can be updated without re-capturing the
    graph, and `write_index` is what the attention kernels index with.
    """

    def __init__(self, model: ShreeTransformerLM, batch_size: int, device: torch.device, dtype: torch.dtype):
        config = model.config
        self.layers = config.num_layers
        self.max_len = config.max_seq_len
        self.device = device

        shape = (batch_size, config.effective_kv_heads, self.max_len, config.head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.layers)]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(self.layers)]

        # Written into by the host between replays; read by the captured graph.
        self.write_index = torch.zeros(1, dtype=torch.long, device=device)
        self.positions = torch.arange(self.max_len, device=device)
        self.length = 0

    def reset(self) -> None:
        for k, v in zip(self.keys, self.values):
            k.zero_()
            v.zero_()
        self.length = 0
        self.write_index.fill_(0)

    def seed(self, past: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Copies an eager prefill cache into the static buffers."""
        prefix = past[0][0].size(2)
        if prefix > self.max_len:
            raise ValueError(f"Prefill of {prefix} exceeds the {self.max_len}-token buffer")
        for layer, (k, v) in enumerate(past):
            self.keys[layer][:, :, :prefix].copy_(k)
            self.values[layer][:, :, :prefix].copy_(v)
        self.length = prefix

    def set_position(self, index: int) -> None:
        self.write_index.fill_(index)

    def bytes(self) -> int:
        per = self.keys[0].numel() * self.keys[0].element_size()
        return per * 2 * self.layers


@torch.inference_mode()
def decode_step(
    model: ShreeTransformerLM,
    token: torch.Tensor,
    cache: StaticKVCache,
) -> torch.Tensor:
    """
    One token through the network against the static cache.

    Written out rather than reusing `model._body` because every operation here
    has to be capture-safe: no Python branching on tensor values, no allocation
    that changes between steps, and no host synchronisation.
    """
    config = model.config
    index = cache.write_index

    x = model.embedding.token_embeddings(token)
    if model.embedding.position_embeddings is not None:
        x = x + model.embedding.position_embeddings(index).unsqueeze(0)

    rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if model.rotary is not None:
        cos_table, sin_table = model.rotary.tables(
            cache.max_len, offset=0, device=x.device, dtype=x.dtype
        )
        # index_select keeps the position on the device; plain slicing would
        # bake the step number into the captured graph.
        cos = cos_table.view(cache.max_len, -1).index_select(0, index).view(1, 1, 1, -1)
        sin = sin_table.view(cache.max_len, -1).index_select(0, index).view(1, 1, 1, -1)
        rope = (cos, sin)

    # Positions at or before the write index are visible; the rest of the
    # buffer is masked out. Additive rather than boolean so it fuses into SDPA.
    visible = (cache.positions.unsqueeze(0) <= index.unsqueeze(-1)).view(1, 1, 1, cache.max_len)
    attn_mask = torch.zeros_like(visible, dtype=x.dtype).masked_fill_(~visible, float("-inf"))

    for layer_index, block in enumerate(model.blocks):
        attention = block.attn
        normed = block.attn_norm(x)

        qkv = attention.qkv_proj(normed)
        q, k, v = qkv.split([attention._q_dim, attention._kv_dim, attention._kv_dim], dim=-1)

        q = q.view(1, 1, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = k.view(1, 1, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
        v = v.view(1, 1, attention.num_kv_heads, attention.head_dim).transpose(1, 2)

        if rope is not None and attention.uses_rope:
            q, k = attention._apply_rope(q, k, rope[0], rope[1])

        key_buffer = cache.keys[layer_index]
        value_buffer = cache.values[layer_index]
        key_buffer.index_copy_(2, index, k)
        value_buffer.index_copy_(2, index, v)

        if attention.group_size > 1:
            k_all = key_buffer.repeat_interleave(attention.group_size, dim=1)
            v_all = value_buffer.repeat_interleave(attention.group_size, dim=1)
        else:
            k_all, v_all = key_buffer, value_buffer

        context = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask)
        context = context.transpose(1, 2).reshape(1, 1, attention._q_dim)

        x = x + attention.out_proj(context)
        x = x + block.ffn(block.ffn_norm(x))

    return model.lm_head(model.norm(x))[:, -1, :]


class FastDecoder:
    """
    Captures `decode_step` as a CUDA graph and replays it once per token.

    Usage is prefill once with the eager path, seed the static cache, then call
    `step()` per token.
    """

    def __init__(self, model: ShreeTransformerLM, device: torch.device, dtype: torch.dtype):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.cache = StaticKVCache(model, batch_size=1, device=device, dtype=dtype)

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.static_logits: Optional[torch.Tensor] = None
        self.capture_error = ""

    @staticmethod
    def available(device: torch.device) -> bool:
        return device.type == "cuda" and hasattr(torch.cuda, "CUDAGraph")

    def reset(self, prefill_cache: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.cache.reset()
        self.cache.seed(prefill_cache)

    def _capture(self) -> bool:
        """
        Records one decode step. Returns False if capture fails, in which case
        the caller keeps using the eager path.
        """
        try:
            # Warm up on a side stream first. Capturing a cold step records
            # cuBLAS handle creation and autotuning into the graph.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    decode_step(self.model, self.static_token, self.cache)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.static_logits = decode_step(self.model, self.static_token, self.cache)
            self.graph = graph
            return True
        except Exception as err:
            self.capture_error = str(err)
            self.graph = None
            return False

    def step(self, token_id: int) -> torch.Tensor:
        """Advances one token and returns the logits for the next position."""
        position = self.cache.length
        if position >= self.cache.max_len:
            raise IndexError("The static cache is full; the context window is exhausted.")

        self.static_token.fill_(token_id)
        self.cache.set_position(position)

        if self.graph is None and not self._capture():
            logits = decode_step(self.model, self.static_token, self.cache)
        else:
            self.graph.replay()
            logits = self.static_logits

        self.cache.length = position + 1
        # The graph writes into the same buffer every replay, so the caller
        # must get a copy or the value changes under it on the next token.
        return logits.clone()

    def info(self) -> Dict[str, object]:
        return {
            "captured": self.graph is not None,
            "error": self.capture_error,
            "cache_mb": round(self.cache.bytes() / (1024 * 1024), 2),
            "max_len": self.cache.max_len,
        }
