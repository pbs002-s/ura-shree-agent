"""
Embeddings and positional encoding for URA-Shree.

Two position schemes are supported and selected by ``ModelConfig.pos_encoding``:

  * ``learned`` - a trainable absolute position table added to the token vectors.
  * ``rope``    - Rotary Position Embeddings applied inside attention to Q and K.
                  No position parameters are allocated at all in this mode, which
                  frees max_seq_len * embed_dim weights and generalises better
                  past the trained context length.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig


class TransformerEmbedding(nn.Module):
    """Maps token IDs to dense vectors, adding absolute positions when configured."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.uses_learned_positions = config.pos_encoding == "learned"

        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)

        if self.uses_learned_positions:
            self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)
            nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        else:
            self.position_embeddings = None

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] token IDs
            position_ids: optional explicit positions, [T] or [B, T]
            position_offset: absolute index of the first token, used when
                decoding incrementally against a KV cache

        Returns:
            [B, T, embed_dim]
        """
        B, T = input_ids.shape
        x = self.token_embeddings(input_ids)

        if self.uses_learned_positions:
            if position_ids is None:
                position_ids = torch.arange(
                    position_offset, position_offset + T, dtype=torch.long, device=input_ids.device
                )
            assert int(position_ids.max()) < self.max_seq_len, (
                f"Position {int(position_ids.max())} exceeds maximum {self.max_seq_len}"
            )
            x = x + self.position_embeddings(position_ids)

        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Rotates query and key vectors by an angle proportional to their absolute
    position, so that the attention inner product depends only on the relative
    distance:  <R_m q, R_n k> = g(q, k, m - n).

    The cos/sin tables are built once and grown on demand, so generating past the
    initial ``max_seq_len`` extends the table rather than failing.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_tables(max_seq_len, device=inv_freq.device, dtype=torch.float32)

    def _build_tables(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self._cached_len = seq_len

    def tables(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cos, sin) shaped [1, 1, seq_len, dim] for absolute positions
        [offset, offset + seq_len), ready to broadcast over batch and heads.
        """
        end = offset + seq_len
        device = device or self.inv_freq.device
        if end > self._cached_len or self.cos_cached.device != device or self.cos_cached.dtype != dtype:
            self._build_tables(max(end, self._cached_len), device=device, dtype=dtype)

        cos = self.cos_cached[offset:end].view(1, 1, seq_len, self.dim)
        sin = self.sin_cached[offset:end].view(1, 1, seq_len, self.dim)
        return cos, sin

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies rotary embeddings to q and k, each shaped [B, H, T, head_dim]."""
        cos, sin = self.tables(seq_len, offset=offset, device=q.device, dtype=q.dtype)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
