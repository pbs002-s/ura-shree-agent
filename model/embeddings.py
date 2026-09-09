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
    Rotary Position Embedding (RoPE), with optional context-extension scaling.

    Rotates query and key vectors by an angle proportional to their absolute
    position, so that the attention inner product depends only on the relative
    distance:  <R_m q, R_n k> = g(q, k, m - n).

    The cos/sin tables are built once and grown on demand, so generating past the
    initial ``max_seq_len`` extends the table rather than failing.

    ``scaling`` selects how positions beyond ``max_seq_len`` (the length the
    model was actually trained at) are handled once the table needs to grow
    past it:

      * ``"none"``        - extend with the original frequencies unchanged.
                             This is the historical behaviour, bit-exact for
                             checkpoints saved before this option existed.
      * ``"dynamic_ntk"``  - rescale the RoPE base by the extension ratio, per
                             "NTK-Aware Scaled RoPE" - cheap and effective for
                             modest extensions.
      * ``"yarn"``         - blend interpolated and extrapolated frequencies
                             per-dimension (YaRN), plus an attention
                             temperature correction, for larger extensions.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling: str = "none",
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling = scaling
        # The length the model was trained at; scaling only kicks in past this.
        self.original_max_seq_len = max_seq_len
        self.scaling_factor = max(1.0, float(scaling_factor))
        self.mscale = 1.0

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        target_len = max(max_seq_len, int(max_seq_len * self.scaling_factor))
        self._build_tables(target_len, device=inv_freq.device, dtype=torch.float32)

    # -- YaRN helpers ---------------------------------------------------------

    def _yarn_find_dim_range(self, num_rotations_low: float, num_rotations_high: float) -> Tuple[float, float]:
        """Dimension indices whose wavelength falls inside [low, high] rotations."""

        def find_dim(num_rotations: float) -> float:
            return (self.dim * math.log(self.original_max_seq_len / (num_rotations * 2 * math.pi))) / (
                2 * math.log(self.base)
            )

        low = math.floor(find_dim(num_rotations_high))
        high = math.ceil(find_dim(num_rotations_low))
        return max(low, 0), min(high, self.dim // 2 - 1)

    def _yarn_inv_freq(self, scaling_factor: float, beta_fast: float = 32.0, beta_slow: float = 1.0) -> torch.Tensor:
        """Per-dimension blend of interpolated and extrapolated frequencies."""
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = self._yarn_find_dim_range(beta_fast, beta_slow)
        # Linear ramp from 0 (fully interpolated) to 1 (fully extrapolated)
        # across the [low, high] dimension band; clamp to stay in [0, 1].
        span = max(high - low, 1e-3)
        idx = torch.arange(self.dim // 2, dtype=torch.float32)
        ramp = torch.clamp((idx - low) / span, 0.0, 1.0)
        mask = 1.0 - ramp
        return inv_freq_interpolation * mask + inv_freq_extrapolation * (1.0 - mask)

    def _resolve_inv_freq(self, seq_len: int) -> Tuple[torch.Tensor, float]:
        """Returns (inv_freq, mscale) for a table covering ``seq_len`` positions."""
        if self.scaling == "none" or seq_len <= self.original_max_seq_len:
            return self.inv_freq, 1.0

        ratio = seq_len / self.original_max_seq_len

        if self.scaling == "dynamic_ntk":
            # NTK-aware base rescaling: stretches wavelengths just enough to
            # cover the new length without touching the highest frequencies.
            adjusted_base = self.base * ((ratio * self.dim / (self.dim - 2)) - (ratio - 1)) ** (
                self.dim / (self.dim - 2)
            )
            inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            return inv_freq, 1.0

        # "yarn"
        inv_freq = self._yarn_inv_freq(ratio)
        # Attention temperature correction so logit magnitudes stay in range
        # after interpolation (YaRN eq. for the "sqrt" mscale variant).
        mscale = 0.1 * math.log(ratio) + 1.0 if ratio > 1.0 else 1.0
        return inv_freq, mscale

    def _build_tables(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        inv_freq, mscale = self._resolve_inv_freq(seq_len)
        inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * mscale).to(dtype), persistent=False)
        self._cached_len = seq_len
        self.mscale = mscale

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
