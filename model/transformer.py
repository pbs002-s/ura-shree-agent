"""
Normalisation, feed-forward networks and the decoder block for URA-Shree.

Pre-normalisation keeps an uninterrupted identity path through the residual
stream, which is what lets deep stacks train without warmup collapse.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.attention import CausalSelfAttention, KVCache


class RMSNorm(nn.Module):
    """
    Root Mean Square layer normalisation:  y = x / sqrt(mean(x^2) + eps) * gamma

    No mean subtraction and no bias, so it costs roughly 10-15% less than
    LayerNorm while being just as stable in practice.

    The fused kernel is used where available. Written out by hand this is six
    separate kernels (pow, mean, add, rsqrt, mul, mul), and at 13 norms per
    forward pass that dominates the launch count of a small model far more than
    it dominates the arithmetic.
    """

    _HAS_FUSED = hasattr(F, "rms_norm")

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._HAS_FUSED:
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        # Accumulate the variance in fp32 even under autocast; the reciprocal
        # square root of a bf16 mean is where low-precision training goes wrong.
        dtype = x.dtype
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f.to(dtype)) * self.weight


class FeedForward(nn.Module):
    """
    Position-wise MLP.

    ``ffn="gelu"``   -> fc2(gelu(fc1(x)))                 (2 matrices)
    ``ffn="swiglu"`` -> fc2(silu(fc1(x)) * gate(x))       (3 matrices)

    SwiGLU is the better quality-per-FLOP choice, so when it is selected the
    hidden width is scaled by 2/3 to keep the parameter count comparable.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.variant = config.ffn
        self.num_layers = config.num_layers

        if self.variant == "swiglu":
            hidden = int(2 * config.intermediate_dim / 3)
            # Round to a multiple of 64 to keep GEMM tiles aligned.
            hidden = max(64, ((hidden + 63) // 64) * 64)
        else:
            hidden = config.intermediate_dim
        self.hidden_dim = hidden

        self.fc1 = nn.Linear(config.embed_dim, hidden, bias=config.bias)
        self.fc2 = nn.Linear(hidden, config.embed_dim, bias=config.bias)
        self.gate = nn.Linear(config.embed_dim, hidden, bias=config.bias) if self.variant == "swiglu" else None

        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        if self.gate is not None:
            nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        # Residual-stream scaling keeps the variance of the sum bounded in depth.
        std = 0.02 / math.sqrt(2 * self.num_layers)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std)
        for layer in (self.fc1, self.fc2, self.gate):
            if layer is not None and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate is not None:
            h = F.silu(self.fc1(x)) * self.gate(x)
        else:
            h = F.gelu(self.fc1(x), approximate="tanh")
        return self.dropout(self.fc2(h))


class TransformerBlock(nn.Module):
    """
    One pre-norm decoder block:

        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache]]:
        attn_out, weights, new_cache = self.attn(
            self.attn_norm(x),
            rope=rope,
            past_kv=past_kv,
            use_cache=use_cache,
            return_attention_weights=return_attention_weights,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, weights, new_cache
