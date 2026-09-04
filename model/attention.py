"""
Causal self-attention for URA-Shree.

Supports:
  * Multi-Head Attention (MHA) and Grouped-Query Attention (GQA)
  * Rotary Position Embeddings applied to Q and K
  * PyTorch fused scaled_dot_product_attention (FlashAttention / mem-efficient kernels)
  * Incremental KV caching for O(1)-per-token autoregressive decoding

The parameter names ``qkv_proj`` / ``out_proj`` are preserved so that
checkpoints trained with earlier revisions still load without remapping.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig

# Cache entry for one layer: (keys, values) each [B, kv_heads, T, head_dim]
KVCache = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Causal attention with optional grouped-query heads and rotary embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.effective_kv_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.dropout_p = config.dropout
        self.uses_rope = config.pos_encoding == "rope"

        # Number of query heads sharing each key/value head.
        self.group_size = self.num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Packed QKV projection. With MHA this is exactly [C, 3C], matching older
        # checkpoints byte for byte; with GQA the K/V slices shrink.
        self.qkv_proj = nn.Linear(self.embed_dim, q_dim + 2 * kv_dim, bias=config.bias)
        self.out_proj = nn.Linear(q_dim, self.embed_dim, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self._q_dim = q_dim
        self._kv_dim = kv_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # A materialised boolean mask is only needed for the slow path
        # (when attention weights are requested for inspection).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)).view(
                1, 1, self.max_seq_len, self.max_seq_len
            ),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        # Shrink the residual-stream contribution so deep stacks stay stable.
        std = 0.02 / math.sqrt(2 * self.config.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache]]:
        """
        Args:
            x: residual stream, [B, T, C]
            rope: (cos, sin) tables already sliced to the current absolute positions
            past_kv: cached (k, v) from previous decoding steps
            use_cache: return the concatenated (k, v) for reuse on the next step
            return_attention_weights: take the slow explicit-softmax path and
                return the [B, H, T, S] attention distribution

        Returns:
            (output [B, T, C], attention weights or None, new cache or None)
        """
        B, T, C = x.shape
        assert C == self.embed_dim, f"Input dim {C} does not match embed_dim {self.embed_dim}"

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self._q_dim, self._kv_dim, self._kv_dim], dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if rope is not None and self.uses_rope:
            cos, sin = rope
            q, k = self._apply_rope(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)

        new_cache: Optional[KVCache] = (k, v) if use_cache else None

        S = k.size(2)

        # Broadcast key/value heads across their query group (GQA).
        if self.group_size > 1:
            k_exp = k.repeat_interleave(self.group_size, dim=1)
            v_exp = v.repeat_interleave(self.group_size, dim=1)
        else:
            k_exp, v_exp = k, v

        if return_attention_weights:
            # Explicit path, used for interpretability and by the causal-masking tests.
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
            # Query i sits at absolute position (S - T + i) and may attend to keys <= that.
            mask = self.causal_mask[:, :, S - T : S, :S]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v_exp)
        else:
            # Fused kernel: fewer passes over HBM, no [B, H, T, S] score matrix
            # ever materialised. This is the whole ball game for memory use.
            attn_mask = None
            is_causal = False
            if T > 1:
                if S == T:
                    is_causal = True
                else:
                    # Prefill on top of an existing cache: queries start at offset S - T.
                    attn_mask = self.causal_mask[:, :, S - T : S, :S]
            attn_output = F.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_weights = None

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self._q_dim)
        out = self.resid_dropout(self.out_proj(attn_output))
        return out, attn_weights, new_cache
