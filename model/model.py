"""
The URA-Shree decoder-only causal language model.

    token IDs -> embedding -> [decoder block] x N -> RMSNorm -> LM head -> logits

Beyond the plain forward pass this module provides `step()`, an incremental
decoding entry point backed by a per-layer key/value cache. Without it every
generated token costs a full quadratic re-read of the prompt; with it each token
costs one row of attention, which is the difference between a toy and something
usable interactively.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig
from model.embeddings import TransformerEmbedding, RotaryEmbedding
from model.transformer import TransformerBlock, RMSNorm
from model.attention import KVCache

PastKeyValues = List[KVCache]


class ShreeTransformerLM(nn.Module):
    """Decoder-only transformer with weight tying, RoPE/GQA support and KV caching."""

    def __init__(self, config: ModelConfig, verbose: bool = True):
        super().__init__()
        self.config = config

        self.embedding = TransformerEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Tying the output projection to the input embedding removes
        # vocab_size * embed_dim parameters and ties the two representation
        # spaces together, which measurably helps small models.
        if config.tie_weights:
            self.lm_head.weight = self.embedding.token_embeddings.weight

        self.rotary: Optional[RotaryEmbedding] = None
        if config.pos_encoding == "rope":
            self.rotary = RotaryEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                base=config.rope_theta,
            )

        if verbose:
            self._report()

    def _report(self) -> None:
        print(f"[URA-Shree Model] Initialized: {self.config.name}")
        print(f"  - Total Parameters        : {self.get_num_params():,}")
        print(f"  - Non-Embedding Parameters: {self.get_num_params(non_embedding=True):,}")
        print(f"  - Context Window (tokens) : {self.config.max_seq_len}")
        print(f"  - Hidden Dim (d_model)    : {self.config.embed_dim}")
        print(f"  - Attention Heads / Layers: {self.config.num_heads} / {self.config.num_layers}")
        print(f"  - Position Encoding       : {self.config.pos_encoding}")
        print(f"  - Feed-Forward            : {self.config.ffn}")
        if self.config.effective_kv_heads != self.config.num_heads:
            print(f"  - Grouped-Query KV Heads  : {self.config.effective_kv_heads}")

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Parameter count, optionally excluding the embedding tables.

        Counts every parameter rather than only the trainable ones: at inference
        the whole model has requires_grad cleared, and a model that reports zero
        parameters because it is in eval mode is just wrong.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.embedding.position_embeddings is not None:
                n_params -= self.embedding.position_embeddings.weight.numel()
            if not self.config.tie_weights:
                n_params -= self.embedding.token_embeddings.weight.numel()
        return n_params

    def memory_footprint(self) -> Dict[str, float]:
        """Resident weight and buffer size in MB."""
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        return {
            "parameters_mb": round(param_bytes / (1024 * 1024), 2),
            "buffers_mb": round(buffer_bytes / (1024 * 1024), 2),
            "total_mb": round((param_bytes + buffer_bytes) / (1024 * 1024), 2),
        }

    def kv_cache_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        """Exact bytes a full KV cache occupies, useful for sizing the context window."""
        cfg = self.config
        per_token = 2 * cfg.num_layers * cfg.effective_kv_heads * cfg.head_dim
        dtype_size = next(self.parameters()).element_size()
        return per_token * seq_len * batch_size * dtype_size

    def _rope_tables(
        self, seq_len: int, offset: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.rotary is None:
            return None
        return self.rotary.tables(seq_len, offset=offset, device=device, dtype=dtype)

    def _body(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[PastKeyValues],
        use_cache: bool,
        return_attention_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[PastKeyValues]]:
        B, T = input_ids.shape
        offset = past_key_values[0][0].size(2) if past_key_values else 0
        assert offset + T <= self.config.max_seq_len, (
            f"Sequence length {offset + T} exceeds model max sequence length {self.config.max_seq_len}"
        )

        x = self.embedding(input_ids, position_offset=offset)
        rope = self._rope_tables(T, offset, x.device, x.dtype)

        new_caches: Optional[PastKeyValues] = [] if use_cache else None
        last_weights: Optional[torch.Tensor] = None

        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values else None
            x, weights, cache = block(
                x,
                rope=rope,
                past_kv=past,
                use_cache=use_cache,
                return_attention_weights=return_attention_weights,
            )
            if weights is not None:
                last_weights = weights
            if new_caches is not None:
                new_caches.append(cache)

        return self.norm(x), last_weights, new_caches

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B, T] token IDs
            targets: optional [B, T] next-token labels; -1 entries are ignored
            return_attention_weights: use the explicit-softmax attention path

        Returns:
            (logits [B, T, vocab_size], cross-entropy loss or None)
        """
        x, _, _ = self._body(
            input_ids,
            past_key_values=None,
            use_cache=False,
            return_attention_weights=return_attention_weights,
        )
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[PastKeyValues] = None,
    ) -> Tuple[torch.Tensor, PastKeyValues]:
        """
        One incremental decoding step.

        Args:
            input_ids: [B, T] - the full prompt on the first call, then [B, 1]
            past_key_values: cache returned by the previous call

        Returns:
            (logits for the final position only [B, vocab_size], updated cache)
        """
        x, _, caches = self._body(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_attention_weights=False,
        )
        # Project only the last position. Running the vocab GEMM over the whole
        # prompt would cost T times more for output we immediately discard.
        logits = self.lm_head(x[:, -1, :])
        return logits, caches

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, sequence: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Vectorised repetition penalty over the whole batch, no Python loop."""
        if penalty == 1.0:
            return logits
        gathered = torch.gather(logits, 1, sequence)
        gathered = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
        return logits.scatter(1, sequence, gathered)

    @staticmethod
    def _filter_logits(
        logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]
    ) -> torch.Tensor:
        """Applies top-k then nucleus (top-p) filtering to a [B, V] logit tensor."""
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_sorted = cumulative > top_p
            # Shift right so the token that crosses the threshold is still kept.
            remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
            remove_sorted[..., 0] = False
            # Scatter the sorted mask back onto the original vocabulary order.
            remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
            logits = logits.masked_fill(remove, float("-inf"))

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive sampling. Returns [B, prompt_len + generated].

        With `use_cache` (the default) the prompt is encoded once and every
        subsequent token is a single-position forward pass.
        """
        self.eval()
        max_len = self.config.max_seq_len

        context = input_ids if input_ids.size(1) <= max_len else input_ids[:, -max_len:]
        cache: Optional[PastKeyValues] = None
        if use_cache:
            logits, cache = self.step(context)
        else:
            logits = self(context)[0][:, -1, :]

        for _ in range(max_new_tokens):
            next_logits = self._apply_repetition_penalty(
                logits.float().clone(), input_ids, repetition_penalty
            )

            if temperature > 0.0:
                next_logits = self._filter_logits(next_logits / temperature, top_k, top_p)
                next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_id is not None and (next_token == eos_id).all():
                break

            if cache is not None and cache[0][0].size(2) < max_len:
                logits, cache = self.step(next_token, past_key_values=cache)
            else:
                # Context window is full: drop the cache and recompute a window.
                window = input_ids[:, -max_len:]
                if use_cache:
                    logits, cache = self.step(window)
                else:
                    logits = self(window)[0][:, -1, :]

        return input_ids
