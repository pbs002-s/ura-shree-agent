"""
Local inference engine for URA-Shree.

Generation runs off the model's incremental `step()` API, so the prompt is
encoded once and every subsequent token costs a single position of attention
instead of a full re-read of the context. On top of that:

  * Weights are loaded straight to the target device in the chosen compute
    dtype, so a float32 copy of the model never exists in host RAM.
  * Sampling filters run on the GPU; nothing crosses the PCIe bus per token
    except the single sampled ID.
  * Byte-level decoding is buffered, so a multi-byte character split across two
    BPE tokens is emitted whole rather than as replacement characters.
  * Stop sequences are matched against a rolling tail, not the whole transcript.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.model import ShreeTransformerLM
from tokenizer.tokenizer import BPETokenizer
from inference.fast_decode import FastDecoder
from inference.runtime import (
    tune_runtime,
    quantize_for_cpu,
    gpu_memory_snapshot,
    host_memory_snapshot,
)

DEFAULT_STOPS = ["<|eos|>", "<|user|>", "<|system|>", "<|endoftext|>"]


@dataclass
class GenerationStats:
    """Timing for one completed generation, for the telemetry panel."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.decode_ms <= 0:
            return 0.0
        return round(self.completion_tokens / (self.decode_ms / 1000.0), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prefill_ms": round(self.prefill_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
            "tokens_per_second": self.tokens_per_second,
        }


class InferenceEngine:
    """Loads a checkpoint and generates from it, entirely on local hardware."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best.pt",
        tokenizer_path: str = "checkpoints/tokenizer.json",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        quantize: bool = False,
        compile_model: bool = False,
        fast_decode: bool = True,
    ):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        self.tokenizer = BPETokenizer.load(tokenizer_path)
        self.checkpoint_path = checkpoint_path

        self.device, self.compute_dtype, self.profile = tune_runtime(device_preference=device)
        if dtype:
            self.compute_dtype = getattr(torch, dtype)
            self.profile.dtype = dtype

        # mmap keeps the checkpoint off the heap while the state dict is copied
        # tensor by tensor, so peak host RAM stays near one tensor rather than
        # a whole second copy of the model.
        try:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError):
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model_config = ModelConfig.from_dict(raw.get("config", {}).get("model", {}))
        # Dropout is a training-time regulariser; leaving it on at inference
        # only adds noise and a kernel launch per layer.
        self.model_config.dropout = 0.0

        self.model = ShreeTransformerLM(self.model_config, verbose=False)
        self.model.load_state_dict(raw["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.step_count = raw.get("step", 0)
        self.val_loss = raw.get("val_loss", 0.0)
        del raw

        if quantize and self.device.type == "cpu":
            self.model, applied = quantize_for_cpu(self.model)
            self.profile.quantized = applied
            if applied:
                self.profile.notes.append("Linear layers quantised to int8 (dynamic)")
        else:
            self.model = self.model.to(device=self.device, dtype=self.compute_dtype)

        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.profile.notes.append("torch.compile applied (reduce-overhead)")
            except Exception as err:
                self.profile.notes.append(f"torch.compile unavailable: {err}")

        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id
        self.last_stats = GenerationStats()

        # Decoding a small model is bound by kernel launches, not arithmetic.
        # The graph decoder collapses a step into one replay; see
        # inference/fast_decode.py. It is built lazily so loading stays cheap.
        self._fast: Optional[FastDecoder] = None
        self.use_graph_decode = (
            fast_decode and FastDecoder.available(self.device) and not self.profile.quantized
        )
        if self.use_graph_decode:
            self.profile.notes.append("CUDA graph decoding enabled")

    # -- introspection ------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Everything the UI needs to render the local-model panel."""
        footprint = self.model.memory_footprint()
        ctx = self.model_config.max_seq_len
        return {
            "checkpoint": self.checkpoint_path,
            "step": self.step_count,
            "val_loss": round(float(self.val_loss), 4),
            "parameters": self.model.get_num_params(),
            "non_embedding_parameters": self.model.get_num_params(non_embedding=True),
            "architecture": {
                "layers": self.model_config.num_layers,
                "heads": self.model_config.num_heads,
                "kv_heads": self.model_config.effective_kv_heads,
                "embed_dim": self.model_config.embed_dim,
                "context": ctx,
                "vocab_size": self.model_config.vocab_size,
                "pos_encoding": self.model_config.pos_encoding,
                "ffn": self.model_config.ffn,
            },
            "memory": {
                **footprint,
                "kv_cache_full_mb": round(self.model.kv_cache_bytes(ctx) / (1024 * 1024), 2),
                **gpu_memory_snapshot(),
                **host_memory_snapshot(),
            },
            "runtime": self.profile.to_dict(),
            "graph_decode": self._fast.info() if self._fast else {"captured": False},
            "last_generation": self.last_stats.to_dict(),
        }

    # -- generation ---------------------------------------------------------

    def _encode(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [self.tokenizer.bos_id]
        # Leave headroom so at least a few tokens can be generated.
        limit = self.model_config.max_seq_len - 8
        if len(ids) > limit:
            ids = ids[-limit:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _sample(
        self,
        logits: torch.Tensor,
        recent: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> int:
        """
        Picks the next token id from one row of logits.

        Filtering happens inside the top-k candidate set rather than across the
        whole vocabulary. Sorting 4096 logits to apply nucleus sampling when
        only 40 candidates can ever be selected is wasted work, and at this
        model size every avoided kernel is a measurable share of the step.
        """
        logits = logits.float()

        if repetition_penalty != 1.0:
            logits = ShreeTransformerLM._apply_repetition_penalty(
                logits.clone(), recent, repetition_penalty
            )

        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        logits = logits / temperature
        vocab = logits.size(-1)
        k = min(top_k, vocab) if top_k and top_k > 0 else vocab
        values, indices = torch.topk(logits, k, dim=-1)

        if 0.0 < top_p < 1.0:
            probs = F.softmax(values, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            # Keeping where (cumulative - probs) < top_p retains the token that
            # crosses the threshold, without a shift-and-scatter pass.
            values = values.masked_fill(cumulative - probs >= top_p, float("-inf"))

        choice = torch.multinomial(F.softmax(values, dim=-1), num_samples=1)
        return int(indices.gather(1, choice).item())

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.12,
        repetition_window: int = 128,
        stop_words: Optional[List[str]] = None,
    ) -> Generator[str, None, None]:
        """
        Yields decoded text as it is produced.

        `repetition_window` bounds the penalty to the recent tail. Penalising
        every token ever seen makes long outputs drift into nonsense, because
        the model ends up forbidden from using common words.
        """
        stops = DEFAULT_STOPS if stop_words is None else stop_words
        max_stop_len = max((len(s) for s in stops), default=0)

        tokens = self._encode(prompt)
        prompt_len = tokens.size(1)
        stats = GenerationStats(prompt_tokens=prompt_len)

        t0 = time.perf_counter()
        logits, cache = self.model.step(tokens)
        stats.prefill_ms = (time.perf_counter() - t0) * 1000.0

        # Hand the prefill cache to the graph decoder when it can take over.
        # It cannot once the prompt already fills the context, since there is
        # then no room left in the static buffer.
        fast = None
        if self.use_graph_decode and prompt_len < self.model_config.max_seq_len:
            if self._fast is None:
                self._fast = FastDecoder(self.model, self.device, self.compute_dtype)
            try:
                self._fast.reset(cache)
                fast = self._fast
            except Exception:
                fast = None

        emitted_ids: List[int] = []
        # Byte buffer: decode incrementally and only emit complete characters.
        pending_ids: List[int] = []
        emitted_text = ""

        # The token history is preallocated. Growing it with torch.cat would
        # allocate and copy the whole sequence on every single token.
        limit = self.model_config.max_seq_len
        history = torch.zeros(1, limit, dtype=torch.long, device=self.device)
        history[0, :prompt_len] = tokens[0]
        length = prompt_len

        decode_start = time.perf_counter()

        for _ in range(max_new_tokens):
            window_start = max(0, length - repetition_window)
            token_id = self._sample(
                logits,
                history[:, window_start:length],
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )
            if token_id == self.eos_id:
                break

            emitted_ids.append(token_id)
            pending_ids.append(token_id)

            # Decode the pending run; if it still ends mid-character, hold it.
            chunk = self.tokenizer.decode(pending_ids)
            if "�" in chunk:
                text_piece = ""
            else:
                text_piece = chunk
                pending_ids = []

            if text_piece:
                boundary = len(emitted_text)
                emitted_text += text_piece
                # A stop sequence can only have been completed by this piece, so
                # start the search just far enough back to catch a straddle.
                search_from = max(0, boundary - max_stop_len)
                cut = min(
                    (i for i in (emitted_text.find(s, search_from) for s in stops) if i >= 0),
                    default=-1,
                )
                if cut >= 0:
                    if cut > boundary:
                        yield emitted_text[boundary:cut]
                    break
                yield text_piece

            history[0, length] = token_id
            length += 1

            if length >= limit:
                # Context is full. Slide the window and rebuild the cache once,
                # then let the graph decoder pick the new state back up.
                keep = limit // 2
                history[0, :keep] = history[0, length - keep:length].clone()
                length = keep
                logits, cache = self.model.step(history[:, :length])
                if fast is not None:
                    try:
                        fast.reset(cache)
                    except Exception:
                        fast = None
            elif fast is not None:
                logits = fast.step(token_id)
            else:
                logits, cache = self.model.step(
                    history[:, length - 1:length], past_key_values=cache
                )

        stats.decode_ms = (time.perf_counter() - decode_start) * 1000.0
        stats.completion_tokens = len(emitted_ids)
        self.last_stats = stats

    def generate(self, prompt: str, **kwargs) -> str:
        """Non-streaming generation; returns the whole completion as one string."""
        return "".join(self.generate_stream(prompt, **kwargs))
