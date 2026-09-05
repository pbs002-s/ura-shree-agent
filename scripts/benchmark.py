"""
Measures what the inference optimisations actually buy, on this machine.

Reports three things:
  * decode throughput with and without the KV cache
  * prefill and decode latency at the resolved precision
  * resident weight memory, and what int8 quantisation saves on CPU

Run it after any change to the model or the runtime tuner. Numbers that move
the wrong way are the point of having it.

    python scripts/benchmark.py --checkpoint checkpoints/coding_best.pt
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from inference.runtime import gpu_memory_snapshot, host_memory_snapshot, tune_runtime
from model.config import ModelConfig
from model.model import ShreeTransformerLM


def synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_generate(model, prompt, tokens, use_cache, device, repeats=3) -> float:
    """Median seconds to generate `tokens` tokens, warm-up excluded."""
    model.generate(prompt, max_new_tokens=4, temperature=0.0, use_cache=use_cache)
    synchronise(device)

    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.generate(prompt, max_new_tokens=tokens, temperature=0.0, use_cache=use_cache)
        synchronise(device)
        runs.append(time.perf_counter() - start)
    return statistics.median(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark URA-Shree inference")
    parser.add_argument("--checkpoint", default="checkpoints/coding_best.pt")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device, dtype, profile = tune_runtime(device_preference=args.device)

    print("=" * 66)
    print("  URA-Shree inference benchmark")
    print("=" * 66)
    print(f"  Device        {profile.device_name} ({profile.device})")
    print(f"  Precision     {profile.dtype}")
    print(f"  CPU threads   {profile.cpu_threads} of {profile.logical_cores} logical")
    for note in profile.notes:
        print(f"                {note}")
    print("-" * 66)

    if os.path.exists(args.checkpoint):
        raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg_dict = raw.get("config", {})
        if isinstance(cfg_dict, dict) and "model" in cfg_dict and isinstance(cfg_dict["model"], dict):
            cfg_dict = cfg_dict["model"]
        config = ModelConfig.from_dict(cfg_dict)
        config.dropout = 0.0
        model = ShreeTransformerLM(config, verbose=False)
        model.load_state_dict(raw["model_state_dict"])
        source = args.checkpoint
        del raw
    else:
        config = ModelConfig(
            vocab_size=2048, max_seq_len=1024, embed_dim=384, num_layers=6,
            num_heads=6, intermediate_dim=1536, dropout=0.0,
        )
        model = ShreeTransformerLM(config, verbose=False)
        source = "randomly initialised (no checkpoint found)"

    model = model.to(device=device, dtype=dtype).eval()
    footprint = model.memory_footprint()

    print(f"  Weights       {source}")
    print(f"  Parameters    {model.get_num_params():,}")
    print(f"  Resident      {footprint['total_mb']} MB")
    print(f"  Context       {config.max_seq_len} tokens")
    print(f"  KV cache      {model.kv_cache_bytes(config.max_seq_len) / 1e6:.1f} MB when full")
    print("-" * 66)

    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_tokens), device=device)

    cached = time_generate(model, prompt, args.new_tokens, True, device)
    uncached = time_generate(model, prompt, args.new_tokens, False, device)

    print(f"  Generating {args.new_tokens} tokens after a {args.prompt_tokens}-token prompt:")
    print(f"    with KV cache     {cached * 1000:8.1f} ms   {args.new_tokens / cached:8.1f} tok/s")
    print(f"    without KV cache  {uncached * 1000:8.1f} ms   {args.new_tokens / uncached:8.1f} tok/s")
    print(f"    speedup           {uncached / cached:8.2f}x")
    print("-" * 66)

    with torch.inference_mode():
        synchronise(device)
        start = time.perf_counter()
        _, cache = model.step(prompt)
        synchronise(device)
        prefill_ms = (time.perf_counter() - start) * 1000

        single = prompt[:, -1:]
        start = time.perf_counter()
        for _ in range(32):
            _, cache = model.step(single, past_key_values=cache)
        synchronise(device)
        decode_ms = (time.perf_counter() - start) * 1000 / 32

    print(f"  Prefill ({args.prompt_tokens} tokens)   {prefill_ms:7.1f} ms")
    print(f"  Decode, eager           {decode_ms:7.2f} ms/token  {1000 / decode_ms:7.1f} tok/s")

    # The graph decoder is the answer to launch-bound decoding; see
    # inference/fast_decode.py for why a small model needs it.
    from inference.fast_decode import FastDecoder

    if FastDecoder.available(device):
        with torch.inference_mode():
            _, prefill = model.step(prompt)
            decoder = FastDecoder(model, device, dtype)
            decoder.reset(prefill)
            for _ in range(8):
                decoder.step(7)
            synchronise(device)
            start = time.perf_counter()
            for _ in range(64):
                decoder.step(7)
            synchronise(device)
            graph_ms = (time.perf_counter() - start) * 1000 / 64

        info = decoder.info()
        if info["captured"]:
            print(f"  Decode, CUDA graph      {graph_ms:7.2f} ms/token  {1000 / graph_ms:7.1f} tok/s")
            print(f"  Launch overhead removed {decode_ms / graph_ms:7.2f}x")
            print(f"  Static cache            {info['cache_mb']} MB")
        else:
            print(f"  CUDA graph capture failed: {info['error']}")

    memory = {**gpu_memory_snapshot(), **host_memory_snapshot()}
    print("-" * 66)
    print(f"  Process RSS   {memory['process_rss_mb']:.0f} MB")
    if torch.cuda.is_available():
        print(f"  VRAM in use   {memory['allocated_mb']:.0f} MB of {memory['total_mb']:.0f} MB")

    if device.type == "cpu":
        from inference.runtime import quantize_for_cpu

        quantized, applied = quantize_for_cpu(model)
        if applied:
            after = sum(
                p.numel() * p.element_size() for p in quantized.parameters()
            ) / (1024 * 1024)
            print(f"  int8 weights  {after:.1f} MB (from {footprint['parameters_mb']} MB)")
    print("=" * 66)


if __name__ == "__main__":
    main()
