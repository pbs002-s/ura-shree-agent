# The model and the inference path

A decoder-only transformer written from scratch in `model/`, with a byte-level
BPE tokenizer trained on this repository's own source. It exists so the stack
is inspectable end to end, not because an 11M-parameter model is going to
write your code.

## Architecture

| | |
| :--- | :--- |
| Positions | RoPE (`rope_theta` 10000), or a learned table on older checkpoints |
| Feed-forward | SwiGLU, hidden width scaled by 2/3 to hold the parameter count |
| Attention | Grouped-query; `small` shares 3 KV heads across 6 query heads |
| Norm | RMSNorm, no bias terms anywhere |
| Embeddings | Tied input and output |

RoPE encodes position as a rotation inside attention, so no position table is
allocated at all and the model degrades gracefully past its trained context
length. Grouped-query attention shrinks the KV cache by the query-to-KV ratio,
which is what makes a long context affordable on a laptop.

`configs/` holds three presets:

| Preset | Params | Layers | `d_model` | Heads | KV heads | Context | Vocab |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | ~11M | 6 | 384 | 6 | 3 | 1024 | 4096 |
| `medium` | ~82M | 12 | 768 | 12 | 4 | 2048 | 4096 |
| `large` | ~283M | 24 | 1024 | 16 | 4 | 2048 | 16384 |

Architecture is read from the checkpoint's own config, and the defaults
reproduce the original pre-RoPE design exactly, so checkpoints saved before
these options existed still load.

## Tokenizer

Byte-level BPE, 4096 merges, trained on the source tree it will be asked to
model. Lossless: any byte sequence round-trips, because the base vocabulary is
all 256 bytes and unknown input falls back to them rather than to an `<unk>`
token.

Training on real source rather than prose is what takes compression from 1.57
to 3.19 characters per token. Those are the same characters; the vocabulary
simply learned that `    def ` and `self.` are single units.

## Fast decoding

Profiling the decode loop showed roughly 180 kernel launches per token, about
0.5 ms of GPU work, and about 10 ms of wall time. The arithmetic was never the
bottleneck - the CPU cost of submitting it was. A large model hides that behind
real work; an 11M-parameter one cannot.

1. **A static KV cache.** The ordinary cache grows a column per step, so every
   step has new shapes, which defeats both graph capture and `torch.compile`.
   The static cache is allocated once at full context length and new keys and
   values are written in place at a position held in a device tensor.
2. **CUDA graph capture.** With shapes fixed, one decode step is recorded once
   and replayed, submitting the whole step as a single operation.
3. **A cheaper sampling path.** Once the model step was fast, the Python around
   it dominated. Nucleus filtering runs inside the top-k candidate set rather
   than sorting the whole vocabulary, and the token history is preallocated
   instead of grown with `torch.cat` per token. That alone took end-to-end
   generation from 197 to 742 tok/s.

Output is bit-identical to the eager path. `tests/test_fast_decode.py` asserts
that on three architecture variants: learned positions, RoPE, and RoPE with
grouped-query attention and SwiGLU. See `inference/fast_decode.py`.

## Memory

- Weights load straight to the device in the compute dtype, so a float32 copy
  never exists in host RAM. `mmap` keeps the checkpoint off the heap while the
  state dict is copied.
- bfloat16 on Ampere and newer, float16 on older CUDA cards, float32 on CPU.
- `--quantize` applies dynamic int8 to the linear layers on CPU, roughly four
  times less resident weight memory.
- Threads are pinned to physical cores. PyTorch's default of one thread per
  logical core oversubscribes a hyperthreaded laptop and makes small GEMMs
  slower.

## Measured

RTX 4060 Laptop GPU (8 GB), 8 physical cores, via `python scripts/benchmark.py`.

| | Before | After | What changed |
| :--- | ---: | ---: | :--- |
| Training throughput | 18,837 tok/s | **116,315 tok/s** | fused SDPA, fused RMSNorm |
| Generation, end to end | 197 tok/s | **742 tok/s** | graph decode plus a cheaper sampling path |
| Decode step, GPU | 10.1 ms | **1.5-2.0 ms** | CUDA graph replay |
| Decode, CPU, 512-token prompt | 7.4 tok/s | **65.4 tok/s** | KV cache |
| Tokenizer compression | 1.57 chars/token | **3.19 chars/token** | 4096-token vocabulary trained on real source |
| KV cache at full context | 18.9 MB | **4.7 MB** | grouped-query attention, bfloat16 |
| Model quality | 5.06 bits/char | **2.05 bits/char** | see below |

End-to-end generation is the median of five 200-token runs. The decode-step
figure is a range because this is a laptop GPU and its clocks move: an isolated
tight loop reaches 0.67 ms, a full benchmark pass sits nearer 2 ms. The ratio
between eager and graph decode holds at roughly 4-7x either way. Re-run
`scripts/benchmark.py` rather than trusting these numbers on your hardware.

Quality is compared in bits per character, not loss, because the tokenizer
changed underneath it. Per-token loss falls automatically when the vocabulary
shrinks, so it would flatter the new model for the wrong reason.
`loss / ln(2) / chars_per_token` normalises that away: 5.5042 at 1.57 chars per
token against 4.5325 at 3.19.
