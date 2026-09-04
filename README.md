# URA-Shree

A coding agent and chat interface that runs on a language model built from
scratch in this repository, or on any model you have an API key for.

Built by [pbs002-s](https://github.com/pbs002-s).

The model, the byte-level BPE tokenizer, the training loop, the agent harness,
the sandboxed tools and the interface are all here. Nothing is wrapped around
someone else's SDK. What is *not* claimed: an 11M-parameter model trained on a
megabyte of text will not write your code. It exists so the whole stack is
inspectable end to end. For real work, paste an API key and pick a model - the
agent, the tools and the interface are the same either way.

---

## What is here

| Piece | What it does |
| :--- | :--- |
| **Local model** | Decoder-only transformer, 11.3M parameters, RoPE, SwiGLU, grouped-query attention, weight tying, KV cache |
| **Tokenizer** | Byte-level BPE trained on this repository's own source, 4096 tokens, lossless |
| **Providers** | Anthropic, OpenAI, Google, Groq, OpenRouter, DeepSeek, Mistral, xAI, Together, Fireworks, Cerebras, Ollama, LM Studio, or any OpenAI-compatible endpoint |
| **Agent** | A real tool-calling loop: read, edit, search, run commands, index symbols, remember decisions |
| **Terminal** | A persistent shell. `cd`, environment variables and activated virtualenvs survive between commands |
| **Time Machine** | Content-addressed snapshots of the whole workspace, with branch, diff and restore |
| **Interface** | React and TypeScript, light and dark themes, streaming replies, no dependencies beyond React |

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch first, matched to your hardware
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or .../cpu
pip install -r requirements.txt

python scripts/launch.py
```

That builds the interface and serves everything on <http://127.0.0.1:8000>.

To use a hosted model: open **Settings**, choose a provider, paste your key,
press **Scan available models**, and pick one. The scan calls the provider's
own model listing, so you get exactly what your account can reach rather than a
hard-coded list. Keys are stored in `.shree/settings.json`, which is gitignored,
and the API only ever returns a masked preview of them.

To point the agent at a different project:

```powershell
python scripts/launch.py --workspace ../my-project
```

---

## The Time Machine

Most coding agents give you an undo. This one gives you a history you can
navigate.

Before every write, edit or command, the entire workspace is snapshotted. A
snapshot records `path -> sha256` and each distinct blob is stored once,
zlib-compressed, so snapshotting a large project where one file changed writes
exactly one new object. That is what makes it cheap enough to do on every single
tool call rather than at checkpoints you have to remember to set.

From the timeline panel you can:

- **Diff any two points**, not just consecutive ones.
- **Read a file as it was** at any point, without restoring anything.
- **Restore**, which snapshots the present first and then records the restored
  state as a child of the point you went back to. Rewinding therefore *forks*
  the history instead of deleting it: the branch you abandoned stays reachable,
  and going forward again is one click.

The branch points are marked in the timeline, so a session where you tried three
approaches reads as three approaches rather than a flat list of edits.

```
python -c "from agent.timemachine import TimeMachine; print(TimeMachine('.').timeline())"
```

---

## Performance

Measured on an RTX 4060 Laptop GPU (8 GB), 8 physical cores, with
`python scripts/benchmark.py`.

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

Model quality is compared in bits per character, not loss, because the
tokenizer changed underneath it. Per-token loss falls automatically when the
vocabulary shrinks, so it would flatter the new model for the wrong reason.
`loss / ln(2) / chars_per_token` normalises that away: 5.5042 at 1.57 chars per
token against 4.5325 at 3.19.

### Why decoding was slow, and what fixed it

Profiling the decode loop showed roughly 180 kernel launches per token, about
0.5 ms of actual GPU work, and about 10 ms of wall time. The arithmetic was
never the bottleneck - the CPU cost of submitting it was. A large model hides
that behind real work; an 11M-parameter one cannot.

Three changes fixed it:

1. **A static KV cache.** The ordinary cache grows a column per step, so every
   step has new shapes, which defeats both graph capture and `torch.compile`.
   The static cache is allocated once at full context length and new keys and
   values are written in place at a position held in a device tensor.
2. **CUDA graph capture.** With shapes fixed, one decode step is recorded once
   and replayed, submitting the whole step as a single operation.
3. **A cheaper sampling path.** Once the model step was fast, the Python around
   it dominated. Nucleus filtering now runs inside the top-k candidate set
   rather than sorting the whole vocabulary, and the token history is
   preallocated instead of grown with `torch.cat` on every token. That alone
   took end-to-end generation from 197 to 742 tok/s.

Output is bit-identical to the eager path; `tests/test_fast_decode.py` asserts
that on three architecture variants (learned positions, RoPE, and RoPE with
grouped-query attention and SwiGLU). See `inference/fast_decode.py`.

### Memory

- Weights load straight to the device in the compute dtype, so a float32 copy
  never exists in host RAM. `mmap` keeps the checkpoint off the heap while the
  state dict is copied.
- bfloat16 on Ampere and newer, float16 on older CUDA cards, float32 on CPU.
- `--quantize` applies dynamic int8 to the linear layers on CPU, roughly four
  times less resident weight memory.
- Threads are pinned to physical cores. PyTorch's default of one thread per
  logical core oversubscribes a hyperthreaded laptop and makes small GEMMs
  slower.

---

## The agent

The loop is in `agent/loop.py` and it is small on purpose: send the
conversation, stream back text and tool calls, run the tools, append the
results, repeat until the model stops asking. There is no keyword matching and
no hand-written intent table. The model decides; this code makes sure it can act
and reports honestly what happened.

Tools available to it:

| Tool | Notes |
| :--- | :--- |
| `read_file` | Returns content with line numbers |
| `write_file` | Replaces a whole file |
| `edit_file` | Exact-string replacement; an ambiguous match is refused, not guessed |
| `list_dir`, `search` | Navigate before reading |
| `run_command` | The persistent shell, so state carries between calls |
| `find_symbols` | Python AST index |
| `git_status` | Working tree and diff |
| `remember` | Writes a durable fact to SQLite |
| `snapshot` | An explicit Time Machine point |

Every mutating tool is snapshotted before it runs, so the model does not have to
remember to make its work reversible.

**Response style** is enforced in `agent/prompts.py`: answer the question, skip
the preamble, prefer a list or a code block over a paragraph, no filler, no
emoji. Un-steered API models pad; this is what stops that.

---

## Training your own checkpoint

```powershell
python -m tokenizer.train_tokenizer --vocab-size 4096   # learns merges from the source tree
python -m data.prepare_dataset                          # builds train.bin and val.bin
python -m training.train --config configs/small.yaml
python -m training.train_coding --steps 200             # supervised fine-tune
python -m inference.chat --model checkpoints/coding_best.pt
```

`configs/` holds small (11M), medium (82M) and large (283M) presets. All three
use RoPE, SwiGLU and grouped-query attention. Checkpoints saved before those
options existed still load: the defaults reproduce the original architecture
exactly, and the architecture is read from the checkpoint's own config.

The honest limit is data. The corpus is this repository plus a seed set - about
320,000 tokens. An 11M-parameter model wants a few hundred million. Point
`data/source_corpus.py` at a larger tree to change that.

---

## Layout

```text
agent/        loop, toolkit, prompts, memory, AST index, time machine
providers/    catalogue, base types, HTTP providers, local provider
model/        config, embeddings, attention, transformer blocks, the model
inference/    engine, runtime tuning, CUDA graph decoder, chat CLI
tools/        sandboxed filesystem, persistent shell, git
server/       FastAPI app and settings storage
training/     training loops, checkpoints, evaluation
tokenizer/    byte-level BPE, vocabulary, trainer
data/         corpus builders, dataset preparation, preprocessing
frontend/     React interface
scripts/      launch, benchmark, hardware probe
tests/        104 tests
```

---

## Tests

```powershell
.\.venv\Scripts\pytest -q
```

101 tests covering: causal masking and no future-token leakage, KV cache parity
against the uncached path, CUDA graph output equality, tokenizer round-trip and
lossless byte fallback, sandbox containment and path traversal, blocked
destructive commands, shell state persistence and timeout recovery, provider
request shaping and streamed tool-call assembly for both the OpenAI and
Anthropic protocols, agent loop tool dispatch and turn budgeting, a real
self-healing run where a failing pytest is diagnosed, patched and re-run, and
the full HTTP and WebSocket API.

---

## Notes on safety

- Filesystem tools resolve every path and refuse anything outside the workspace.
- Uploads refuse executables, path traversal, and anything over 8 MB.
- A short list of unrecoverable shell commands is refused outright. This is not
  a sandbox; the shell runs with your permissions. Point `--workspace` at what
  you intend the agent to touch.
- API keys are stored obfuscated in `.shree/settings.json` and protected by file
  permissions where the OS supports it. That is not encryption, and it is not
  presented as such - the key never leaves your machine except to the provider
  it belongs to, and the API returns only a masked preview.
