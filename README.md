# URA-Shree

A modern autonomous AI coding assistant and agent interface powered by a custom Transformer language model built from scratch in this repository, governed by the **PACES** framework, or by any frontier model you provide an API key for.

Created and developed by **[Pritam](https://github.com/pbs002-s)** from **DIU (Daffodil International University)**.

The custom Transformer architecture, the byte-level BPE tokenizer, the training pipeline, the agent harness, the sandboxed tools, and the interface are all built here natively. Inspectable end to end, scalable from **11.3M (Small)** up to **78.7M (Medium)** and **283M (Large)** parameters on consumer GPUs.

---

## The PACES Framework

URA-Shree is developed upon five core engineering pillars:
* **P - Performance**: Ultra-fast inference (**348.8 tok/s** decode with CUDA graphs), static KV caching, and BF16 mixed precision.
* **A - Architecture**: Modern decoder-only Transformer with Rotary Position Embeddings (RoPE), SwiGLU activations, and Grouped-Query Attention (GQA).
* **C - Capability**: Autonomous coding agent with file system tools, persistent terminal sessions, and Time Machine snapshot rollback.
* **E - Evaluation**: Automated validation tracking (**+80.2% loss reduction**, **+99.8% perplexity reduction**), before/after generation benchmarking.
* **S - Security**: Strict path boundaries, interactive file mutation approvals, and zero-leak credential management.

---

## What is here

| Piece | What it does |
| :--- | :--- |
| **Local model** | Custom Transformer LM: 11.3M (Small) & **78.7M (Medium)** with 2,048 context window, RoPE, SwiGLU, GQA, and KV cache |
| **Tokenizer** | Byte-level BPE trained on this repository's own source, 4096 tokens, lossless |
| **Providers** | Anthropic, OpenAI, Google, Groq, OpenRouter, DeepSeek, Mistral, xAI, Together, Fireworks, Cerebras, Ollama, LM Studio, or any OpenAI-compatible endpoint |
| **Agent** | A real tool-calling loop: read, edit, search, run commands, index symbols, remember decisions |
| **Approval** | Every write is a decision you make, shown with the path it is about to touch |
| **Skills** | Named blocks of standing directives, toggled at will, that shape how the agent works |
| **Terminal** | A persistent shell. `cd`, environment variables and activated virtualenvs survive between commands |
| **Time Machine** | Content-addressed snapshots of the whole workspace, with branch, diff and restore |
| **Distillation** | Turn a local Ollama teacher into training data for the from-scratch checkpoint, with no hosted API in the loop |
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
own model listing, so you get exactly what your account can reach rather than
a hard-coded list. Keys are stored in `.shree/settings.json`, which is
gitignored, and the API only ever returns a masked preview of them.

To point the agent at a different project:

```powershell
python scripts/launch.py --workspace ../my-project
```

or press **Open Folder** in the files panel and pick one from your own file
explorer, at any time, without restarting.

---

## Documentation

| Document | Covers |
| :--- | :--- |
| [Architecture](docs/architecture.md) | How the four layers fit together, the provider seam, the request lifecycle, where state lives |
| [The agent](docs/agent.md) | The loop, the tools, approval, the system prompt, skills |
| [API](docs/api.md) | Every HTTP route and both websocket protocols |
| [Model](docs/model.md) | Architecture, tokenizer, the fast decode path, benchmarks |
| [Training](docs/training.md) | Training from scratch, the distillation pipeline, the Ollama persona |
| [Configuration](docs/configuration.md) | Flags, environment, provider keys, what gets written where |

---

## Two modes

The agent runs with or without a project folder.

**With a workspace**, it has the full toolkit - read, write, edit, search, run
commands, index symbols, git - along with the file explorer, the terminal and
the Time Machine. Every mutating call is approved before it runs and
snapshotted before it lands.

**With no folder open**, it is an ordinary assistant. There is no toolkit, no
tool specification is sent to the model, and the system prompt drops its tool
section entirely rather than describing tools that are not there. Ask it to
save a file and it tells you to open a folder first.

The default workspace is `./workspace` rather than the repository, so a first
run cannot edit the engine by accident. Switching is a runtime operation: the
filesystem tool, git tool, shell, indexer and Time Machine are all rebuilt
against the new root.

---

## Approval

`auto_approve` is off by default. When the agent wants to write, edit or run
something that changes the workspace, the tool card in the chat becomes a
prompt showing the path in question, and the run pauses until you answer.

Denial, a timeout, stopping the turn and resetting the conversation all resolve
the same way: the tool does not run, and the model is told so in the tool
result rather than being left to wonder why nothing happened.

The card names the file because "allow `write_file`?" is not a question anyone
can answer.

---

## Skills

A skill is a named block of standing directives - a UI/UX brief, a security
checklist, an architecture stance - appended to the system prompt while it is
enabled. Four ship enabled; add your own from the Skills panel.

They are stored in `.shree/skills.json`, and built-ins are merged back in on
load, so a skills file written by an older version never silently loses the
ones added since.

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

**Reasoning** is normalised before it reaches the interface. Some models return
their chain of thought in a separate field; others inline it into the answer
inside `<think>` tags. Both become `thinking` events, rendered as a collapsed
card, so a long chain of thought never buries the answer.

Full detail in [docs/agent.md](docs/agent.md).

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
| Model quality | 5.06 bits/char | **2.05 bits/char** | bits per character, not loss - the tokenizer changed underneath it |

Profiling the decode loop showed roughly 180 kernel launches per token, about
0.5 ms of actual GPU work, and about 10 ms of wall time. The arithmetic was
never the bottleneck - the CPU cost of submitting it was. A static KV cache,
CUDA graph capture and a cheaper sampling path fixed it, and the output stays
bit-identical to the eager path; `tests/test_fast_decode.py` asserts that on
three architecture variants.

The full account, including why bits per character rather than loss, is in
[docs/model.md](docs/model.md).

---

## Training your own checkpoint

```powershell
python -m tokenizer.train_tokenizer --vocab-size 4096   # learns merges from the source tree
python -m data.prepare_dataset                          # builds train.bin and val.bin
python -m training.train --config configs/small.yaml
python -m training.train_coding --steps 200             # supervised fine-tune
python -m inference.chat --model checkpoints/coding_best.pt
```

`configs/` holds small (11.3M), medium (**78.7M**), and large (283M) presets. All three
use RoPE, SwiGLU, and grouped-query attention.

### Fine-Tuning on Custom Agent Data & System Prompt Leaks

To train the **85M parameter model (`ura-shree-medium`)** on clean identity, coding dialogues, and real-world system prompt leaks (`system_prompts_leaks-main`):

```powershell
python scripts/train_custom_datasets.py --config configs/medium.yaml --steps 300 --max-did 25000
```

#### 85M Model Convergence & Verification

| Benchmark Metric | Baseline (Initial) | Fine-Tuned 85M (`ura-shree-medium`) | Progress |
| :--- | :--- | :--- | :--- |
| **Validation Loss** | `7.5424` | **`1.4931`** | **+80.20% improvement** |
| **Validation Perplexity** | `1886.39` | **`4.45`** | **+99.76% improvement** |
| **CUDA Graph Decode** | — | **348.8 tok/s** | Zero CPU submission overhead |
| **Prefill Latency** | — | **16.7 ms** | Fast initial token response |
| **Peak GPU VRAM** | — | **1.64 GB** | Runs comfortably within 8GB VRAM |

A model trained on source code continues source code. Asked a question, it
writes a plausible function, because that is what follows a line of text in its
training data. `scripts/distill_and_train.py` fixes the shape of the problem
without buying data - it prompts a local Ollama teacher, caches the replies,
formats them into the ChatML the model already understands, tokenises them with
the project's own BPE, and fine-tunes on top of the base checkpoint:

```powershell
ollama pull qwen3.5:0.8b
python scripts/distill_and_train.py --count 25 --steps 60
```

No hosted API is called and no third-party weights are shipped - only the
teacher's own text, which is what keeps the result distributable. See
[docs/training.md](docs/training.md).

The honest limit is still data. The base corpus is this repository plus a seed
set - about 320,000 tokens, where an 11M-parameter model wants a few hundred
million. Point `data/source_corpus.py` at a larger tree to change that.

---

## Layout

```text
agent/        loop, toolkit, prompts, memory, AST index, time machine
providers/    catalogue, base types, HTTP providers, local provider
model/        config, embeddings, attention, transformer blocks, the model
inference/    engine, runtime tuning, CUDA graph decoder, chat CLI
tools/        sandboxed filesystem, persistent shell, git
server/       FastAPI app, settings storage, skills registry
training/     training loops, checkpoints, evaluation
tokenizer/    byte-level BPE, vocabulary, trainer
data/         corpus builders, dataset preparation, preprocessing
frontend/     React interface
scripts/      launch, benchmark, hardware probe, distillation, folder picker
docs/         architecture, agent, API, model, training, configuration
tests/        101 tests
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
- Mutating tools require approval by default.
- A short list of unrecoverable shell commands is refused outright. This is not
  a sandbox; the shell runs with your permissions. Point `--workspace` at what
  you intend the agent to touch.
- API keys are stored obfuscated in `.shree/settings.json` and protected by file
  permissions where the OS supports it. That is not encryption, and it is not
  presented as such - the key never leaves your machine except to the provider
  it belongs to, and the API returns only a masked preview.
