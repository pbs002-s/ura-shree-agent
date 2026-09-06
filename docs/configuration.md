# Configuration

## Launching

```powershell
python scripts/launch.py                       # build the frontend, serve on 127.0.0.1:8000
python scripts/launch.py --workspace ../app    # point the agent at another project
python scripts/launch.py --dev                 # Vite dev server on 5173, hot reload
python scripts/launch.py --no-build            # serve the existing bundle as-is
python scripts/launch.py --host 0.0.0.0 --port 9000
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--workspace` | `./workspace` | Directory the agent works in |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port |
| `--dev` | off | Run Vite alongside for hot reload |
| `--no-build` | off | Skip the frontend build |

## Environment

| Variable | Effect |
| :--- | :--- |
| `SHREE_WORKSPACE` | Initial workspace root. Set by the launcher from `--workspace` |
| `SHREE_DEV` | When the workspace is the repository itself, show the framework directories in the explorer instead of hiding them |
| `SHREE_MODE` | `local` (default) or `cloud`. Cloud turns on accounts, projects, container sandboxes, rate limits and the audit trail |
| `SHREE_API_KEY` | Optional shared secret for a local install. Compared in constant time, and **not** honoured in cloud mode - a single static string that makes its holder an administrator is the opposite of per-user identity |
| `SHREE_ALLOWED_ORIGINS` | Comma-separated CORS origins. `*` is accepted but disables credentialed requests, because a wildcard origin with credentials turns any page on the internet into an authenticated client |
| `SHREE_ALLOW_HOST_PICKER` | Enables the native folder dialog. Local installs only: it draws a window on the machine running the server |
| `SHREE_LOG_JSON` | Structured JSON logs. On by default in cloud mode |
| `SHREE_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Cloud mode adds about thirty more, covering identity, storage, the sandbox
limits, rate limits and tracing. They are all listed with their defaults in
[`.env.example`](../.env.example) and explained in
[Deployment](deployment.md); none of them are needed for a local run.

The workspace defaults to `./workspace` rather than the repository, so a first
run cannot edit the engine by accident. When the root *is* the repository and
`SHREE_DEV` is unset, `agent/`, `model/`, `server/`, `frontend/` and the rest
are filtered out of the file tree - the agent can still reach them, the
explorer simply does not lead with them.

It can also be changed at runtime from the files panel, including to no folder
at all, which puts the agent into general chat mode.

## Providers and keys

Open **Settings**, choose a provider, paste a key, press **Scan available
models**, pick one. The scan calls the provider's own model listing, so you
get exactly what your account can reach rather than a hard-coded list.

Supported: Anthropic, OpenAI, Google, Groq, OpenRouter, DeepSeek, Mistral,
xAI, Together, Fireworks, Cerebras, Ollama, LM Studio, or any
OpenAI-compatible endpoint.

Keys live in `.shree/settings.json`, which is gitignored. They are stored
obfuscated and protected by file permissions where the OS supports it. That is
not encryption and is not presented as such - the key never leaves your
machine except to the provider it belongs to, and the API returns only a
masked preview.

## Files the app writes

| Path | Contents |
| :--- | :--- |
| `.shree/settings.json` | Keys, active provider and model, generation defaults |
| `.shree/skills.json` | Built-in and custom skills |
| `<workspace>/.timemachine/` | Snapshot object store and history database |
| `<workspace>/checkpoints/shree_memory.db` | Facts written by the `remember` tool |
| `workspace/` | The default working directory |

All of them are gitignored.

## Safety

- Filesystem tools resolve every path and refuse anything outside the
  workspace.
- Uploads refuse executables, path traversal, absolute paths, drive letters
  and anything over 8 MB.
- A short list of unrecoverable shell commands is refused outright.
- Mutating tools require approval by default; `auto_approve` turns that off
  per request.

This is not a sandbox. The shell runs with your permissions, and a model that
is allowed to run commands is allowed to run *commands*. Point `--workspace`
at what you intend the agent to touch, and leave approval on for anything you
have not read.

## Model Presets & Hardware Sizing

`configs/` contains pre-tuned configurations for training and local inference:

| Config | Architecture | Parameters | Context | Recommended Hardware |
| :--- | :--- | ---: | ---: | :--- |
| `configs/small.yaml` | 6 layers, 384 dim | 11.3M | 1,024 | Any CPU or Entry GPU (<1 GB VRAM) |
| `configs/medium.yaml` | 12 layers, 768 dim | **78.7M** | **2,048** | Consumer GPUs (RTX 4060 8GB, ~2.5 GB VRAM) |
| `configs/large.yaml` | 24 layers, 1024 dim | ~283M | 2,048 | 8GB - 16GB Dedicated VRAM with BF16 |
