# Architecture

URA-Shree is four layers that only talk to each other through narrow seams: a
model, a set of providers, an agent, and a server that a browser drives. Any
one of them can be replaced without touching the other three, which is the
point - the local model is the reference implementation of a provider, not a
dependency of the agent.

```
browser (React)
   │  HTTP for state, two websockets for streams
   ▼
server/api.py ──────────── settings, skills, workspace, Time Machine, uploads
   │
   ├── agent/ ──────────── CodingAgent → AgentSession → provider.stream()
   │      │
   │      └── tools/ ───── filesystem, shell, git   (every write snapshotted)
   │
   └── providers/ ─────── HTTP providers │ local provider
                                          └── inference/ → model/ → tokenizer/
```

## The seam that matters

Everything above `providers/` is written against one interface:

```python
async def stream(model, messages, system, tools, temperature, max_tokens)
    -> AsyncGenerator[StreamEvent, None]
```

`StreamEvent` has a `type` of `text`, `thinking`, `tool_call`, `usage` or
`error`. The agent loop never learns which provider produced an event, so the
same loop, the same tools and the same interface run against a hosted frontier
model or against the 11M-parameter checkpoint in `checkpoints/`. Adding a
provider is a subclass and a catalogue entry, not a change to the agent.

Two protocol shapes are implemented behind that interface: the
OpenAI-compatible one (which most vendors and every local server speak) and
the Anthropic one. Reasoning arrives in either a `reasoning_content` delta or
inlined into `content` inside `<think>` tags; both are normalised to
`thinking` events before they leave the provider, so nothing above has to know
the difference.

## Request lifecycle

A message typed in the browser takes this path:

1. The browser sends `{action: "send", message, model, temperature, ...}` over
   `/ws/agent`.
2. `agent_socket` reads the active provider and model from settings, builds or
   reuses the `CodingAgent` for that session id, and calls `run()` with the
   enabled skills' prompt and an approval callback bound to this socket.
3. `AgentSession.run` assembles the system prompt once
   (`agent/prompts.py`), then loops: stream from the provider, collect text
   and tool calls, and stop when a turn produces no tool calls.
4. Each tool call is approved if approval is required, then executed on a
   worker thread - tools are blocking, and running them on the event loop
   would stall every other socket. A mutating tool takes a Time Machine
   snapshot inside `Toolkit.execute`, immediately before it runs.
5. Results are appended to the conversation as tool messages and the loop goes
   round again, up to `max_turns`.
6. Every step is forwarded to the browser as a JSON event, so the interface is
   a projection of the loop's state rather than a second copy of it.

## Two modes

The server holds an optional workspace root. With one set, the agent gets the
full toolkit, the file explorer, the shell, the indexer and the Time Machine.
With none, `WORKSPACE_ROOT` is `None`, every workspace-dependent object is
`None`, the system prompt drops its tool discipline entirely, and the agent is
an ordinary assistant. The mode is switched at runtime through
`POST /api/workspace/select`, which rebuilds all of those objects and clears
the cached agents.

This is why so many endpoints begin with a null check. It is not defensive
padding: a null workspace is a supported state, not an error.

## State, and where it lives

| What | Where | Notes |
| :--- | :--- | :--- |
| API keys, provider selection | `.shree/settings.json` | Obfuscated, masked on read, gitignored |
| Skills | `.shree/skills.json` | Built-ins merged back in on load |
| Workspace snapshots | `<workspace>/.timemachine/` | Content-addressed, zlib-compressed |
| Project memory | `<workspace>/checkpoints/shree_memory.db` | SQLite, written by the `remember` tool |
| Conversation | Process memory | Per session id, cleared on `reset` |
| Model weights | `checkpoints/*.pt` | Not in git; retrain or download |

Conversations are deliberately not persisted. A crash loses the thread but
never loses work, because the work is on disk and in the Time Machine.

## Concurrency

The server is a single asyncio process. Three things could block it and none
of them do:

- **Tools** run through `asyncio.to_thread`.
- **The folder dialog** runs in a separate process (`scripts/pick_folder.py`).
  Tkinter requires the main thread of its process; a thread would deadlock and
  a coroutine would freeze the loop for as long as the dialog is open.
- **Shell commands** stream through a queue fed by a reader thread, so a
  command that produces output for a minute yields it continuously.

Tool approval is a `Future` per tool call id, resolved when the browser sends
`{action: "tool_approval", id, approved}` and cancelled on stop or reset. It
times out after five minutes and denies by default.
