# The agent

`agent/loop.py` is under three hundred lines and there is no keyword matching,
no intent classifier and no hand-written dispatch table anywhere in it. The
loop sends the conversation, streams back text and tool calls, runs the tools,
appends the results and repeats until the model stops asking for tools or the
turn budget runs out. The model decides; this code makes sure it can act and
reports honestly what happened.

## The loop

```
for turn in 1..max_turns:
    stream from provider
      ├── text      → forwarded to the client
      ├── thinking  → forwarded, rendered collapsed
      └── tool_call → collected
    if no tool calls: done
    for each call:
        emit tool_start
        if approval required: emit tool_approval_prompt, await the answer
        run on a worker thread (snapshotting first if it mutates)
        emit tool_end, append the result to the conversation
```

Events emitted: `turn_start`, `thinking`, `text`, `tool_start`,
`tool_approval_prompt`, `tool_end`, `usage`, `error`, `done`.

Tool results are truncated at `MAX_TOOL_RESULT_CHARS` (12,000) before they go
back into the conversation. A `run_command` that prints a hundred thousand lines would
otherwise eat the whole context window and the model would forget what it was
doing.

### Empty replies

Reasoning models sometimes close a turn having spent the whole budget in
thinking tokens, emitting no text and no tool call. That leaves the user
staring at an empty bubble with no indication anything happened. The loop asks
once, without tools, for the final answer; if that also comes back empty it
substitutes a fixed reply rather than rendering nothing.

## Tools

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

`edit_file` refusing an ambiguous match is the important one. A model that
asked to replace `return None` in a file with nine of them has not told you
which; guessing produces an edit that looks applied and is wrong somewhere
else. Refusing costs one more turn and cannot corrupt the file.

Every mutating tool is bracketed by a Time Machine snapshot inside
`Toolkit.execute`, so the model does not have to remember to make its work
reversible. A failure to snapshot does not block the edit - losing the ability
to undo is better than losing the ability to work.

## Approval

`auto_approve` defaults to **off**. When a mutating tool comes up, the loop
emits `tool_approval_prompt` and awaits an `approval_callback`. The websocket
handler answers it with a `Future` keyed by tool call id, resolved when the
browser sends:

```json
{"action": "tool_approval", "id": "<tool call id>", "approved": true}
```

Denial, a five-minute timeout, a `stop`, a `reset` and a missing callback all
resolve the same way: the tool is not run, and the model is told so in the
tool result rather than being left to wonder.

The approval card in the interface shows the first string argument of the call
- usually the path - because "allow `write_file`?" is not a question anyone
can answer.

## Two modes

| | Workspace open | No workspace |
| :--- | :--- | :--- |
| Toolkit | Full | None |
| Tool specs sent to the model | Yes | No |
| System prompt | Tool discipline, workspace root, file tree | General assistant brief |
| Indexer, memory, Time Machine | Active | Skipped |

A system prompt that describes tools the model does not have produces tool
calls that cannot be served, so the workspace-free prompt drops the tool
section entirely rather than merely disabling it.

## The system prompt

`agent/prompts.py` assembles it once per session, in this order:

1. **Identity.** Who the model is, and the rule that its provenance is only
   mentioned when the user asks about it. The negative rule is load-bearing: a
   model fine-tuned on an identity statement will otherwise recite it as a
   preamble to every unrelated answer.
2. **Response style.** Answer the question, no preamble, no restatement of the
   request, prefer a list or a code block over a paragraph, language tags on
   every code block, no emoji, say what you are unsure of in one line instead
   of hedging through the whole answer. Un-steered API models pad; this is
   what stops that.
3. **Tool discipline**, workspace root, platform and file tree - workspace
   mode only.
4. **Skills** - the prompt of every enabled skill (see below).
5. **Project memory** - facts the `remember` tool has written.

## Skills

A skill is a named block of standing directives that is appended to the system
prompt while it is enabled. Four ship enabled by default - a UI/UX brief, a
security checklist, an architecture stance and a CLI workflow guide - and
users can add their own from the Skills panel or `POST /api/skills`.

They are stored in `.shree/skills.json`. On load, built-ins missing from the
file are merged back in by id, so a file written by an older version does not
silently lose the ones added since. Built-ins can be disabled but not deleted.

The system prompt is assembled once per session, so a skill toggled mid-thread
takes effect after a reset rather than on the next message.
