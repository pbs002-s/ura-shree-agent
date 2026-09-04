# HTTP and websocket API

The server binds to loopback by default and holds no cross-site cookies. Every
route is under `/api`, the two streams are websockets, and everything else is
the built frontend served as static files.

Interactive documentation is at `/docs` while the server is running.

## Status and settings

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Version, workspace, platform, active model, host and GPU memory, Time Machine head and store size |
| `GET` | `/api/settings` | Full settings, with keys masked |
| `POST` | `/api/settings` | Patch settings |

## Providers

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/providers` | The catalogue, with which providers hold a key |
| `POST` | `/api/providers/key` | Store a key for one provider |
| `DELETE` | `/api/providers/{provider_id}` | Forget a key |
| `POST` | `/api/providers/scan` | Ask the provider for the models this key can reach |
| `POST` | `/api/providers/select` | Set the active provider and model |

`scan` calls the provider's own model listing rather than returning a
hard-coded list, so what comes back is what the account can actually use.
Keys are returned only as a masked preview, never in full.

## Workspace

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/workspace/current` | The open workspace, or `null` |
| `POST` | `/api/workspace/select` | Bind a new root; an empty path unbinds and enters general chat mode |
| `POST` | `/api/workspace/browse` | Open the platform's folder dialog and bind what is chosen |
| `POST` | `/api/browse-directory` | Open the folder dialog and return the path without binding it |
| `GET` | `/api/workspace/suggestions` | Default workspace, project root, Desktop, Documents, Downloads |

`select` rebuilds the filesystem tool, git tool, shell manager, indexer and
Time Machine against the new root and clears the cached agents. A path that
does not exist is created; a path that exists and is not a directory is a 400.

Both browse endpoints run the dialog in a subprocess (`scripts/pick_folder.py`)
and are cancellable - a cancelled dialog returns `{"ok": false,
"cancelled": true}` and changes nothing.

## Files

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/tree` | The workspace as a nested tree, capped at `max_entries` |
| `GET` | `/api/file?path=` | Read a file |
| `POST` / `PUT` | `/api/file` | Write a file; snapshotted first |
| `DELETE` | `/api/file?path=` | Delete a file or directory; snapshotted first |
| `POST` | `/api/upload` | Base64 upload into a target directory |

With no workspace open, `/api/tree` returns `{"tree": null}` and the rest
return 400. Uploads reject executables, path traversal, absolute paths, drive
letters and anything over 8 MB. An upload endpoint that accepts `.exe` is a
foothold, not a feature.

## Skills

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/skills` | Every skill, built-in and custom |
| `POST` | `/api/skills` | Add a custom skill (`name`, `description`, `prompt`) |
| `PATCH` | `/api/skills/{skill_id}` | Toggle; omitting `enabled` flips it |
| `DELETE` | `/api/skills/{skill_id}` | Delete a custom skill; built-ins are refused |

## Codebase and memory

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/memory` | Facts the `remember` tool has stored |
| `POST` | `/api/index` | Rescan and return the symbol index and tree summary |
| `GET` | `/api/git/status` | Working tree status and diff |

## Time Machine

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/timemachine` | The timeline, newest first |
| `POST` | `/api/timemachine/snapshot` | An explicit labelled point |
| `GET` | `/api/timemachine/diff?from_id=&to_id=` | Diff any two points, not only consecutive ones |
| `GET` | `/api/timemachine/file?snapshot_id=&path=` | Read a file as it was, without restoring |
| `POST` | `/api/timemachine/restore` | Restore a point; `dry_run` reports what would change |
| `POST` | `/api/timemachine/prune?keep=` | Drop old snapshots and unreferenced blobs |

## Terminal

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/terminal/sessions` | Live shell sessions |
| `POST` | `/api/terminal` | Create one |
| `DELETE` | `/api/terminal/{session_id}` | Close one |

### `WS /ws/terminal`

Client sends `{"action": "run", "command": "..."}`, and also `restart`,
`info` and `ping`. The server replies `ready`, `started`, a stream of output
events, then `exit` with `code` and `success`. Refused commands come back as
`exit` with code `-1` before anything runs.

## `WS /ws/agent`

The main stream. Connect with `?session=<id>` to keep separate conversations.

**Client → server**

| Action | Fields |
| :--- | :--- |
| `chat` (the default) | `message`, and optional `provider`, `model`, `temperature`, `max_tokens`, `use_tools`, `auto_approve`, `attachments`, `fresh` |
| `tool_approval` | `id`, `approved` |
| `stop` | – |
| `reset` | – |
| `ping` | – |

**Server → client**

`turn_start`, `thinking`, `text`, `tool_start`, `tool_approval_prompt`,
`tool_end`, `usage`, `error`, `done`, plus `stopped`, `reset` and `pong`.

A `chat` sent while a run is in flight is refused with an `error` rather than
queued - two concurrent runs on one conversation would interleave tool calls
against the same files. `stop` cancels the run and every pending approval.

Anything the client omits falls back to the stored active settings, so a
minimal client can send `{"message": "..."}` and nothing else.
