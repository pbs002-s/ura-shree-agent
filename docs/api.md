# HTTP and websocket API

The server binds to loopback by default and holds no cross-site cookies. Every
route is under `/api`, the two streams are websockets, and everything else is
the built frontend served as static files.

Interactive documentation is at `/docs` while the server is running.

## Authentication and scope

Every route except the ones listed below is refused without credentials. That
is enforced by middleware rather than per route, so a route added later is
protected whether or not its author remembered to say so.

Public: `/api/health`, `/api/status`, everything under `/api/auth/`, `/docs`,
`/openapi.json`, and the static frontend.

| Header | Meaning |
| :--- | :--- |
| `Authorization: Bearer <token>` | The access token. Required in cloud mode |
| `X-Project-Id: <id>` | Which project the call is about. Required in cloud mode, optional locally |
| `X-Request-Id: <id>` | Optional. Reused as the correlation id, and echoed back on every response |

WebSockets cannot set headers on the handshake, so they take `?token=` and
`?project=` instead. A socket is authenticated *before* it is accepted and
closed with code 1008 otherwise.

In local mode with no `SHREE_API_KEY` set, every request resolves to a single
synthetic administrator and none of the above is needed.

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Liveness for a load balancer. Cheap, unauthenticated, no database |
| `GET` | `/api/auth/providers` | Which sign-in methods this deployment has configured |
| `POST` | `/api/auth/register` | Create an account. The first one becomes the administrator |
| `POST` | `/api/auth/login` | Exchange a password for an access and refresh token |
| `POST` | `/api/auth/refresh` | Trade a refresh token for a fresh access token |
| `GET` | `/api/auth/me` | The caller, their role, and the server's mode |
| `GET` | `/api/auth/oauth/{provider}/start` | Begin a GitHub or Google login |
| `GET` | `/api/auth/oauth/{provider}/callback` | Complete it; redirects with the token in the URL fragment |

A missing account and a wrong password return the same message, and the hash is
verified either way, so neither the response text nor its timing says whether
an address is registered.

## Projects

A project is a tenant's workspace: one persistent volume, one execution
sandbox, one timeline. Access is by membership, and a caller who is not a
member gets 404 rather than 403 - confirming that an id exists is itself a
leak.

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/projects` | Every project the caller can open, with their role in each |
| `POST` | `/api/projects` | Create one. The caller becomes its owner |
| `POST` | `/api/projects/{id}/members` | Add or re-role a member. Owner only |
| `DELETE` | `/api/projects/{id}/members/{user_id}` | Remove a member and close their live session |

Roles are ordered: `viewer` reads, `member` writes, `owner` manages membership.

## Administration

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/admin/sessions` | Every live session, with its sandbox limits and idle time |
| `DELETE` | `/api/admin/sessions/{user_id}/{project_id}` | Destroy one session and its sandbox |
| `GET` | `/api/admin/audit` | The tool invocation trail, newest first, filterable by user, project or tool |
| `GET` | `/api/admin/audit/verify` | Walk the audit hash chain and name the first broken link |
| `GET` | `/api/jobs/{task_id}` | Where a queued background job has got to |

Administrator role required for everything under `/api/admin`.

## Rate limits

A caller over budget gets `429` with a `Retry-After` header. Buckets refill
continuously rather than resetting on a boundary, so a window cannot be banked
and spent in one burst.

| Bucket | Applies to | Default |
| :--- | :--- | :--- |
| `api` | general REST | 600/60 |
| `scan` | `/api/providers/scan`, register, login | 10/60 |
| `chat` | agent turns | 60/60 |
| `ws` | WebSocket frames, checked before the frame is acted on | 120/60 |

## Status and settings

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Version, mode, workspace, platform, active model, host and GPU memory, Time Machine head and store size, and the health of the database, job broker, rate limiter and tracing |
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
| `POST` | `/api/workspace/select` | Bind a new root. **Local mode only**: refused with 403 in cloud mode |
| `POST` | `/api/workspace/browse` | Open the platform's folder dialog and bind what is chosen |
| `POST` | `/api/browse-directory` | Open the folder dialog and return the path without binding it |
| `GET` | `/api/workspace/suggestions` | Default workspace, project root, Desktop, Documents, Downloads |

`select` rebinds the single-user root and closes every live session, so the
next request rebuilds against the new folder. A path that does not exist is
created; a path that exists and is not a directory is a 400.

It is refused outright in cloud mode. An endpoint that takes an absolute host
path and starts serving its contents is a filesystem read primitive for anyone
who can call it; multi-tenant deployments select a project instead, and a
project only ever names a directory under the workspaces root.

Both browse endpoints run the dialog in a subprocess (`scripts/pick_folder.py`)
and are cancellable - a cancelled dialog returns `{"ok": false,
"cancelled": true}` and changes nothing. They draw a window on the machine
running the server, which on a server is either nothing or a request that hangs
for two minutes holding a worker, so they are disabled unless
`SHREE_ALLOW_HOST_PICKER=1` is set on a local install.

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
