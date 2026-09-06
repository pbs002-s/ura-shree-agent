# Deploying URA-Shree

URA-Shree runs in one of two modes. Local mode is the tool as it has always
been: one developer, one folder, no login, commands on the host. Cloud mode is
the same application with the parts a hosted service needs — accounts,
projects, sandboxes, limits and an audit trail — switched on.

```
SHREE_MODE=local   # default: single user, host execution, no database
SHREE_MODE=cloud   # authenticated, multi-tenant, sandboxed
```

Nothing about local mode changed. `python scripts/launch.py` still works with
`requirements.txt` alone; none of the cloud dependencies need to be installed.

---

## Architecture

```
                    ┌──────────┐
   browser ── TLS ──│  nginx   │── HTTP/1.1 ──┐
                    └──────────┘              │  (HTTP/2 to the browser;
                                              │   HTTP/1.1 upstream, because
                                              │   WebSocket upgrade needs it)
                                     ┌────────▼────────┐
                                     │  API (gunicorn  │
                                     │  + uvicorn      │
                                     │   workers)      │
                                     └───┬────┬────┬───┘
                                         │    │    │
        ┌────────────────────────────────┘    │    └──────────────────┐
        │                                     │                       │
 ┌──────▼──────┐                    ┌─────────▼────────┐    ┌─────────▼────────┐
 │  Postgres   │                    │  Redis           │    │ Docker (proxied) │
 │  users      │                    │  rate limits     │    │ one ephemeral    │
 │  projects   │                    │  Celery broker   │    │ sandbox per      │
 │  chat       │                    └─────────┬────────┘    │ session          │
 │  timeline   │                              │             └──────────────────┘
 │  audit log  │                    ┌─────────▼────────┐
 └─────────────┘                    │  Celery worker   │
                                    │  indexing, prune │
 ┌─────────────┐                    └──────────────────┘
 │  S3 / MinIO │  snapshot blobs, content-addressed
 └─────────────┘
```

### Request path

1. `CorrelationMiddleware` assigns a `request_id` (or reuses the proxy's) and
   binds it, plus `session_id`, `user_id` and `project_id`, into contextvars.
   Every log line emitted anywhere under that request carries them.
2. `require_authentication` rejects anything not on the public list. This is
   deny-by-default: a route added later is protected whether or not its author
   remembered a dependency.
3. The route resolves a `Principal` and a project id, authorises the pair, and
   gets back that tenant's `WorkspaceSession`.
4. The session owns an `ExecutionDriver`, a `CodingAgent`, a `TimeMachine`, an
   index and a git view. There is no shared object between tenants.

---

## The five things that make it multi-tenant

### 1. Sandboxed execution

`tools/drivers.py` puts an interface between a tool call and the machine that
runs it.

| | `LocalDriver` | `ContainerDriver` |
|---|---|---|
| Runs on | the host process | an ephemeral Docker container |
| CPU / memory | unbounded | `--cpus`, `--memory`, `--memory-swap` |
| Processes | unbounded | `--pids-limit` |
| Network | the host's | `--network=none` by default |
| Filesystem | the workspace directory | one bind mount, `--read-only` root |
| Privileges | the server's | `--cap-drop ALL`, `no-new-privileges`, uid 1000 |
| Lifetime | the process | destroyed after 30 minutes idle |

Build the sandbox image before the first agent command:

```bash
docker build -f deploy/sandbox.Dockerfile -t shree-sandbox:latest .
```

The command blocklist in `tools/shell.py` still applies in both drivers. It is
a guard rail against an accidental `rm -rf /`, not a security boundary — any
blocklist can be spelled around. The container is the boundary.

**Sandboxes share the host kernel.** For untrusted code from the public
internet that is not enough on its own: run the sandboxes on dedicated hosts
with a hypervisor-isolated runtime (Firecracker via firecracker-containerd,
gVisor, or a hosted microVM service such as E2B). `ContainerDriver` is the
place to add one — it is the only code that knows how a sandbox is started.

### 2. Sessions keyed by `(user_id, project_id)`

`server/sessions.py`. `WorkspaceSessionManager.get(user_id, project_id)` is the
only way to reach a workspace, and it is only ever called after
`authorize_project`. Volume paths are built from validated id segments and
resolved against the workspaces root, so no combination of inputs addresses a
path outside it.

Idle sessions are reaped by a background task every `SHREE_REAPER_INTERVAL`
seconds. Anything that must outlive a reap — chat history, timeline metadata —
is in Postgres.

### 3. Authentication and authorisation

`server/auth.py`. Password login, refresh tokens, and optional GitHub or Google
OAuth2. The first account created becomes the administrator, so there is no
default password to forget to change.

Two role scopes:

* **Account**: `admin` (audit log, every session) or `user`.
* **Project**: `owner` (membership), `member` (write), `viewer` (read).

REST and both WebSockets go through the same `resolve_principal`. A socket is
authenticated *before* `accept()` and closed with 1008 otherwise, so an
unauthenticated client never holds a channel at all.

The JWT implementation is intentionally small and supports one algorithm. It
never reads `alg` from the token to choose a verifier, which removes the
`alg: none` family of bugs by construction rather than by a check.

### 4. Rate limiting

Token buckets in Redis (`server/ratelimit.py`), refilled continuously so a
caller cannot bank a whole window and spend it at the turn of the minute.
Authenticated callers are keyed by user id, so rotating IP addresses does not
multiply a quota.

| Bucket | Default | Why it is separate |
|---|---|---|
| `SHREE_RATE_API` | 600/60 | general REST |
| `SHREE_RATE_SCAN` | 10/60 | each scan is an outbound call on the user's provider quota |
| `SHREE_RATE_CHAT` | 60/60 | each turn costs tokens |
| `SHREE_RATE_WS` | 120/60 | per-frame, checked before the frame is acted on |

Without Redis it degrades to a per-process bucket: with N workers the effective
limit is N times the configured one. Fine locally, not a substitute in
production.

### 5. Audit trail

Every tool invocation — agent, terminal or REST — writes one row: tool name,
redacted arguments, who approved it, status, exit code, duration, and the
correlation ids. Rows are append-only and each carries the hash of the one
before it.

```
GET /api/admin/audit          # newest first, filterable
GET /api/admin/audit/verify   # walks the chain, names the first broken link
```

That is tamper-*evidence*, not tamper-proofing: anyone with write access to the
database can recompute the chain. It is what an audit log is actually for.
Credentials are stripped before a row is written (`redact_arguments`).

---

## Quick start

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"   # SHREE_JWT_SECRET
# set POSTGRES_PASSWORD and AWS_SECRET_ACCESS_KEY too

docker build -f deploy/sandbox.Dockerfile -t shree-sandbox:latest .
docker compose build
docker compose up -d
```

Put `fullchain.pem` and `privkey.pem` in `deploy/certs/`. Without them nginx
serves plain HTTP on port 80 only.

The first account you register through the UI becomes the administrator.

### Without Docker

```bash
pip install -r requirements.txt -r requirements-cloud.txt
export SHREE_MODE=cloud SHREE_JWT_SECRET=... SHREE_DATABASE_URL=... SHREE_REDIS_URL=...
gunicorn server.api:app --config deploy/gunicorn.conf.py
celery -A server.tasks worker --loglevel=info --concurrency=4
```

---

## The Docker socket

`ContainerDriver` needs a Docker endpoint to create sandboxes. **Do not mount
`/var/run/docker.sock` into the API container.** The socket is root on the host:
anything that can reach it can run `docker run -v /:/host` and own the machine,
and no application-level check prevents that.

`docker-compose.yml` gives the API a filtered socket proxy that allows only the
container calls the driver makes. Better still, run the daemon rootless, or put
sandboxes on their own hosts entirely.

## Observability

Structured JSON logs with `request_id`, `session_id`, `user_id` and
`project_id` on every line. Set `OTEL_EXPORTER_OTLP_ENDPOINT` and OpenTelemetry
tracing turns on, instrumenting request lifecycles, provider scans and agent
turns; without a collector the same spans still emit duration log lines, so
latency is measurable from logs alone.

`GET /api/health` is the load balancer probe: cheap, unauthenticated, no
database. `GET /api/status` reports database, broker, rate limiter and tracing
health.

## Scaling

Scale with replicas, not with workers. Each Gunicorn worker holds its own live
sessions and their sandboxes, so more workers on one host mean more idle
containers rather than more throughput. `SHREE_WORKERS` defaults to 4.

Two things must be shared for replicas to agree with each other:

* `SHREE_JWT_SECRET` — otherwise a token minted by one replica fails on the next.
* `SHREE_OBJECT_STORE_URL` — otherwise each replica has a private, incomplete
  snapshot history.

## Migrating an existing single-user install

1. Set `SHREE_MODE=cloud` and the database URL, and start the server once so
   the tables are created.
2. Register the first account; it becomes the administrator.
3. Create a project and copy the old workspace into
   `$SHREE_WORKSPACES_ROOT/<user_id>/<project_id>/`, including `.timemachine/`
   to keep the history.
4. Provider keys are per user in cloud mode. Re-enter them once in Settings;
   the old `.shree/settings.json` is not read for other accounts, which is the
   point.

`POST /api/workspace/select` is refused in cloud mode. Projects are the
mechanism there, and a project only ever names a directory under the
workspaces root.
