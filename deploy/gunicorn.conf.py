"""
Gunicorn configuration for the API.

Gunicorn supervises; Uvicorn serves. The worker class is Uvicorn's, so each
worker is a real asyncio server that can hold WebSockets, and Gunicorn's job is
restarting one that dies and rolling them during a deploy.

Worker count is deliberately modest. The usual `2 * cores + 1` assumes
short CPU-bound requests; this server holds long-lived WebSockets and each
worker keeps its own live sessions, so more workers mean more idle sandboxes
rather than more throughput. Scale out with replicas, not with workers.
"""

from __future__ import annotations

import multiprocessing
import os

bind = os.environ.get("SHREE_BIND", "0.0.0.0:8000")
workers = int(os.environ.get("SHREE_WORKERS", min(4, multiprocessing.cpu_count())))
worker_class = "uvicorn.workers.UvicornWorker"

# An agent turn can legitimately run for minutes. The default 30s timeout would
# kill a worker mid-turn and take every other WebSocket on it down too.
timeout = int(os.environ.get("SHREE_WORKER_TIMEOUT", 1800))
graceful_timeout = 60
keepalive = 75

# The proxy in front terminates TLS and sets these; without the allow-list
# Uvicorn ignores them and every client looks like it came from the proxy.
forwarded_allow_ips = os.environ.get("SHREE_FORWARDED_ALLOW_IPS", "*")
proxy_protocol = False

# JSON access logs are emitted by the application's own middleware, with
# correlation ids attached. Gunicorn's would be a second, less useful stream.
accesslog = None
errorlog = "-"
loglevel = os.environ.get("SHREE_LOG_LEVEL", "info").lower()

# Restarting a worker periodically bounds the damage from a slow leak in a
# long-lived process. The jitter stops every worker recycling at once.
max_requests = int(os.environ.get("SHREE_MAX_REQUESTS", 0))
max_requests_jitter = max_requests // 10 if max_requests else 0

preload_app = False  # each worker loads its own model; sharing CUDA across a fork does not work
