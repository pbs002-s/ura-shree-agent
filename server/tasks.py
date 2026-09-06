"""
Long jobs, off the request path.

Indexing a large repository, running a benchmark or kicking off a fine-tune all
take longer than a request should live. Doing them in a thread inside the API
process is worse than it looks: the work dies with a deploy, it competes with
the event loop for the GIL, and there is no way to see or cancel it once it has
started.

Celery over Redis moves them to a worker fleet that can be scaled and restarted
independently. `enqueue` is the only entry point the API uses, and when no
broker is configured it runs the job inline and says so - which is exactly what
a single-user local run wants, and keeps one code path instead of two.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from server.config import config
from server.observability import get_logger

log = get_logger("tasks")

celery_app = None

if config.redis_url:
    try:
        from celery import Celery

        celery_app = Celery(
            "shree",
            broker=config.redis_url,
            backend=config.redis_url,
        )
        celery_app.conf.update(
            task_serializer="json",
            result_serializer="json",
            accept_content=["json"],
            timezone="UTC",
            enable_utc=True,
            # A job that outlives its usefulness is worse than a failed one:
            # it holds a worker slot and the caller has long since given up.
            task_soft_time_limit=1800,
            task_time_limit=2100,
            task_acks_late=True,
            # With acks_late, a prefetching worker holds messages it will not
            # start for minutes, so a restart delays them all.
            worker_prefetch_multiplier=1,
            result_expires=86400,
        )
    except ImportError:  # pragma: no cover - depends on the install
        log.warning("celery_missing", detail="SHREE_REDIS_URL is set but celery is not installed")


def _task(name: str):
    """Registers a Celery task when there is a broker, and is a no-op when not."""

    def decorator(func):
        if celery_app is None:
            return func
        return celery_app.task(name=name, bind=False)(func)

    return decorator


# -- jobs ---------------------------------------------------------------------

# Jobs take the workspace path rather than a (user, project) pair. The caller
# has already resolved and authorised it; re-deriving it inside the worker
# would mean the worker needs the same routing rules as the API, and would get
# the single-user case wrong, where every key maps to one chosen folder.

@_task("shree.index_workspace")
def index_workspace(workspace: str) -> Dict[str, Any]:
    """Rebuilds the symbol index for one workspace."""
    from agent.indexer import CodebaseIndexer

    started = time.perf_counter()
    indexer = CodebaseIndexer(workspace)
    stats = indexer.scan_and_index()
    return {
        "ok": True,
        "stats": stats,
        "tree_summary": indexer.get_tree_summary(),
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }


@_task("shree.prune_snapshots")
def prune_snapshots(workspace: str, keep: int = 100) -> Dict[str, Any]:
    """Drops old snapshots and the blobs nothing references any more."""
    from agent.timemachine import TimeMachine

    machine = TimeMachine(workspace)
    try:
        return machine.prune(keep=keep)
    finally:
        machine.close()


@_task("shree.run_benchmark")
def run_benchmark(checkpoint: str = "", tokens: int = 128) -> Dict[str, Any]:
    """Measures decode throughput for a checkpoint."""
    from inference.engine import InferenceEngine

    started = time.perf_counter()
    engine = InferenceEngine(
        checkpoint_path=checkpoint or "checkpoints/coding_best.pt",
        tokenizer_path="checkpoints/tokenizer.json",
    )
    engine.generate("def fibonacci(n):", max_new_tokens=tokens)
    return {
        "ok": True,
        "result": engine.describe(),
        "duration_ms": int((time.perf_counter() - started) * 1000),
    }


JOBS = {
    "index_workspace": index_workspace,
    "prune_snapshots": prune_snapshots,
    "run_benchmark": run_benchmark,
}


# -- dispatch -----------------------------------------------------------------

def enqueue(job: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Submits a job, or runs it inline when there is no broker.

    Returns `{"queued": bool, "task_id": str|None, "result": dict|None}`, so a
    caller can render either outcome without asking which mode it is in.
    """
    func = JOBS.get(job)
    if func is None:
        raise ValueError(f"Unknown job '{job}'. Known jobs: {', '.join(sorted(JOBS))}")

    if celery_app is None:
        log.info("job_inline", job=job)
        return {"queued": False, "task_id": None, "result": func(**kwargs)}

    async_result = func.delay(**kwargs)
    log.info("job_queued", job=job, task_id=async_result.id)
    return {"queued": True, "task_id": async_result.id, "result": None}


def job_status(task_id: str) -> Dict[str, Any]:
    """Where a queued job has got to."""
    if celery_app is None:
        return {"task_id": task_id, "state": "UNAVAILABLE", "detail": "No broker configured."}
    result = celery_app.AsyncResult(task_id)
    payload: Dict[str, Any] = {"task_id": task_id, "state": result.state}
    if result.successful():
        payload["result"] = result.result
    elif result.failed():
        # The traceback belongs in the worker log, not in an API response.
        payload["error"] = str(result.result)
    return payload


def broker_health() -> Dict[str, Any]:
    """Whether jobs will actually be picked up, for the status endpoint."""
    if celery_app is None:
        return {"available": False, "mode": "inline"}
    try:
        workers = celery_app.control.ping(timeout=1.0)
        return {"available": True, "mode": "celery", "workers": len(workers or [])}
    except Exception as err:
        return {"available": False, "mode": "celery", "error": str(err)}


def get_celery_app() -> Optional[Any]:
    """The app object, for `celery -A server.tasks worker`."""
    return celery_app


# `celery -A server.tasks` looks for a module attribute named `app`.
app = celery_app
