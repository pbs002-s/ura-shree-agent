"""
One workspace per tenant, keyed by who is asking and what they are working on.

The server used to hold exactly one agent, one shell manager and one Time
Machine as module-level singletons. That is correct for a tool serving a single
person and catastrophic for anything else: two users share a conversation, one
person's `cd` moves another person's terminal, and a restore rolls back a
workspace nobody asked about.

Everything is now keyed by `(user_id, project_id)`. A `WorkspaceSession` owns
that pair's execution driver, agent, timeline and index, and nothing is
reachable without going through the authorisation check that produced the key.

Sessions are created on demand and reaped when idle, because a container per
session is not free. The reaper is a plain background task rather than a
scheduler: it wakes on an interval, closes what has been quiet too long, and
the next request rebuilds whatever it needs. State that has to survive that -
history, timeline metadata - is in the database, not in the session object.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.agent import CodingAgent
from agent.indexer import CodebaseIndexer
from agent.memory import ProjectMemory
from agent.timemachine import TimeMachine
from server.config import config
from server.observability import get_logger
from server.storage import build_object_store
from tools.drivers import ExecutionDriver, build_driver
from tools.git import GitTool

log = get_logger("sessions")

SessionKey = Tuple[str, str]

# Volume names are built from ids, never from anything a client types. This is
# the belt to the braces: even if an id ever reached here unvalidated, it
# cannot contain a separator or a dot segment.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def safe_segment(value: str, label: str) -> str:
    """Validates one path segment, or refuses."""
    cleaned = (value or "").strip()
    if not _SAFE_SEGMENT.match(cleaned) or cleaned in (".", ".."):
        raise ValueError(f"Invalid {label}: {value!r}")
    return cleaned


@dataclass
class WorkspaceSession:
    """Everything one `(user_id, project_id)` pair needs to do work."""

    user_id: str
    project_id: str
    root: Path
    driver: ExecutionDriver
    agent: CodingAgent
    time_machine: TimeMachine
    indexer: CodebaseIndexer
    git: GitTool
    memory: ProjectMemory
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def key(self) -> SessionKey:
        return (self.user_id, self.project_id)

    @property
    def session_id(self) -> str:
        """The stable id the shell, agent and audit rows all share."""
        return f"{self.user_id}:{self.project_id}"

    def touch(self) -> None:
        self.last_used = time.time()
        self.driver.touch()

    @property
    def idle_seconds(self) -> float:
        # The driver knows about work the session object never sees, such as a
        # long command still streaming, so it wins.
        return min(time.time() - self.last_used, self.driver.idle_seconds)

    def info(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "project_id": self.project_id,
            "workspace": str(self.root),
            "created_at": self.created_at,
            "idle_sec": round(self.idle_seconds, 1),
            "execution": self.driver.info(),
        }

    def close(self) -> None:
        for closer in (self.driver.close, self.agent.close, self.time_machine.close):
            try:
                closer()
            except Exception:
                # Teardown is best effort; one stuck resource must not strand
                # the others or wedge the reaper.
                pass


class WorkspaceSessionManager:
    """
    The registry of live sessions.

    Everything is under one lock. Session creation is rare - once per user per
    project per idle window - and the cost of getting concurrent creation
    subtly wrong is two containers fighting over one volume.
    """

    def __init__(
        self,
        workspaces_root: Optional[Path] = None,
        driver_kind: Optional[str] = None,
        idle_timeout: Optional[int] = None,
    ):
        self.workspaces_root = Path(workspaces_root or config.workspaces_root).resolve()
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self.driver_kind = driver_kind or config.driver_kind
        self.idle_timeout = idle_timeout if idle_timeout is not None else config.session_idle_timeout
        # Single-user mode still points at one folder the developer chose, so
        # every key resolves to the same directory. Never set in cloud mode -
        # it would collapse every tenant into one workspace.
        self.local_root: Optional[Path] = None
        self._sessions: Dict[SessionKey, WorkspaceSession] = {}
        self._lock = threading.RLock()
        self._reaper: Optional[asyncio.Task] = None

    # -- volumes ------------------------------------------------------------

    def volume_for(self, user_id: str, project_id: str) -> Path:
        """
        The persistent directory backing one project.

        Built only from validated ids and resolved against the root, so no
        combination of inputs can address a path outside it.
        """
        if self.local_root is not None and not config.cloud:
            return self.local_root
        user = safe_segment(user_id, "user id")
        project = safe_segment(project_id, "project id")
        volume = (self.workspaces_root / user / project).resolve()
        if not volume.is_relative_to(self.workspaces_root):
            raise ValueError("Workspace path escapes the workspaces root.")
        return volume

    # -- lifecycle ----------------------------------------------------------

    def _build(self, user_id: str, project_id: str) -> WorkspaceSession:
        root = self.volume_for(user_id, project_id)
        root.mkdir(parents=True, exist_ok=True)
        session_id = f"{user_id}:{project_id}"

        driver_options: Dict[str, Any] = {}
        if self.driver_kind == "container":
            driver_options = {
                "image": config.sandbox_image,
                "cpus": config.sandbox_cpus,
                "memory_mb": config.sandbox_memory_mb,
                "pids_limit": config.sandbox_pids_limit,
                "network": config.sandbox_network,
                "labels": {"shree.user": user_id, "shree.project": project_id},
            }
        driver = build_driver(
            self.driver_kind, str(root), session_id=session_id, **driver_options
        )

        # Blobs may live in object storage while the tree index stays on the
        # workspace volume, so a replica that has never seen this workspace can
        # still read every snapshot's content.
        time_machine = TimeMachine(
            str(root),
            object_store=build_object_store(
                config.object_store_url,
                fallback_dir=str(root / ".timemachine" / "objects"),
                endpoint_url=config.s3_endpoint_url or None,
                region=config.s3_region,
            )
            if config.object_store_url
            else None,
        )

        memory = ProjectMemory(str(root / ".shree" / "memory.db"))
        agent = CodingAgent(
            workspace_root=str(root),
            memory_db=str(root / ".shree" / "memory.db"),
            time_machine=time_machine,
            session_id=session_id,
            driver=driver,
            audit_context={
                "user_id": user_id,
                "project_id": project_id,
                "session_id": session_id,
            },
        )

        log.info(
            "session_created",
            user_id=user_id,
            project_id=project_id,
            driver=self.driver_kind,
            workspace=str(root),
        )
        return WorkspaceSession(
            user_id=user_id,
            project_id=project_id,
            root=root,
            driver=driver,
            agent=agent,
            time_machine=time_machine,
            indexer=CodebaseIndexer(str(root)),
            git=GitTool(str(root)),
            memory=memory,
        )

    def get(self, user_id: str, project_id: str) -> WorkspaceSession:
        """The session for this pair, created on first use."""
        key = (user_id, project_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = self._build(user_id, project_id)
                self._sessions[key] = session
            session.touch()
            return session

    def peek(self, user_id: str, project_id: str) -> Optional[WorkspaceSession]:
        """The session if it is already live, without creating one."""
        with self._lock:
            return self._sessions.get((user_id, project_id))

    def close(self, user_id: str, project_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop((user_id, project_id), None)
        if session is None:
            return False
        session.close()
        log.info("session_closed", user_id=user_id, project_id=project_id, reason="explicit")
        return True

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [session.info() for session in self._sessions.values()]

    # -- reaping ------------------------------------------------------------

    def reap_idle(self, timeout: Optional[float] = None) -> int:
        """Closes every session quiet for longer than the timeout."""
        limit = self.idle_timeout if timeout is None else timeout
        if limit <= 0:
            return 0
        with self._lock:
            stale = [key for key, s in self._sessions.items() if s.idle_seconds > limit]
            sessions = [self._sessions.pop(key) for key in stale]
        for session in sessions:
            session.close()
            log.info(
                "session_closed",
                user_id=session.user_id,
                project_id=session.project_id,
                reason="idle",
                idle_sec=round(session.idle_seconds, 1),
            )
        return len(sessions)

    async def run_reaper(self, interval: Optional[int] = None) -> None:
        """
        The background loop that keeps idle sandboxes from accumulating.

        Reaping runs in a worker thread because closing a container is a
        blocking subprocess call, and stalling the event loop to tidy up would
        pause every live WebSocket.
        """
        period = interval or config.reaper_interval
        while True:
            try:
                await asyncio.sleep(period)
                closed = await asyncio.to_thread(self.reap_idle)
                if closed:
                    log.info("reaper_pass", closed=closed, live=len(self._sessions))
            except asyncio.CancelledError:
                raise
            except Exception as err:  # pragma: no cover - defensive
                log.error("reaper_failed", error=str(err))

    def start_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self.run_reaper())

    async def stop_reaper(self) -> None:
        if self._reaper is not None and not self._reaper.done():
            self._reaper.cancel()
            try:
                await self._reaper
            except asyncio.CancelledError:
                pass
        self._reaper = None


def mirror_snapshot(project_id: str, snapshot: Dict[str, Any]) -> None:
    """
    Copies snapshot metadata into the shared database.

    The Time Machine's own SQLite index is authoritative and travels with the
    workspace volume; this mirror is what lets the API list a tenant's timeline
    without mounting it, and what survives the volume being recycled.

    ponytail: a mirror can drift if a write lands in one store and not the
    other. Acceptable while the mirror is only ever read for listings; if it
    ever drives a restore, move the index itself into Postgres.
    """
    if not config.cloud:
        return
    try:
        from server.db import SnapshotRecord, session_scope

        with session_scope() as session:
            if session.get(SnapshotRecord, snapshot["id"]) is not None:
                return
            session.add(
                SnapshotRecord(
                    id=snapshot["id"],
                    project_id=project_id,
                    parent_id=snapshot.get("parent_id"),
                    label=snapshot.get("label", ""),
                    kind=snapshot.get("kind", "manual"),
                    file_count=int(snapshot.get("file_count", 0)),
                    total_bytes=int(snapshot.get("total_bytes", 0)),
                    created_at=float(snapshot.get("created_at", time.time())),
                )
            )
    except Exception as err:
        log.warning("snapshot_mirror_failed", project_id=project_id, error=str(err))


def persist_chat_message(
    project_id: str, user_id: str, session_id: str, role: str, content: str, tool_calls=None
) -> None:
    """Appends one turn to the durable chat history."""
    if not config.cloud or not content:
        return
    try:
        import json as _json

        from server.db import ChatMessage, session_scope

        with session_scope() as session:
            session.add(
                ChatMessage(
                    project_id=project_id,
                    user_id=user_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                    tool_calls_json=_json.dumps(tool_calls or [], default=str),
                )
            )
    except Exception as err:
        log.warning("chat_persist_failed", project_id=project_id, error=str(err))
