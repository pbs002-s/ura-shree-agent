"""
Relational state: who exists, what they own, and what was done.

Everything the single-user tool kept in process memory or in a JSON file next
to the workspace has to outlive a process here, because there is more than one
process. Users, projects and membership, the chat history behind a session,
the timeline index, and the audit trail all live in one database - Postgres in
production, SQLite for a local run so nothing extra has to be installed to try
the code.

The audit table is append-only by construction. Each row carries the hash of
the row before it, so a deleted or edited entry breaks the chain and
`verify_audit_chain` says where. That is not tamper-proofing - anyone with
write access to the database can recompute the chain - it is tamper-evidence,
which is what an audit log is actually for.

SQLModel and the driver are optional imports: a local install that never sets
SHREE_DATABASE_URL should not need Postgres on the machine.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from server.config import config

try:
    from sqlalchemy import text as _sql_text
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Field, Session, SQLModel, create_engine, select

    SQLMODEL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    SQLMODEL_AVAILABLE = False

    class SQLModel:  # type: ignore[no-redef]
        """Placeholder so module import does not fail without the cloud extras."""

    def Field(*_args, **_kwargs):  # type: ignore[no-redef]
        return None

    Session = object  # type: ignore[assignment,misc]
    select = None  # type: ignore[assignment]
    create_engine = None  # type: ignore[assignment]
    StaticPool = None  # type: ignore[assignment]
    _sql_text = None  # type: ignore[assignment]


CLOUD_EXTRAS_HINT = (
    "This needs the cloud extras. Install them with: pip install -r requirements-cloud.txt"
)


def _uid() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


# -- roles --------------------------------------------------------------------

# Account-level. `admin` can see every project and the full audit log;
# `user` can only ever reach projects it is a member of.
ROLE_ADMIN = "admin"
ROLE_USER = "user"
ACCOUNT_ROLES = (ROLE_ADMIN, ROLE_USER)

# Project-level, checked on every workspace-scoped route.
PROJECT_OWNER = "owner"
PROJECT_MEMBER = "member"
PROJECT_VIEWER = "viewer"
PROJECT_ROLES = (PROJECT_OWNER, PROJECT_MEMBER, PROJECT_VIEWER)

# Ordered weakest to strongest, so a required role is a numeric comparison
# rather than a set of special cases at every call site.
_PROJECT_RANK = {PROJECT_VIEWER: 0, PROJECT_MEMBER: 1, PROJECT_OWNER: 2}


def project_role_allows(held: str, required: str) -> bool:
    return _PROJECT_RANK.get(held, -1) >= _PROJECT_RANK.get(required, 99)


# -- tables -------------------------------------------------------------------

if SQLMODEL_AVAILABLE:

    class User(SQLModel, table=True):
        __tablename__ = "users"

        id: str = Field(default_factory=_uid, primary_key=True)
        email: str = Field(index=True, unique=True)
        display_name: str = ""
        # Empty for OAuth-only accounts, which have no password to verify.
        password_hash: str = ""
        role: str = Field(default=ROLE_USER, index=True)
        oauth_provider: str = ""
        oauth_subject: str = Field(default="", index=True)
        is_active: bool = True
        created_at: float = Field(default_factory=_now)
        last_login_at: float = 0.0

    class Project(SQLModel, table=True):
        __tablename__ = "projects"

        id: str = Field(default_factory=_uid, primary_key=True)
        owner_id: str = Field(index=True, foreign_key="users.id")
        name: str
        slug: str = Field(index=True)
        # Relative to the configured workspaces root; never an absolute path
        # from a client, which is how a path traversal gets in.
        volume: str = ""
        created_at: float = Field(default_factory=_now)
        last_active_at: float = Field(default_factory=_now)
        settings_json: str = "{}"

    class ProjectMember(SQLModel, table=True):
        __tablename__ = "project_members"

        id: str = Field(default_factory=_uid, primary_key=True)
        project_id: str = Field(index=True, foreign_key="projects.id")
        user_id: str = Field(index=True, foreign_key="users.id")
        role: str = Field(default=PROJECT_MEMBER)
        created_at: float = Field(default_factory=_now)

    class ChatMessage(SQLModel, table=True):
        __tablename__ = "chat_messages"

        id: str = Field(default_factory=_uid, primary_key=True)
        project_id: str = Field(index=True)
        user_id: str = Field(index=True)
        session_id: str = Field(index=True)
        role: str = ""
        content: str = ""
        tool_calls_json: str = "[]"
        created_at: float = Field(default_factory=_now)

    class SnapshotRecord(SQLModel, table=True):
        """
        Timeline metadata, mirrored out of the workspace Time Machine.

        The content-addressed tree stays with the workspace volume; this table
        is what makes a timeline listable per tenant without mounting it.
        """

        __tablename__ = "snapshots"

        id: str = Field(primary_key=True)
        project_id: str = Field(index=True)
        parent_id: Optional[str] = None
        label: str = ""
        kind: str = "manual"
        file_count: int = 0
        total_bytes: int = 0
        created_at: float = Field(default_factory=_now, index=True)

    class AuditEntry(SQLModel, table=True):
        """One tool invocation. Written once, never updated."""

        __tablename__ = "audit_log"

        id: str = Field(default_factory=_uid, primary_key=True)
        sequence: Optional[int] = Field(default=None, primary_key=False, index=True)
        created_at: float = Field(default_factory=_now, index=True)
        request_id: str = ""
        session_id: str = Field(default="", index=True)
        user_id: str = Field(default="", index=True)
        project_id: str = Field(default="", index=True)
        tool_name: str = Field(default="", index=True)
        arguments_json: str = "{}"
        approved_by: str = ""
        status: str = ""
        exit_code: Optional[int] = None
        duration_ms: int = 0
        prev_hash: str = ""
        entry_hash: str = ""


# -- engine -------------------------------------------------------------------

_engine = None


def get_engine():
    """The process-wide engine, created on first use."""
    global _engine
    if _engine is not None:
        return _engine
    if not SQLMODEL_AVAILABLE:
        raise RuntimeError(CLOUD_EXTRAS_HINT)

    url = config.database_url
    kwargs: Dict[str, Any] = {"echo": False, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        # SQLite would otherwise refuse the connection FastAPI hands to a
        # threadpool worker, and each new pooled connection to :memory: would
        # get its own empty database.
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs.pop("pool_pre_ping")
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
        else:
            from pathlib import Path

            Path(url.split("///", 1)[-1]).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, **kwargs)
    return _engine


def init_db() -> None:
    """Creates any missing tables. Idempotent, safe on every boot."""
    if not SQLMODEL_AVAILABLE:
        raise RuntimeError(CLOUD_EXTRAS_HINT)
    SQLModel.metadata.create_all(get_engine())


def reset_engine() -> None:
    """Drops the cached engine, so a test can point at a different database."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


@contextmanager
def session_scope() -> Iterator[Any]:
    """A transactional session that commits on success and rolls back on error."""
    # Without expire_on_commit=False, every attribute of every row read here is
    # invalidated the moment the block exits, and touching one outside raises
    # DetachedInstanceError. Callers read rows after the block routinely.
    session = Session(get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Any]:
    """FastAPI dependency form of `session_scope`."""
    with session_scope() as session:
        yield session


# -- audit --------------------------------------------------------------------

# Argument keys whose values never belong in a log line, however useful the
# rest of the call is to keep.
SENSITIVE_ARG_KEYS = {
    "api_key", "apikey", "token", "password", "secret", "authorization",
    "access_token", "refresh_token", "private_key", "credential",
}
MAX_AUDIT_ARG_CHARS = 4000


def redact_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Strips credentials and caps size before an argument set is persisted."""
    cleaned: Dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if key.lower() in SENSITIVE_ARG_KEYS:
            cleaned[key] = "[redacted]"
        elif isinstance(value, str) and len(value) > MAX_AUDIT_ARG_CHARS:
            cleaned[key] = value[:MAX_AUDIT_ARG_CHARS] + f"... [{len(value)} chars]"
        elif isinstance(value, dict):
            cleaned[key] = redact_arguments(value)
        else:
            cleaned[key] = value
    return cleaned


def _hash_entry(entry: Dict[str, Any], prev_hash: str) -> str:
    payload = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.sha256(f"{prev_hash}|{payload}".encode("utf-8")).hexdigest()


def record_tool_invocation(
    *,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    status: str,
    approved_by: str = "",
    exit_code: Optional[int] = None,
    duration_ms: int = 0,
    user_id: str = "",
    project_id: str = "",
    session_id: str = "",
    request_id: str = "",
) -> Optional[str]:
    """
    Appends one immutable audit row and returns its hash.

    Auditing must never be the reason a tool call fails, so a database that is
    down degrades to a log line rather than an exception. The log carries the
    same fields, which is what a later reconciliation needs.
    """
    entry = {
        "created_at": _now(),
        "request_id": request_id,
        "session_id": session_id,
        "user_id": user_id,
        "project_id": project_id,
        "tool_name": tool_name,
        "arguments": redact_arguments(arguments or {}),
        "approved_by": approved_by,
        "status": status,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
    }

    if not SQLMODEL_AVAILABLE or not config.cloud:
        from server.observability import get_logger

        get_logger("audit").info("tool_invocation", **entry)
        return None

    try:
        with session_scope() as session:
            previous = session.exec(
                select(AuditEntry).order_by(AuditEntry.created_at.desc()).limit(1)
            ).first()
            prev_hash = previous.entry_hash if previous else ""
            row = AuditEntry(
                created_at=entry["created_at"],
                request_id=request_id,
                session_id=session_id,
                user_id=user_id,
                project_id=project_id,
                tool_name=tool_name,
                arguments_json=json.dumps(entry["arguments"], default=str),
                approved_by=approved_by,
                status=status,
                exit_code=exit_code,
                duration_ms=duration_ms,
                prev_hash=prev_hash,
                entry_hash=_hash_entry(entry, prev_hash),
            )
            session.add(row)
            return row.entry_hash
    except Exception as err:  # pragma: no cover - only on a database outage
        from server.observability import get_logger

        get_logger("audit").error("audit_write_failed", error=str(err), **entry)
        return None


def verify_audit_chain(limit: int = 10_000) -> Dict[str, Any]:
    """
    Walks the chain oldest to newest and reports the first break.

    A break means a row was removed or edited after the fact, which is the one
    thing an audit log exists to make visible.
    """
    if not SQLMODEL_AVAILABLE:
        raise RuntimeError(CLOUD_EXTRAS_HINT)

    with session_scope() as session:
        rows: List[AuditEntry] = list(
            session.exec(select(AuditEntry).order_by(AuditEntry.created_at).limit(limit))
        )

    prev_hash = ""
    for index, row in enumerate(rows):
        entry = {
            "created_at": row.created_at,
            "request_id": row.request_id,
            "session_id": row.session_id,
            "user_id": row.user_id,
            "project_id": row.project_id,
            "tool_name": row.tool_name,
            "arguments": json.loads(row.arguments_json or "{}"),
            "approved_by": row.approved_by,
            "status": row.status,
            "exit_code": row.exit_code,
            "duration_ms": row.duration_ms,
        }
        if row.prev_hash != prev_hash or row.entry_hash != _hash_entry(entry, prev_hash):
            return {"ok": False, "checked": index, "broken_at": row.id, "total": len(rows)}
        prev_hash = row.entry_hash

    return {"ok": True, "checked": len(rows), "broken_at": None, "total": len(rows)}


def healthcheck() -> Dict[str, Any]:
    """Whether the database is reachable, for the status endpoint."""
    if not SQLMODEL_AVAILABLE:
        return {"available": False, "reason": "sqlmodel is not installed"}
    try:
        with Session(get_engine()) as session:
            session.exec(_sql_text("SELECT 1"))
        return {"available": True, "url": config.database_url.split("@")[-1]}
    except Exception as err:
        return {"available": False, "reason": str(err)}
