"""
URA-Shree backend.

REST for state the UI reads once (status, settings, file tree, timeline), and
WebSockets for the two things that stream: the agent turn and the terminal.

Nothing here is a singleton any more. Every request resolves to a principal and
a project, and from that pair to a `WorkspaceSession` that owns its own
execution sandbox, agent, terminal and timeline. Two users cannot see each
other's files, share a shell, or roll back each other's work, because there is
no shared object between them to do it through.

The server runs in one of two modes:

  * local  the way it always worked. One developer, one folder, host
           execution, no login. `SHREE_MODE` unset or "local".
  * cloud  authenticated, database-backed, container-sandboxed, rate limited.
           `SHREE_MODE=cloud`.

Authentication is deny-by-default: the middleware rejects anything that is not
on the public list, so a route added later is protected whether or not its
author remembered to say so.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import re
import secrets
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from providers import ProviderError, build_provider, list_specs, scan_models
from server import db
from server.auth import (
    LOCAL_PRINCIPAL, OAUTH_PROVIDERS, Principal, TokenError, authorize_project,
    consume_oauth_state, create_token, current_principal, decode_token, hash_password,
    issue_token_pair, oauth_authorize_url, oauth_configured, oauth_exchange,
    principal_for_socket, remember_oauth_state, require_admin, resolve_principal,
    verify_password,
)
from server.config import config
from server.observability import (
    CorrelationMiddleware, bind_context, configure_logging, get_logger, setup_tracing, span,
)
from server.ratelimit import SocketLimiter, limiter, rate_limit
from server.sessions import (
    WorkspaceSession, WorkspaceSessionManager, mirror_snapshot, persist_chat_message, safe_segment,
)
from server.settings import Settings
from server.skills import SkillsManager
from server.tasks import broker_health, enqueue, job_status
from tools.drivers import docker_available
from tools.shell import check_command

configure_logging()
log = get_logger("api")

DEFAULT_WORKSPACE = (PROJECT_ROOT / "workspace").resolve()
DEFAULT_WORKSPACE.mkdir(parents=True, exist_ok=True)
_configured_ws = os.environ.get("SHREE_WORKSPACE", "")
WORKSPACE_ROOT: Optional[Path] = Path(_configured_ws).resolve() if _configured_ws else None

skills_mgr = SkillsManager(PROJECT_ROOT / ".shree" / "skills.json")

# Internal directories that belong to Shree's engine, hidden in production view
INTERNAL_FRAMEWORK_DIRS = {
    "agent", "checkpoints", "configs", "data", "datasets", "frontend",
    "inference", "model", "providers", "scripts", "server", "skills",
    "tests", "tokenizer", "tools", "training",
}

# Hidden from the file explorer. ".shree" holds credentials and ".timemachine"
# holds the snapshot object store; neither is content the user browses.
BASE_IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    "dist", "build", ".idea", ".vscode", ".mypy_cache", ".ruff_cache",
    ".shree", ".timemachine", ".cache", ".next", ".turbo", "checkpoints",
}

# Files a browser upload is allowed to place in the workspace. Everything else
# is rejected: an upload endpoint that accepts .exe or .dll is a foothold, not a
# feature, and nothing in this app needs one.
BLOCKED_UPLOAD_SUFFIXES = {
    ".exe", ".dll", ".so", ".dylib", ".msi", ".bat", ".cmd", ".com", ".scr",
    ".ps1", ".vbs", ".jar", ".apk", ".sys", ".drv",
}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_FILES = 500

# Routes reachable without credentials. Everything else is refused by the
# middleware, so forgetting a dependency on a new route fails closed.
PUBLIC_PATH_PREFIXES = ("/assets/", "/api/auth/")
PUBLIC_PATHS = {
    "/", "/index.html", "/landing.html", "/favicon.ico",
    "/docs", "/redoc", "/openapi.json", "/api/status", "/api/health",
}


def safe_relative_path(raw: str) -> Optional[str]:
    """
    Normalises a browser-supplied path, or None when it tries to climb out.
    """
    cleaned = raw.replace("\\", "/").strip()
    if not cleaned or cleaned.startswith("/") or ":" in cleaned.split("/")[0]:
        return None
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    return "/".join(parts) or None


def ignored_dirs_for(root: Path) -> set:
    """Directories the explorer hides for this workspace."""
    ignored = set(BASE_IGNORED_DIRS)
    if root == PROJECT_ROOT and not os.environ.get("SHREE_DEV"):
        ignored |= INTERNAL_FRAMEWORK_DIRS
    return ignored


# -- application --------------------------------------------------------------

app = FastAPI(
    title="Ura-Shree",
    description="Multi-tenant AI coding agent with sandboxed execution.",
    version="3.0.0",
)

session_manager = WorkspaceSessionManager()
if not config.cloud:
    session_manager.local_root = WORKSPACE_ROOT or DEFAULT_WORKSPACE


# Middleware order matters, and Starlette applies it in reverse: the last one
# added is the outermost. These three are registered innermost first, so a
# request arrives through CORS, then correlation, then authentication.
#
# CORS has to be outermost or a 401 from the gate below comes back without the
# headers a browser needs to read it, and a cross-origin frontend sees an
# opaque network error instead of "your session expired".

@app.middleware("http")
async def require_authentication(request: Request, call_next):
    """Deny-by-default gate in front of every non-public route."""
    # A preflight carries no credentials by design; rejecting it would make
    # every cross-origin request fail before the real one is ever sent.
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PATH_PREFIXES):
        return await call_next(request)
    # The compiled frontend is served from the root mount; a request for an
    # asset is not an API call and carries no credentials.
    if not path.startswith("/api/") and not path.startswith("/ws/"):
        return await call_next(request)

    try:
        resolve_principal(request.headers, request.query_params)
    except TokenError as err:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(err)},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


app.add_middleware(CorrelationMiddleware)

_allowed_origins = config.default_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    # Credentials with a wildcard origin is what turns any page on the internet
    # into an authenticated client of this API.
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-Id"],
)


# -- shared, process-wide state -----------------------------------------------
# The local checkpoint is the one thing legitimately shared: it is read-only
# weights, loading it costs seconds and gigabytes, and no tenant data flows
# through it that is not already in that tenant's own request.

_engine = None
_engine_error = ""
_startup_profile = None
_tracing_enabled = False
_settings_cache: Dict[str, Settings] = {}


def settings_for(principal: Principal) -> Settings:
    """
    That principal's settings, including their provider keys.

    Per user, not per process: provider credentials are the user's own billing
    relationship, and one tenant being able to spend another's quota is the
    whole multi-tenancy failure in miniature.

    The file sits one level above the project volumes, so it is never inside
    anything a sandbox mounts - an agent cannot read the key it is spending.

    ponytail: the cache grows one small object per user seen and is never
    evicted. Give it an LRU bound if a deployment's user count makes that
    measurable.
    """
    if not config.cloud:
        path = (WORKSPACE_ROOT / ".shree" / "settings.json") if WORKSPACE_ROOT else (
            PROJECT_ROOT / ".shree" / "settings.json"
        )
        if not path.exists() and (PROJECT_ROOT / ".shree" / "settings.json").exists():
            path = PROJECT_ROOT / ".shree" / "settings.json"
        key = "local"
    else:
        key = safe_segment(principal.id, "user id")
        path = config.workspaces_root / key / ".shree" / "settings.json"

    cached = _settings_cache.get(key)
    if cached is None:
        cached = Settings(str(path))
        _settings_cache[key] = cached
    return cached


def get_engine(force_reload: bool = False):
    """Loads the local checkpoint on first use, so startup stays fast."""
    global _engine, _engine_error
    if _engine is not None and not force_reload:
        return _engine

    local_cfg = settings_for(LOCAL_PRINCIPAL).get("local", {}) or {}
    checkpoint = local_cfg.get("checkpoint") or ""
    if not checkpoint or not os.path.exists(checkpoint):
        for candidate in ("checkpoints/coding_best.pt", "checkpoints/best.pt", "checkpoints/last.pt"):
            if os.path.exists(candidate):
                checkpoint = candidate
                break

    tokenizer_path = "checkpoints/tokenizer.json"
    if not checkpoint or not os.path.exists(tokenizer_path):
        _engine_error = "No local checkpoint and tokenizer found under checkpoints/."
        return None

    try:
        from inference.engine import InferenceEngine

        _engine = InferenceEngine(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer_path,
            device=local_cfg.get("device") or None,
            quantize=bool(local_cfg.get("quantize")),
            compile_model=bool(local_cfg.get("compile")),
        )
        _engine_error = ""
    except Exception as err:
        _engine = None
        _engine_error = f"Could not load {checkpoint}: {err}"
    return _engine


def provider_for(principal: Principal, provider_id: str):
    """Builds a provider client from that principal's stored credentials."""
    user_settings = settings_for(principal)
    return build_provider(
        provider_id,
        api_key=user_settings.api_key(provider_id),
        base_url=user_settings.base_url(provider_id),
        engine_factory=get_engine,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Tune the runtime once at boot, and tear down live sessions on exit."""
    global _startup_profile, _tracing_enabled

    _tracing_enabled = setup_tracing(_app)
    try:
        from inference.runtime import tune_runtime

        _, _, _startup_profile = tune_runtime()
    except Exception as err:
        log.warning("runtime_tuning_skipped", error=str(err))

    if config.cloud:
        db.init_db()
        log.info(
            "server_ready",
            mode=config.mode,
            driver=config.driver_kind,
            docker=docker_available(),
            rate_limit_backend=limiter.backend,
        )
    session_manager.start_reaper()
    try:
        yield
    finally:
        await session_manager.stop_reaper()
        await asyncio.to_thread(session_manager.close_all)
        await limiter.close()


app.router.lifespan_context = lifespan


# -- schemas ------------------------------------------------------------------

class FileWriteRequest(BaseModel):
    path: str
    content: str


class ProviderKeyRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class ScanRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    save: bool = True


class SelectModelRequest(BaseModel):
    provider: str
    model: str


class SettingsRequest(BaseModel):
    section: str
    values: Dict[str, Any]


class UploadItem(BaseModel):
    path: str = Field(..., description="Destination path relative to the workspace root")
    content_base64: str


class UploadRequest(BaseModel):
    target_dir: str = "uploads"
    files: List[UploadItem]


class SnapshotRequest(BaseModel):
    label: str = "Manual snapshot"


class RestoreRequest(BaseModel):
    snapshot_id: str
    dry_run: bool = False


class CommandRequest(BaseModel):
    command: str
    session_id: str = "default"
    timeout: int = 120


class WorkspaceSelectRequest(BaseModel):
    path: str


class SkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    prompt: str


class SkillToggleRequest(BaseModel):
    enabled: Optional[bool] = None


# Deliberately not `EmailStr`: that pulls in a whole dependency to enforce a
# shape this server never relies on. Deliverability is proven by sending mail,
# not by a regex, so all that is needed here is a normalised, plausible key.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


def normalize_email(value: str) -> str:
    cleaned = (value or "").strip().lower()
    if not _EMAIL_RE.match(cleaned) or len(cleaned) > 254:
        raise ValueError("Enter a valid email address.")
    return cleaned


class RegisterRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=12, max_length=256)
    display_name: str = ""

    _clean_email = field_validator("email")(lambda cls, v: normalize_email(v))


class LoginRequest(BaseModel):
    email: str
    password: str

    _clean_email = field_validator("email")(lambda cls, v: normalize_email(v))


class RefreshRequest(BaseModel):
    refresh_token: str


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class MemberRequest(BaseModel):
    email: str
    role: str = db.PROJECT_MEMBER

    _clean_email = field_validator("email")(lambda cls, v: normalize_email(v))


# -- workspace resolution -----------------------------------------------------

def project_id_from(request_or_socket: Any) -> str:
    """
    The project this call is about.

    Local mode has exactly one, so the header is optional there. Cloud mode
    requires it: guessing which tenant a request meant is not a thing to guess.
    """
    headers = request_or_socket.headers
    params = request_or_socket.query_params
    raw = (headers.get("x-project-id") or params.get("project") or "").strip()
    if raw:
        return safe_segment(raw, "project id")
    if config.cloud:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No project selected. Send an X-Project-Id header.",
        )
    return "default"


def session_for(principal: Principal, project_id: str, required_role: str = db.PROJECT_MEMBER) -> WorkspaceSession:
    """Authorises the caller for a project, then hands back its live session."""
    if config.cloud:
        authorize_project(principal, project_id, required_role)
    session = session_manager.get(principal.id, project_id)
    from server.observability import project_id_var

    project_id_var.set(project_id)
    return session


async def workspace(
    request: Request, principal: Principal = Depends(current_principal)
) -> WorkspaceSession:
    """FastAPI dependency: the caller's session for the requested project."""
    return session_for(principal, project_id_from(request))


async def writable_workspace(
    request: Request, principal: Principal = Depends(current_principal)
) -> WorkspaceSession:
    """As `workspace`, but refuses a read-only project member."""
    return session_for(principal, project_id_from(request), required_role=db.PROJECT_MEMBER)


# -- authentication -----------------------------------------------------------

@app.post("/api/auth/register", dependencies=[Depends(rate_limit("auth", "rate_limit_scan"))])
async def register(req: RegisterRequest) -> Dict[str, Any]:
    """
    Creates an account.

    The first account created becomes the administrator; every later one is a
    plain user. That avoids shipping a default password, which is the single
    most reliable way to hand over a deployment.
    """
    if not config.cloud:
        raise HTTPException(status_code=400, detail="Accounts only exist in cloud mode.")

    from sqlmodel import select

    email = req.email.lower()
    with db.session_scope() as session:
        # The very first account always goes through, so a fresh deployment can
        # be claimed; after that, open signup is a deployment decision.
        bootstrap = session.exec(select(db.User).limit(1)).first() is None
        if not bootstrap and not config.allow_signup:
            raise HTTPException(
                status_code=403,
                detail="Self-service registration is disabled. Ask an administrator for an invite.",
            )
        if session.exec(select(db.User).where(db.User.email == email)).first():
            raise HTTPException(status_code=409, detail="An account with that email already exists.")
        user = db.User(
            email=email,
            display_name=req.display_name or email.split("@")[0],
            password_hash=hash_password(req.password),
            role=db.ROLE_ADMIN if bootstrap else db.ROLE_USER,
        )
        session.add(user)
        session.flush()
        payload = issue_token_pair(user.id, user.role, user.email)

    log.info("user_registered", user_id=payload.get("role"), email_domain=email.split("@")[-1])
    return payload


@app.post("/api/auth/login", dependencies=[Depends(rate_limit("auth", "rate_limit_scan"))])
async def login(req: LoginRequest) -> Dict[str, Any]:
    """
    Exchanges a password for a token pair.

    A missing account and a wrong password return the same message, and the
    hash is verified either way, so response text and response time do not tell
    an attacker which addresses are registered.
    """
    if not config.cloud:
        raise HTTPException(status_code=400, detail="Accounts only exist in cloud mode.")

    from sqlmodel import select

    email = req.email.lower()
    with db.session_scope() as session:
        user = session.exec(select(db.User).where(db.User.email == email)).first()
        stored = user.password_hash if user else hash_password("placeholder-for-timing")
        valid = verify_password(req.password, stored)
        if not user or not valid or not user.is_active:
            log.warning("login_failed", email_domain=email.split("@")[-1])
            raise HTTPException(status_code=401, detail="Incorrect email or password.")
        user.last_login_at = time.time()
        session.add(user)
        return issue_token_pair(user.id, user.role, user.email)


@app.post("/api/auth/refresh")
async def refresh_token(req: RefreshRequest) -> Dict[str, Any]:
    """Trades a valid refresh token for a fresh access token."""
    try:
        claims = decode_token(req.refresh_token, expect_type="refresh")
    except TokenError as err:
        raise HTTPException(status_code=401, detail=str(err)) from err
    return {
        "access_token": create_token(
            str(claims["sub"]), role=str(claims.get("role", db.ROLE_USER)), token_type="access"
        ),
        "token_type": "bearer",
        "expires_in": config.access_token_ttl,
    }


@app.get("/api/auth/me")
async def whoami(principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    return {**principal.to_dict(), "mode": config.mode, "local": principal.local}


@app.get("/api/auth/providers")
async def auth_providers() -> Dict[str, Any]:
    """Which sign-in methods this deployment actually has configured."""
    return {
        "password": config.cloud,
        "signup": config.cloud and config.allow_signup,
        "oauth": [name for name in OAUTH_PROVIDERS if oauth_configured(name)],
        "mode": config.mode,
    }


@app.get("/api/auth/oauth/{provider}/start")
async def oauth_start(provider: str) -> Dict[str, Any]:
    if provider not in OAUTH_PROVIDERS or not oauth_configured(provider):
        raise HTTPException(status_code=404, detail=f"{provider} sign-in is not configured.")
    # The state parameter is what makes the callback verifiable as ours rather
    # than a login CSRF planted by another site, so it is recorded here and
    # required - and consumed - on the way back.
    state = secrets.token_urlsafe(24)
    remember_oauth_state(state)
    return {"url": oauth_authorize_url(provider, state), "state": state}


@app.get("/api/auth/oauth/{provider}/callback")
async def oauth_callback(provider: str, code: str = Query(...), state: str = Query("")):
    """Completes an OAuth login and hands the browser back to the app."""
    if provider not in OAUTH_PROVIDERS or not oauth_configured(provider):
        raise HTTPException(status_code=404, detail=f"{provider} sign-in is not configured.")
    if not consume_oauth_state(state):
        raise HTTPException(
            status_code=400,
            detail="This sign-in link is expired or was not started here. Try again.",
        )
    try:
        identity = await oauth_exchange(provider, code)
    except Exception as err:
        raise HTTPException(status_code=401, detail=f"{provider} sign-in failed: {err}") from err

    from sqlmodel import select

    with db.session_scope() as session:
        user = session.exec(select(db.User).where(db.User.email == identity["email"])).first()
        if user is None:
            first_user = session.exec(select(db.User).limit(1)).first() is None
            user = db.User(
                email=identity["email"],
                display_name=identity["name"],
                oauth_provider=provider,
                oauth_subject=identity["subject"],
                role=db.ROLE_ADMIN if first_user else db.ROLE_USER,
            )
            session.add(user)
            session.flush()
        user.last_login_at = time.time()
        tokens = issue_token_pair(user.id, user.role, user.email)

    # The fragment keeps the token out of the Referer header and out of server
    # access logs, unlike a query string.
    return RedirectResponse(url=f"/#access_token={tokens['access_token']}&state={state}")


# -- projects -----------------------------------------------------------------

@app.get("/api/projects")
async def list_projects(principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    """Every project the caller can open."""
    if not config.cloud:
        return {
            "projects": [{
                "id": "default",
                "name": (session_manager.local_root or DEFAULT_WORKSPACE).name,
                "role": db.PROJECT_OWNER,
                "workspace": str(session_manager.local_root or DEFAULT_WORKSPACE),
            }]
        }

    from sqlmodel import select

    with db.session_scope() as session:
        owned = list(session.exec(select(db.Project).where(db.Project.owner_id == principal.id)))
        member_ids = [
            m.project_id
            for m in session.exec(
                select(db.ProjectMember).where(db.ProjectMember.user_id == principal.id)
            )
        ]
        shared = (
            list(session.exec(select(db.Project).where(db.Project.id.in_(member_ids))))
            if member_ids
            else []
        )
        seen = {}
        for project, role in [(p, db.PROJECT_OWNER) for p in owned] + [
            (p, db.PROJECT_MEMBER) for p in shared
        ]:
            seen.setdefault(
                project.id,
                {
                    "id": project.id,
                    "name": project.name,
                    "role": role,
                    "created_at": project.created_at,
                    "last_active_at": project.last_active_at,
                },
            )
    return {"projects": list(seen.values())}


@app.post("/api/projects")
async def create_project(
    req: ProjectCreateRequest, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    if not config.cloud:
        raise HTTPException(status_code=400, detail="Projects only exist in cloud mode.")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in req.name.lower())[:60]
    with db.session_scope() as session:
        project = db.Project(owner_id=principal.id, name=req.name, slug=slug)
        session.add(project)
        session.flush()
        result = {"id": project.id, "name": project.name, "role": db.PROJECT_OWNER}
    # Creating the volume up front means the first request does not pay for it.
    session_manager.volume_for(principal.id, result["id"]).mkdir(parents=True, exist_ok=True)
    log.info("project_created", project_id=result["id"])
    return result


@app.post("/api/projects/{project_id}/members")
async def add_member(
    project_id: str, req: MemberRequest, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    """Grants another account access. Owner only."""
    authorize_project(principal, project_id, db.PROJECT_OWNER)
    if req.role not in db.PROJECT_ROLES:
        raise HTTPException(status_code=400, detail=f"Role must be one of {db.PROJECT_ROLES}.")

    from sqlmodel import select

    with db.session_scope() as session:
        user = session.exec(select(db.User).where(db.User.email == req.email.lower())).first()
        if user is None:
            raise HTTPException(status_code=404, detail="No account with that email.")
        existing = session.exec(
            select(db.ProjectMember).where(
                db.ProjectMember.project_id == project_id,
                db.ProjectMember.user_id == user.id,
            )
        ).first()
        if existing:
            existing.role = req.role
            session.add(existing)
        else:
            session.add(
                db.ProjectMember(project_id=project_id, user_id=user.id, role=req.role)
            )
        return {"ok": True, "user_id": user.id, "role": req.role}


@app.delete("/api/projects/{project_id}/members/{user_id}")
async def remove_member(
    project_id: str, user_id: str, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    authorize_project(principal, project_id, db.PROJECT_OWNER)

    from sqlmodel import select

    with db.session_scope() as session:
        membership = session.exec(
            select(db.ProjectMember).where(
                db.ProjectMember.project_id == project_id,
                db.ProjectMember.user_id == user_id,
            )
        ).first()
        if membership is None:
            raise HTTPException(status_code=404, detail="Not a member of this project.")
        session.delete(membership)
    # Their live session is now unauthorised; do not leave it running.
    session_manager.close(user_id, project_id)
    return {"ok": True}


# -- status and settings ------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Liveness for the load balancer. Deliberately cheap and unauthenticated."""
    return {"status": "ok", "version": app.version}


@app.get("/api/status")
def get_status(request: Request) -> Dict[str, Any]:
    """Hardware, model and workspace telemetry for the header."""
    engine = _engine  # never triggers a load; the UI polls this

    model_block: Dict[str, Any] = {"loaded": engine is not None, "error": _engine_error}
    if engine is not None:
        described = engine.describe()
        model_block.update({
            "checkpoint": described["checkpoint"],
            "parameters": described["parameters"],
            "architecture": described["architecture"],
            "memory": described["memory"],
            "last_generation": described["last_generation"],
        })

    try:
        principal = resolve_principal(request.headers, request.query_params)
    except TokenError:
        principal = None

    payload: Dict[str, Any] = {
        "status": "online",
        "version": app.version,
        "mode": config.mode,
        "platform": f"{platform.system()} {platform.release()}",
        "local_model": model_block,
        "runtime": _startup_profile.to_dict() if _startup_profile else {},
        "hardware": hardware_snapshot(),
        "execution": {
            "driver": config.driver_kind,
            "docker": docker_available() if config.driver_kind == "container" else None,
            "idle_timeout_sec": config.session_idle_timeout,
        },
        "infrastructure": {
            "database": db.healthcheck() if config.cloud else {"available": False, "reason": "local mode"},
            "jobs": broker_health(),
            "rate_limit": limiter.backend,
            "tracing": _tracing_enabled,
        },
    }

    if principal is None:
        return payload

    user_settings = settings_for(principal)
    payload["active"] = user_settings.get("active", {})
    payload["user"] = principal.to_dict()

    # Workspace facts need a project, which an unauthenticated poll does not
    # have; the header stays useful without them.
    #
    # This deliberately peeks instead of creating: the UI polls status on a
    # timer, and a poll that touched the session would keep a sandbox alive
    # forever and make the idle timeout unreachable.
    try:
        project_id = project_id_from(request)
    except HTTPException:
        return payload

    session = session_manager.peek(principal.id, project_id)
    if session is None:
        # Local mode has exactly one workspace and it is known without opening
        # a session, so the header still shows the folder before first use.
        payload["workspace"] = None if config.cloud else str(session_manager.local_root)
        payload["time_machine"] = {"head": None, "store_bytes": 0}
        return payload

    payload["workspace"] = str(session.root)
    payload["time_machine"] = {
        "head": session.time_machine.head,
        "store_bytes": session.time_machine.store_size(),
    }
    return payload


def hardware_snapshot() -> Dict[str, Any]:
    """Device telemetry, tolerant of an install with no torch."""
    try:
        import torch

        from inference.runtime import gpu_memory_snapshot, host_memory_snapshot

        return {
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            **gpu_memory_snapshot(),
            **host_memory_snapshot(),
        }
    except Exception:
        return {"cuda_available": False, "device": "CPU"}


@app.get("/api/settings")
def get_settings(principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    return settings_for(principal).public()


@app.post("/api/settings")
def update_settings(
    req: SettingsRequest, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    user_settings = settings_for(principal)
    user_settings.update(req.section, req.values)
    if req.section == "local":
        # Force the next request to pick up the new checkpoint or device.
        global _engine
        _engine = None
    return user_settings.public()


@app.get("/api/providers")
def get_providers(principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    """The provider catalogue plus which ones already hold a key."""
    return {"providers": list_specs(), "configured": settings_for(principal).all_providers_public()}


@app.post("/api/providers/key")
def save_provider_key(
    req: ProviderKeyRequest, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    """Stores a key. The key is never echoed back, only a masked preview."""
    return settings_for(principal).set_provider(
        req.provider, api_key=req.api_key, base_url=req.base_url
    )


@app.delete("/api/providers/{provider_id}")
def forget_provider(
    provider_id: str, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    return {"removed": settings_for(principal).forget_provider(provider_id)}


@app.post("/api/providers/scan", dependencies=[Depends(rate_limit("scan", "rate_limit_scan"))])
async def scan_provider_models(
    req: ScanRequest, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    """
    Discovers the models a key can reach.

    Rate limited harder than the rest of the API: every call is an outbound
    request against the user's own provider quota, and it is trivially
    scriptable into a way to burn it.
    """
    user_settings = settings_for(principal)
    key = req.api_key if req.api_key is not None else user_settings.api_key(req.provider)
    base = req.base_url if req.base_url is not None else user_settings.base_url(req.provider)

    with span("provider.scan", provider=req.provider):
        result = await scan_models(req.provider, api_key=key, base_url=base, engine_factory=get_engine)

    if req.save and result.get("ok"):
        user_settings.set_provider(
            req.provider,
            api_key=req.api_key if req.api_key else None,
            base_url=req.base_url if req.base_url else None,
            models=result["models"],
        )
    return result


@app.post("/api/providers/select")
def select_model(
    req: SelectModelRequest, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    user_settings = settings_for(principal)
    user_settings.set_provider(req.provider, selected_model=req.model)
    user_settings.update("active", {"provider": req.provider, "model": req.model})
    if req.provider == "local":
        global _engine
        _engine = None
        user_settings.update("local", {"checkpoint": req.model})
    return user_settings.public()


# -- workspace ----------------------------------------------------------------

@app.get("/api/workspace/current")
def get_current_workspace(session: WorkspaceSession = Depends(workspace)) -> Dict[str, Any]:
    return {"workspace": str(session.root), "project_id": session.project_id}


@app.post("/api/workspace/select")
def select_workspace(
    req: WorkspaceSelectRequest, principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    """
    Points the single-user install at a different folder.

    Refused outright in cloud mode. An endpoint that takes an absolute host
    path and starts serving its contents is a filesystem read primitive for
    anyone who can call it; multi-tenant deployments select a project instead,
    and a project only ever names a directory under the workspaces root.
    """
    if config.cloud:
        raise HTTPException(
            status_code=403,
            detail="Workspaces are selected by project in cloud mode.",
        )

    raw_path = req.path.strip() if req.path else ""
    if not raw_path:
        session_manager.local_root = DEFAULT_WORKSPACE
        session_manager.close_all()
        return {"ok": True, "workspace": str(DEFAULT_WORKSPACE)}

    target = Path(raw_path).resolve()
    if not target.exists():
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"Cannot create directory: {err}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {raw_path}")

    session_manager.local_root = target
    session_manager.close_all()
    log.info("workspace_selected", workspace=str(target))
    return {"ok": True, "workspace": str(target)}


def ask_directory_dialog(title: str = "Select Folder", initial_dir: Optional[str] = None) -> Optional[str]:
    """Opens the native folder picker on the host desktop, in a subprocess."""
    script_path = PROJECT_ROOT / "scripts" / "pick_folder.py"
    try:
        import subprocess

        res = subprocess.run(
            [sys.executable, str(script_path), title, initial_dir or str(DEFAULT_WORKSPACE)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = res.stdout.strip()
        if out and Path(out).is_dir():
            return out
    except Exception:
        pass
    return None


def _require_host_picker() -> None:
    """
    The picker draws a window on the machine running the server.

    On a server that is either nothing (no display) or a request that hangs for
    two minutes holding a worker, so it is refused unless a local install opts
    in.
    """
    if config.cloud or not config.allow_host_directory_picker:
        raise HTTPException(
            status_code=403,
            detail="The host directory picker is disabled. Set SHREE_ALLOW_HOST_PICKER=1 for a local install.",
        )


@app.post("/api/workspace/browse")
async def browse_workspace(principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    _require_host_picker()
    folder = await asyncio.to_thread(
        ask_directory_dialog,
        "Select Project Workspace Folder for Ura-Shree",
        str(session_manager.local_root or DEFAULT_WORKSPACE),
    )
    if not folder:
        return {
            "ok": False, "cancelled": True,
            "workspace": str(session_manager.local_root or DEFAULT_WORKSPACE),
        }
    return select_workspace(WorkspaceSelectRequest(path=folder), principal)


@app.post("/api/browse-directory")
async def browse_directory(_principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    _require_host_picker()
    folder = await asyncio.to_thread(
        ask_directory_dialog,
        "Select Working Directory",
        str(session_manager.local_root or DEFAULT_WORKSPACE),
    )
    if not folder:
        return {"ok": False, "cancelled": True, "path": None}
    return {"ok": True, "path": folder}


@app.get("/api/workspace/suggestions")
def get_workspace_suggestions(_principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    """Convenience paths for the folder picker. Local installs only."""
    if config.cloud:
        return {"workspace": None, "project_root": None}
    home = Path.home()
    return {
        "workspace": str(DEFAULT_WORKSPACE) if DEFAULT_WORKSPACE.exists() else None,
        "project_root": str(PROJECT_ROOT),
        "desktop": str(home / "Desktop") if (home / "Desktop").exists() else None,
        "documents": str(home / "Documents") if (home / "Documents").exists() else None,
        "downloads": str(home / "Downloads") if (home / "Downloads").exists() else None,
    }


@app.get("/api/tree")
def get_file_tree(
    max_entries: int = 4000, session: WorkspaceSession = Depends(workspace)
) -> Dict[str, Any]:
    """The workspace as a nested tree for the explorer."""
    root_path = session.root
    ignored = ignored_dirs_for(root_path)
    counter = {"n": 0}

    def build(path: Path) -> Optional[Dict[str, Any]]:
        if counter["n"] >= max_entries:
            return None
        counter["n"] += 1

        node: Dict[str, Any] = {
            "name": path.name or str(path),
            "path": str(path.relative_to(root_path)).replace("\\", "/"),
            "isDir": path.is_dir(),
        }
        if path.is_dir():
            children = []
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except (PermissionError, OSError):
                entries = []
            for entry in entries:
                if entry.name in ignored:
                    continue
                if root_path == PROJECT_ROOT and not os.environ.get("SHREE_DEV"):
                    if not entry.is_dir() and entry.name in {".gitignore", "requirements.txt"}:
                        continue
                child = build(entry)
                if child:
                    children.append(child)
            node["children"] = children
        else:
            try:
                node["size"] = path.stat().st_size
            except OSError:
                node["size"] = 0
        return node

    root = build(root_path) or {"name": root_path.name, "path": "", "isDir": True, "children": []}
    root["path"] = ""
    return {
        "tree": root,
        "truncated": counter["n"] >= max_entries,
        "workspace": str(root_path),
    }


@app.get("/api/file")
def read_file(path: str = Query(...), session: WorkspaceSession = Depends(workspace)) -> Dict[str, Any]:
    try:
        result = session.driver.read_file(path)
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err))
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "read failed"))
    return result


@app.post("/api/file")
@app.put("/api/file")
def write_file(
    req: FileWriteRequest, session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    try:
        snapshot = session.time_machine.snapshot(f"Before editing {req.path}", kind="auto")
        mirror_snapshot(session.project_id, snapshot)
        result = session.driver.write_file(req.path, req.content)
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err))
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "write failed"))
    db.record_tool_invocation(
        tool_name="api.write_file",
        arguments={"path": req.path, "bytes": len(req.content)},
        status="ok",
        approved_by="api",
        user_id=session.user_id,
        project_id=session.project_id,
        session_id=session.session_id,
    )
    return result


@app.delete("/api/file")
def delete_file(
    path: str = Query(...), session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    rel = safe_relative_path(path)
    if not rel:
        raise HTTPException(status_code=400, detail="Invalid path.")
    root = session.root.resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(status_code=403, detail="Path outside workspace.")
    if target == root:
        raise HTTPException(status_code=400, detail="Cannot delete workspace root directory.")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File or directory not found.")

    try:
        mirror_snapshot(
            session.project_id, session.time_machine.snapshot(f"Before deleting {rel}", kind="auto")
        )
    except Exception:
        # A snapshot failure must not make the delete impossible, only
        # unrecoverable - which the audit row below still records.
        pass

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    db.record_tool_invocation(
        tool_name="api.delete_file",
        arguments={"path": rel},
        status="ok",
        approved_by="api",
        user_id=session.user_id,
        project_id=session.project_id,
        session_id=session.session_id,
    )
    return {"ok": True, "deleted": rel}


# -- skills -------------------------------------------------------------------

@app.get("/api/skills")
def get_skills(_principal: Principal = Depends(current_principal)) -> List[Dict[str, Any]]:
    return skills_mgr.list_skills()


@app.post("/api/skills")
def add_skill(
    req: SkillCreateRequest, _principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    if not req.name.strip() or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Skill name and prompt are required.")
    return skills_mgr.add_skill(req.name, req.description, req.prompt)


@app.patch("/api/skills/{skill_id}")
def update_skill(
    skill_id: str, req: SkillToggleRequest, _principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    updated = skills_mgr.toggle_skill(skill_id, req.enabled)
    if not updated:
        raise HTTPException(status_code=404, detail="Skill not found.")
    return updated


@app.delete("/api/skills/{skill_id}")
def remove_skill(
    skill_id: str, _principal: Principal = Depends(current_principal)
) -> Dict[str, Any]:
    ok = skills_mgr.delete_skill(skill_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot delete built-in skill or skill not found.")
    return {"ok": True}


@app.post("/api/upload")
def upload_files(
    req: UploadRequest, session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    """
    Accepts files or a whole folder from the browser.

    Content arrives base64-encoded in JSON, which keeps the dependency list
    short and handles both the file picker and a directory drop the same way.
    """
    if len(req.files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"At most {MAX_UPLOAD_FILES} files per upload.",
        )

    written: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []

    target_dir = safe_relative_path(req.target_dir) if req.target_dir else ""
    if req.target_dir and target_dir is None:
        raise HTTPException(status_code=400, detail="Invalid target directory.")

    for item in req.files:
        name = safe_relative_path(item.path)
        if name is None:
            rejected.append({"path": item.path, "reason": "path escapes the upload directory"})
            continue
        relative = f"{target_dir}/{name}" if target_dir else name

        suffix = Path(name).suffix.lower()
        if suffix in BLOCKED_UPLOAD_SUFFIXES:
            rejected.append({"path": item.path, "reason": f"'{suffix}' files are not accepted"})
            continue

        try:
            data = base64.b64decode(item.content_base64, validate=True)
        except Exception:
            rejected.append({"path": item.path, "reason": "content was not valid base64"})
            continue

        if len(data) > MAX_UPLOAD_BYTES:
            rejected.append({
                "path": item.path,
                "reason": f"{len(data) // 1024}KB exceeds the {MAX_UPLOAD_BYTES // 1024}KB limit",
            })
            continue

        try:
            # Second line of defence: the sandbox still confirms containment.
            destination = session.driver.filesystem._resolve_safe_path(relative)
        except PermissionError as err:
            rejected.append({"path": item.path, "reason": str(err)})
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        written.append({
            "path": str(destination.relative_to(session.root)).replace("\\", "/"),
            "bytes": len(data),
        })

    if written:
        mirror_snapshot(
            session.project_id,
            session.time_machine.snapshot(f"Uploaded {len(written)} file(s)", kind="upload"),
        )
    return {"written": written, "rejected": rejected, "count": len(written)}


@app.get("/api/memory")
def get_memory(session: WorkspaceSession = Depends(workspace)) -> Dict[str, Any]:
    return {
        "summary": session.memory.get_summary_context(),
        "facts": session.memory.get_facts_by_category(),
        "decisions": session.memory.get_recent_decisions(limit=25),
    }


@app.post("/api/index")
def build_index(session: WorkspaceSession = Depends(workspace)) -> Dict[str, Any]:
    """
    Rebuilds the symbol index.

    Handed to the job queue when one is configured: indexing a large repository
    takes long enough that holding a request open for it is how a deploy ends
    up killing the work halfway through.
    """
    queued = enqueue("index_workspace", workspace=str(session.root))
    if queued["queued"]:
        return {"queued": True, "task_id": queued["task_id"]}

    stats = session.indexer.scan_and_index()
    symbols = [
        {
            "file": file_path,
            "name": sym["name"],
            "kind": sym.get("kind", "symbol"),
            "line": sym["line"],
            "signature": sym.get("signature", ""),
            "doc": sym.get("doc", ""),
        }
        for file_path, sym_list in session.indexer.file_symbols.items()
        for sym in sym_list
    ]
    return {
        "queued": False,
        "stats": stats,
        "symbols": symbols,
        "tree_summary": session.indexer.get_tree_summary(),
    }


@app.get("/api/jobs/{task_id}")
def get_job(task_id: str, _principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
    return job_status(task_id)


@app.get("/api/git/status")
def git_status(session: WorkspaceSession = Depends(workspace)) -> Dict[str, Any]:
    return {"status": session.git.status(), "diff": session.git.diff()}


# -- time machine -------------------------------------------------------------

@app.get("/api/timemachine")
def get_timeline(limit: int = 200, session: WorkspaceSession = Depends(workspace)) -> Dict[str, Any]:
    return session.time_machine.timeline(limit=limit)


@app.post("/api/timemachine/snapshot")
def create_snapshot(
    req: SnapshotRequest, session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    snapshot = session.time_machine.snapshot(req.label, kind="manual")
    mirror_snapshot(session.project_id, snapshot)
    return snapshot


@app.get("/api/timemachine/diff")
def snapshot_diff(
    from_id: str = Query(...), to_id: str = Query(...),
    session: WorkspaceSession = Depends(workspace),
) -> Dict[str, Any]:
    try:
        return session.time_machine.diff(from_id, to_id)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.get("/api/timemachine/file")
def snapshot_file(
    snapshot_id: str = Query(...), path: str = Query(...),
    session: WorkspaceSession = Depends(workspace),
) -> Dict[str, Any]:
    try:
        return session.time_machine.file_at(snapshot_id, path)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.post("/api/timemachine/restore")
def restore_snapshot(
    req: RestoreRequest, session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    try:
        result = session.time_machine.restore(req.snapshot_id, dry_run=req.dry_run)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))
    db.record_tool_invocation(
        tool_name="api.restore_snapshot",
        arguments={"snapshot_id": req.snapshot_id, "dry_run": req.dry_run},
        status="ok",
        approved_by="api",
        user_id=session.user_id,
        project_id=session.project_id,
        session_id=session.session_id,
    )
    return result


@app.post("/api/timemachine/prune")
def prune_snapshots(
    keep: int = 100, session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    queued = enqueue("prune_snapshots", workspace=str(session.root), keep=keep)
    if queued["queued"]:
        return {"queued": True, "task_id": queued["task_id"]}
    return queued["result"]


# -- terminal -----------------------------------------------------------------

@app.get("/api/terminal/sessions")
def terminal_sessions(session: WorkspaceSession = Depends(workspace)) -> Dict[str, Any]:
    return {"sessions": [session.driver.info()]}


@app.post("/api/terminal")
def run_command(
    req: CommandRequest, session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    """Blocking command execution, for callers that do not want the websocket."""
    blocked = check_command(req.command)
    if blocked:
        db.record_tool_invocation(
            tool_name="terminal.run", arguments={"command": req.command}, status="blocked",
            approved_by="api", user_id=session.user_id, project_id=session.project_id,
            session_id=session.session_id,
        )
        # A refusal is a result, not a protocol error: the caller asked a valid
        # question and this is the answer, rendered in the terminal like any
        # other failed command.
        return {
            "success": False,
            "command": req.command,
            "returncode": -1,
            "output": "",
            "error": f"Refused: the command matches a blocked pattern ({blocked}).",
            "duration_ms": 0,
            "cwd": session.driver.info().get("relative_cwd", ""),
        }

    result = session.driver.run(req.command, timeout=float(req.timeout))
    db.record_tool_invocation(
        tool_name="terminal.run",
        arguments={"command": req.command},
        status="ok" if result.get("success") else "failed",
        approved_by="api",
        exit_code=result.get("returncode"),
        duration_ms=result.get("duration_ms", 0),
        user_id=session.user_id,
        project_id=session.project_id,
        session_id=session.session_id,
    )
    return result


@app.delete("/api/terminal/{session_id}")
def close_terminal(
    session_id: str, session: WorkspaceSession = Depends(writable_workspace)
) -> Dict[str, Any]:
    """Restarts this project's shell. The id is accepted for API compatibility."""
    session.driver.restart()
    return {"closed": True}


@app.websocket("/ws/terminal")
async def terminal_socket(websocket: WebSocket) -> None:
    """
    Live terminal. Output streams line by line as the command produces it,
    instead of arriving in one block when the process exits.
    """
    principal = await principal_for_socket(websocket)
    if principal is None:
        return
    try:
        project_id = project_id_from(websocket)
        session = session_for(principal, project_id)
    except HTTPException as err:
        await websocket.close(code=1008, reason=str(err.detail)[:120])
        return

    await websocket.accept()
    frames = SocketLimiter(f"terminal:{principal.id}", config.rate_limit_ws)
    with bind_context(user_id=principal.id, project_id=project_id, session_id=session.session_id):
        await websocket.send_json({"type": "ready", "info": session.driver.info()})

        try:
            while True:
                raw = await websocket.receive_text()
                allowed, retry_after = await frames.allow()
                if not allowed:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Too many messages. Wait {retry_after:.0f}s.",
                    })
                    continue

                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON."})
                    continue

                action = message.get("action", "run")
                session.touch()

                if action == "run":
                    command = (message.get("command") or "").strip()
                    if not command:
                        continue

                    blocked = check_command(command)
                    if blocked:
                        db.record_tool_invocation(
                            tool_name="terminal.run", arguments={"command": command},
                            status="blocked", approved_by="user", user_id=principal.id,
                            project_id=project_id, session_id=session.session_id,
                        )
                        await websocket.send_json({
                            "type": "exit", "code": -1, "success": False,
                            "error": f"Refused: matches a blocked pattern ({blocked}).",
                        })
                        continue

                    await websocket.send_json({"type": "started", "command": command})
                    queue: asyncio.Queue = asyncio.Queue()
                    loop = asyncio.get_running_loop()
                    timeout = min(float(message.get("timeout", 600)), 3600.0)

                    def pump() -> None:
                        for event in session.driver.stream(command, timeout=timeout):
                            loop.call_soon_threadsafe(queue.put_nowait, event)
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                    asyncio.create_task(asyncio.to_thread(pump))
                    exit_event: Dict[str, Any] = {}
                    while True:
                        event = await queue.get()
                        if event is None:
                            break
                        if event.get("type") == "exit":
                            exit_event = event
                        await websocket.send_json(event)

                    db.record_tool_invocation(
                        tool_name="terminal.run",
                        arguments={"command": command},
                        status="ok" if exit_event.get("success") else "failed",
                        approved_by="user",
                        exit_code=exit_event.get("code"),
                        duration_ms=exit_event.get("duration_ms", 0),
                        user_id=principal.id,
                        project_id=project_id,
                        session_id=session.session_id,
                    )

                elif action == "restart":
                    session.driver.restart()
                    await websocket.send_json({"type": "ready", "info": session.driver.info()})

                elif action == "info":
                    await websocket.send_json({"type": "info", "info": session.driver.info()})

                elif action == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            return
        except Exception as err:
            log.error("terminal_socket_failed", error=str(err))
            try:
                await websocket.send_json({"type": "error", "message": "Terminal session failed."})
            except Exception:
                pass


# -- agent --------------------------------------------------------------------

@app.websocket("/ws/agent")
async def agent_socket(websocket: WebSocket) -> None:
    """
    The chat and agent channel.

    Client sends `{"action": "chat", "message": ..., "provider": ..., "model": ...}`
    and receives the loop's events verbatim, so the UI renders tool calls as
    they run rather than after the fact.
    """
    principal = await principal_for_socket(websocket)
    if principal is None:
        return
    try:
        project_id = project_id_from(websocket)
        session = session_for(principal, project_id)
    except HTTPException as err:
        await websocket.close(code=1008, reason=str(err.detail)[:120])
        return

    await websocket.accept()
    running: Optional[asyncio.Task] = None
    pending_approvals: Dict[str, asyncio.Future] = {}
    frames = SocketLimiter(f"agent:{principal.id}", config.rate_limit_ws)
    user_settings = settings_for(principal)

    async def handle_approval(call: Any) -> bool:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        pending_approvals[call.id] = fut
        try:
            await websocket.send_json({
                "type": "tool_approval_prompt",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })
            return await asyncio.wait_for(fut, timeout=300)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        finally:
            pending_approvals.pop(call.id, None)

    async def drive(message: Dict[str, Any]) -> None:
        active = user_settings.get("active", {})
        provider_id = message.get("provider") or active.get("provider", "local")
        model = message.get("model") or active.get("model") or ""

        if not model and provider_id != "local":
            await websocket.send_json({
                "type": "error",
                "message": f"No model selected for {provider_id}. Scan models and pick one in Settings.",
            })
            return

        allowed, retry_after = await frames.allow(cost=5.0)
        if not allowed:
            await websocket.send_json({
                "type": "error",
                "message": f"Too many turns started. Wait {retry_after:.0f}s.",
            })
            return

        try:
            provider = provider_for(principal, provider_id)
        except ProviderError as err:
            await websocket.send_json({"type": "error", "message": str(err), "status": err.status})
            return

        prompt = message.get("message", "")
        persist_chat_message(project_id, principal.id, session.session_id, "user", prompt)
        session.touch()
        await websocket.send_json({
            "type": "run_start", "provider": provider_id, "model": model or "(local checkpoint)",
        })

        reply: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        try:
            with span("agent.turn", provider=provider_id, model=model or "local"):
                async for event in session.agent.stream_task(
                    request=prompt,
                    provider=provider,
                    model=model,
                    temperature=float(message.get("temperature", active.get("temperature", 0.7))),
                    max_tokens=int(message.get("max_tokens", active.get("max_tokens", 4096))),
                    use_tools=bool(message.get("use_tools", active.get("use_tools", True))),
                    auto_approve=bool(message.get("auto_approve", active.get("auto_approve", False))),
                    attachments=message.get("attachments"),
                    fresh=bool(message.get("fresh")),
                    skills_prompt=skills_mgr.get_active_prompt(),
                    approval_callback=handle_approval,
                ):
                    if event.get("type") == "text":
                        reply.append(event.get("text", ""))
                    elif event.get("type") == "tool_end":
                        tool_calls.append({"name": event.get("name"), "ok": event.get("ok")})
                    await websocket.send_json(event)
        except asyncio.CancelledError:
            await websocket.send_json({"type": "cancelled"})
            raise
        except Exception as err:
            log.error("agent_turn_failed", error=str(err))
            await websocket.send_json({"type": "error", "message": str(err)})
        finally:
            persist_chat_message(
                project_id, principal.id, session.session_id,
                "assistant", "".join(reply), tool_calls,
            )
            session.touch()
            await provider.aclose()

    with bind_context(user_id=principal.id, project_id=project_id, session_id=session.session_id):
        try:
            while True:
                raw = await websocket.receive_text()
                allowed, retry_after = await frames.allow()
                if not allowed:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Too many messages. Wait {retry_after:.0f}s.",
                    })
                    continue

                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON."})
                    continue

                action = message.get("action", "chat")

                if action == "chat":
                    if running and not running.done():
                        await websocket.send_json({
                            "type": "error", "message": "A turn is already running. Stop it first.",
                        })
                        continue
                    running = asyncio.create_task(drive(message))

                elif action == "tool_approval":
                    tool_id = str(message.get("id", ""))
                    approved = bool(message.get("approved", False))
                    if tool_id in pending_approvals and not pending_approvals[tool_id].done():
                        pending_approvals[tool_id].set_result(approved)

                elif action == "stop":
                    if running and not running.done():
                        running.cancel()
                    for fut in pending_approvals.values():
                        if not fut.done():
                            fut.cancel()
                    await websocket.send_json({"type": "stopped"})

                elif action == "reset":
                    session.agent.reset()
                    for fut in pending_approvals.values():
                        if not fut.done():
                            fut.cancel()
                    await websocket.send_json({"type": "reset"})

                elif action == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            if running and not running.done():
                running.cancel()
        except Exception as err:
            log.error("agent_socket_failed", error=str(err))


# -- administration -----------------------------------------------------------

@app.get("/api/admin/sessions")
def admin_sessions(_admin: Principal = Depends(require_admin)) -> Dict[str, Any]:
    return {"sessions": session_manager.list_sessions()}


@app.delete("/api/admin/sessions/{user_id}/{project_id}")
def admin_close_session(
    user_id: str, project_id: str, _admin: Principal = Depends(require_admin)
) -> Dict[str, Any]:
    return {"closed": session_manager.close(user_id, project_id)}


@app.get("/api/admin/audit")
def admin_audit(
    limit: int = Query(200, le=1000),
    user_id: str = "",
    project_id: str = "",
    tool_name: str = "",
    _admin: Principal = Depends(require_admin),
) -> Dict[str, Any]:
    """The tool invocation trail, newest first."""
    if not config.cloud:
        raise HTTPException(status_code=400, detail="The audit table only exists in cloud mode.")

    from sqlmodel import select

    with db.session_scope() as session:
        query = select(db.AuditEntry).order_by(db.AuditEntry.created_at.desc()).limit(limit)
        if user_id:
            query = query.where(db.AuditEntry.user_id == user_id)
        if project_id:
            query = query.where(db.AuditEntry.project_id == project_id)
        if tool_name:
            query = query.where(db.AuditEntry.tool_name == tool_name)
        rows = [
            {
                "id": row.id,
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
            for row in session.exec(query)
        ]
    return {"entries": rows, "count": len(rows)}


@app.get("/api/admin/audit/verify")
def admin_audit_verify(_admin: Principal = Depends(require_admin)) -> Dict[str, Any]:
    """Walks the hash chain and reports the first broken link, if any."""
    if not config.cloud:
        raise HTTPException(status_code=400, detail="The audit table only exists in cloud mode.")
    return db.verify_audit_chain()


@app.exception_handler(ProviderError)
async def provider_error_handler(_request, exc: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=exc.status or 502, content={"detail": str(exc)})


# The compiled frontend is mounted last so it never shadows an /api route.
_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
