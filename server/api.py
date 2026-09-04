"""
URA-Shree backend.

REST for state the UI reads once (status, settings, file tree, timeline), and
WebSockets for the two things that stream: the agent turn and the terminal.

The server owns one shell manager and one Time Machine for the workspace, so a
terminal session survives page reloads and every agent edit lands in the same
history the user is scrubbing through.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import CodingAgent
from agent.indexer import CodebaseIndexer
from agent.memory import ProjectMemory
from agent.timemachine import TimeMachine
from inference.runtime import gpu_memory_snapshot, host_memory_snapshot, tune_runtime
from providers import ProviderError, build_provider, list_specs, scan_models
from server.settings import Settings
from tools.filesystem import FileSystemTool
from tools.git import GitTool
from tools.shell import ShellManager, check_command
from server.skills import SkillsManager

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

# Hidden from the file explorer. ".shree" holds API keys and ".timemachine"
# holds the snapshot object store; neither is content the user browses.
IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    "dist", "build", ".idea", ".vscode", ".mypy_cache", ".ruff_cache",
    ".shree", ".timemachine", ".cache", ".next", ".turbo", "checkpoints",
}

if WORKSPACE_ROOT and WORKSPACE_ROOT == PROJECT_ROOT and not os.environ.get("SHREE_DEV"):
    IGNORED_DIRS |= INTERNAL_FRAMEWORK_DIRS

# Files a browser upload is allowed to place in the workspace. Everything else
# is rejected: an upload endpoint that accepts .exe or .dll is a foothold, not a
# feature, and nothing in this app needs one.
BLOCKED_UPLOAD_SUFFIXES = {
    ".exe", ".dll", ".so", ".dylib", ".msi", ".bat", ".cmd", ".com", ".scr",
    ".ps1", ".vbs", ".jar", ".apk", ".sys", ".drv",
}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


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

app = FastAPI(
    title="Ura-Shree",
    description="Local AI model and coding agent, with bring-your-own-key provider support.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- shared state -------------------------------------------------------------

fs_tool = FileSystemTool(str(WORKSPACE_ROOT)) if WORKSPACE_ROOT else None
git_tool = GitTool(str(WORKSPACE_ROOT)) if WORKSPACE_ROOT else None
shells = ShellManager(str(WORKSPACE_ROOT)) if WORKSPACE_ROOT else ShellManager(str(PROJECT_ROOT))
indexer = CodebaseIndexer(str(WORKSPACE_ROOT)) if WORKSPACE_ROOT else None
memory_file = (WORKSPACE_ROOT / "checkpoints" / "shree_memory.db") if WORKSPACE_ROOT else (PROJECT_ROOT / "checkpoints" / "shree_memory.db")
if not memory_file.exists() and (PROJECT_ROOT / "checkpoints" / "shree_memory.db").exists():
    memory_file = PROJECT_ROOT / "checkpoints" / "shree_memory.db"
else:
    memory_file.parent.mkdir(parents=True, exist_ok=True)
memory_store = ProjectMemory(str(memory_file))
time_machine = TimeMachine(str(WORKSPACE_ROOT)) if WORKSPACE_ROOT else None

settings_file = (WORKSPACE_ROOT / ".shree" / "settings.json") if WORKSPACE_ROOT else (PROJECT_ROOT / ".shree" / "settings.json")
if not settings_file.exists() and (PROJECT_ROOT / ".shree" / "settings.json").exists():
    settings_file = PROJECT_ROOT / ".shree" / "settings.json"
settings = Settings(str(settings_file))

_agents: Dict[str, CodingAgent] = {}
_engine = None
_engine_error = ""
_startup_profile = None


def get_agent(session_id: str = "default") -> CodingAgent:
    agent = _agents.get(session_id)
    if agent is None:
        agent = CodingAgent(
            workspace_root=str(WORKSPACE_ROOT) if WORKSPACE_ROOT else None,
            shell_manager=shells,
            time_machine=time_machine,
            session_id=session_id,
        )
        _agents[session_id] = agent
    return agent


def get_engine(force_reload: bool = False):
    """Loads the local checkpoint on first use, so startup stays fast."""
    global _engine, _engine_error
    if _engine is not None and not force_reload:
        return _engine

    local_cfg = settings.get("local", {}) or {}
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


def provider_for(provider_id: str):
    """Builds a provider client from stored credentials."""
    return build_provider(
        provider_id,
        api_key=settings.api_key(provider_id),
        base_url=settings.base_url(provider_id),
        engine_factory=get_engine,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Tune the runtime once at boot, and tear down live shells on exit."""
    global _startup_profile
    _, _, _startup_profile = tune_runtime()
    yield
    shells.close_all()


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


# -- status and settings ------------------------------------------------------

@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Hardware, model and workspace telemetry for the header."""
    engine = _engine  # never triggers a load; the UI polls this
    active = settings.get("active", {})

    model_block: Dict[str, Any] = {
        "loaded": engine is not None,
        "error": _engine_error,
    }
    if engine is not None:
        described = engine.describe()
        model_block.update({
            "checkpoint": described["checkpoint"],
            "parameters": described["parameters"],
            "architecture": described["architecture"],
            "memory": described["memory"],
            "last_generation": described["last_generation"],
        })

    return {
        "status": "online",
        "version": app.version,
        "workspace": str(WORKSPACE_ROOT) if WORKSPACE_ROOT else None,
        "platform": f"{platform.system()} {platform.release()}",
        "active": active,
        "local_model": model_block,
        "runtime": _startup_profile.to_dict() if _startup_profile else {},
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            **gpu_memory_snapshot(),
            **host_memory_snapshot(),
        },
        "time_machine": {
            "head": time_machine.head if time_machine else None,
            "store_bytes": time_machine.store_size() if time_machine else 0,
        },
    }


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    return settings.public()


@app.post("/api/settings")
def update_settings(req: SettingsRequest) -> Dict[str, Any]:
    settings.update(req.section, req.values)
    if req.section == "local":
        # Force the next request to pick up the new checkpoint or device.
        global _engine
        _engine = None
    return settings.public()


@app.get("/api/providers")
def get_providers() -> Dict[str, Any]:
    """The provider catalogue plus which ones already hold a key."""
    return {"providers": list_specs(), "configured": settings.all_providers_public()}


@app.post("/api/providers/key")
def save_provider_key(req: ProviderKeyRequest) -> Dict[str, Any]:
    """Stores a key. The key is never echoed back, only a masked preview."""
    return settings.set_provider(req.provider, api_key=req.api_key, base_url=req.base_url)


@app.delete("/api/providers/{provider_id}")
def forget_provider(provider_id: str) -> Dict[str, Any]:
    return {"removed": settings.forget_provider(provider_id)}


@app.post("/api/providers/scan")
async def scan_provider_models(req: ScanRequest) -> Dict[str, Any]:
    """
    Discovers the models a key can reach.

    This is the step right after pasting a key: it proves the key works and
    fills the model picker with what that account actually has access to,
    rather than a hard-coded list that goes stale.
    """
    key = req.api_key if req.api_key is not None else settings.api_key(req.provider)
    base = req.base_url if req.base_url is not None else settings.base_url(req.provider)

    result = await scan_models(req.provider, api_key=key, base_url=base, engine_factory=get_engine)

    if req.save and result.get("ok"):
        settings.set_provider(
            req.provider,
            api_key=req.api_key if req.api_key else None,
            base_url=req.base_url if req.base_url else None,
            models=result["models"],
        )
    return result


@app.post("/api/providers/select")
def select_model(req: SelectModelRequest) -> Dict[str, Any]:
    settings.set_provider(req.provider, selected_model=req.model)
    settings.update("active", {"provider": req.provider, "model": req.model})
    if req.provider == "local":
        global _engine
        _engine = None
        settings.update("local", {"checkpoint": req.model})
    return settings.public()


# -- workspace ----------------------------------------------------------------

class WorkspaceSelectRequest(BaseModel):
    path: str


class SkillCreateRequest(BaseModel):
    name: str
    description: str = ""
    prompt: str


class SkillToggleRequest(BaseModel):
    enabled: Optional[bool] = None


@app.get("/api/workspace/current")
def get_current_workspace() -> Dict[str, Any]:
    return {"workspace": str(WORKSPACE_ROOT) if WORKSPACE_ROOT else None}


@app.post("/api/workspace/select")
def select_workspace(req: WorkspaceSelectRequest) -> Dict[str, Any]:
    global WORKSPACE_ROOT, fs_tool, git_tool, shells, indexer, time_machine, _agents, IGNORED_DIRS
    raw_path = req.path.strip() if req.path else ""
    if not raw_path:
        WORKSPACE_ROOT = None
        fs_tool = None
        git_tool = None
        indexer = None
        time_machine = None
        shells = ShellManager(str(PROJECT_ROOT))
        _agents.clear()
        return {"ok": True, "workspace": None}

    target = Path(raw_path).resolve()
    if not target.exists():
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"Cannot create directory: {err}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {raw_path}")

    WORKSPACE_ROOT = target
    fs_tool = FileSystemTool(str(WORKSPACE_ROOT))
    git_tool = GitTool(str(WORKSPACE_ROOT))
    shells = ShellManager(str(WORKSPACE_ROOT))
    indexer = CodebaseIndexer(str(WORKSPACE_ROOT))
    time_machine = TimeMachine(str(WORKSPACE_ROOT))
    IGNORED_DIRS = {
        ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
        "dist", "build", ".idea", ".vscode", ".mypy_cache", ".ruff_cache",
        ".shree", ".timemachine", ".cache", ".next", ".turbo", "checkpoints",
    }
    if WORKSPACE_ROOT == PROJECT_ROOT and not os.environ.get("SHREE_DEV"):
        IGNORED_DIRS |= INTERNAL_FRAMEWORK_DIRS
    _agents.clear()
    return {"ok": True, "workspace": str(WORKSPACE_ROOT)}


def ask_directory_dialog(title: str = "Select Folder", initial_dir: Optional[str] = None) -> Optional[str]:
    """Opens native OS File Explorer folder picker dialog on the desktop via a dedicated subprocess."""
    script_path = PROJECT_ROOT / "scripts" / "pick_folder.py"
    try:
        import subprocess
        res = subprocess.run(
            [sys.executable, str(script_path), title, initial_dir or (str(WORKSPACE_ROOT) if WORKSPACE_ROOT else str(PROJECT_ROOT))],
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


@app.post("/api/workspace/browse")
async def browse_workspace() -> Dict[str, Any]:
    """Opens native File Explorer to select a workspace folder."""
    folder = await asyncio.to_thread(
        ask_directory_dialog,
        "Select Project Workspace Folder for Ura-Shree",
        str(WORKSPACE_ROOT) if WORKSPACE_ROOT else str(PROJECT_ROOT),
    )
    if not folder:
        return {"ok": False, "cancelled": True, "workspace": str(WORKSPACE_ROOT) if WORKSPACE_ROOT else None}

    res = select_workspace(WorkspaceSelectRequest(path=folder))
    return res


@app.post("/api/browse-directory")
async def browse_directory() -> Dict[str, Any]:
    """Opens native File Explorer to select a directory for terminal or file viewing."""
    folder = await asyncio.to_thread(
        ask_directory_dialog,
        "Select Working Directory",
        str(WORKSPACE_ROOT) if WORKSPACE_ROOT else str(PROJECT_ROOT),
    )
    if not folder:
        return {"ok": False, "cancelled": True, "path": None}
    return {"ok": True, "path": folder}


@app.get("/api/workspace/suggestions")
def get_workspace_suggestions() -> Dict[str, Any]:
    home = Path.home()
    return {
        "workspace": str(DEFAULT_WORKSPACE) if DEFAULT_WORKSPACE.exists() else None,
        "project_root": str(PROJECT_ROOT),
        "desktop": str(home / "Desktop") if (home / "Desktop").exists() else None,
        "documents": str(home / "Documents") if (home / "Documents").exists() else None,
        "downloads": str(home / "Downloads") if (home / "Downloads").exists() else None,
    }




@app.get("/api/tree")
def get_file_tree(max_entries: int = 4000) -> Dict[str, Any]:
    """The workspace as a nested tree for the explorer."""
    if not WORKSPACE_ROOT:
        return {"tree": None, "truncated": False, "workspace": None}

    counter = {"n": 0}

    def build(path: Path) -> Optional[Dict[str, Any]]:
        if counter["n"] >= max_entries:
            return None
        counter["n"] += 1

        node: Dict[str, Any] = {
            "name": path.name or str(path),
            "path": str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "isDir": path.is_dir(),
        }
        if path.is_dir():
            children = []
            try:
                entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except (PermissionError, OSError):
                entries = []
            for entry in entries:
                if entry.name in IGNORED_DIRS:
                    continue
                if WORKSPACE_ROOT == PROJECT_ROOT and not os.environ.get("SHREE_DEV"):
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

    root = build(WORKSPACE_ROOT) or {"name": WORKSPACE_ROOT.name, "path": "", "isDir": True, "children": []}
    root["path"] = ""
    return {"tree": root, "truncated": counter["n"] >= max_entries, "workspace": str(WORKSPACE_ROOT)}


@app.get("/api/file")
def read_file(path: str = Query(...)) -> Dict[str, Any]:
    if not fs_tool:
        raise HTTPException(status_code=400, detail="No workspace folder open.")
    try:
        result = fs_tool.read(path)
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err))
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "read failed"))
    return result


@app.post("/api/file")
@app.put("/api/file")
def write_file(req: FileWriteRequest) -> Dict[str, Any]:
    if not fs_tool or not WORKSPACE_ROOT:
        raise HTTPException(status_code=400, detail="No workspace folder open.")
    try:
        if time_machine:
            time_machine.snapshot(f"Before editing {req.path}", kind="auto")
        result = fs_tool.write(req.path, req.content)
    except PermissionError as err:
        raise HTTPException(status_code=403, detail=str(err))
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "write failed"))
    return result


@app.delete("/api/file")
def delete_file(path: str = Query(...)) -> Dict[str, Any]:
    if not WORKSPACE_ROOT:
        raise HTTPException(status_code=400, detail="No workspace folder open.")
    rel = safe_relative_path(path)
    if not rel:
        raise HTTPException(status_code=400, detail="Invalid path.")
    target = (WORKSPACE_ROOT / rel).resolve()
    if not str(target).startswith(str(WORKSPACE_ROOT)):
        raise HTTPException(status_code=403, detail="Path outside workspace.")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File or directory not found.")

    try:
        if time_machine:
            time_machine.snapshot(f"Before deleting {rel}", kind="auto")
    except Exception:
        pass

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"ok": True, "deleted": rel}


# -- skills -------------------------------------------------------------------

@app.get("/api/skills")
def get_skills() -> List[Dict[str, Any]]:
    return skills_mgr.list_skills()


@app.post("/api/skills")
def add_skill(req: SkillCreateRequest) -> Dict[str, Any]:
    if not req.name.strip() or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Skill name and prompt are required.")
    return skills_mgr.add_skill(req.name, req.description, req.prompt)


@app.patch("/api/skills/{skill_id}")
def update_skill(skill_id: str, req: SkillToggleRequest) -> Dict[str, Any]:
    updated = skills_mgr.toggle_skill(skill_id, req.enabled)
    if not updated:
        raise HTTPException(status_code=404, detail="Skill not found.")
    return updated


@app.delete("/api/skills/{skill_id}")
def remove_skill(skill_id: str) -> Dict[str, Any]:
    ok = skills_mgr.delete_skill(skill_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot delete built-in skill or skill not found.")
    return {"ok": True}


@app.post("/api/upload")
def upload_files(req: UploadRequest) -> Dict[str, Any]:
    """
    Accepts files or a whole folder from the browser.

    Content arrives base64-encoded in JSON, which keeps the dependency list
    short and handles both the file picker and a directory drop the same way.
    """
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
            destination = fs_tool._resolve_safe_path(relative)
        except PermissionError as err:
            rejected.append({"path": item.path, "reason": str(err)})
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        written.append({
            "path": str(destination.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "bytes": len(data),
        })

    if written and time_machine:
        time_machine.snapshot(f"Uploaded {len(written)} file(s)", kind="upload")
    return {"written": written, "rejected": rejected, "count": len(written)}


@app.get("/api/memory")
def get_memory() -> Dict[str, Any]:
    return {
        "summary": memory_store.get_summary_context(),
        "facts": memory_store.get_facts_by_category(),
        "decisions": memory_store.get_recent_decisions(limit=25),
    }


@app.post("/api/index")
def build_index() -> Dict[str, Any]:
    if not indexer:
        return {"stats": {}, "symbols": [], "tree_summary": "No workspace active."}
    stats = indexer.scan_and_index()
    symbols = [
        {
            "file": file_path,
            "name": sym["name"],
            "kind": sym.get("kind", "symbol"),
            "line": sym["line"],
            "signature": sym.get("signature", ""),
            "doc": sym.get("doc", ""),
        }
        for file_path, sym_list in indexer.file_symbols.items()
        for sym in sym_list
    ]
    return {"stats": stats, "symbols": symbols, "tree_summary": indexer.get_tree_summary()}


@app.get("/api/git/status")
def git_status() -> Dict[str, Any]:
    if not git_tool:
        return {"status": "No workspace active", "diff": ""}
    return {"status": git_tool.status(), "diff": git_tool.diff()}


# -- time machine -------------------------------------------------------------

@app.get("/api/timemachine")
def get_timeline(limit: int = 200) -> Dict[str, Any]:
    if not time_machine:
        return {"nodes": [], "head": None, "store_bytes": 0}
    return time_machine.timeline(limit=limit)


@app.post("/api/timemachine/snapshot")
def create_snapshot(req: SnapshotRequest) -> Dict[str, Any]:
    if not time_machine:
        raise HTTPException(status_code=400, detail="No active workspace for snapshots.")
    return time_machine.snapshot(req.label, kind="manual")


@app.get("/api/timemachine/diff")
def snapshot_diff(from_id: str = Query(...), to_id: str = Query(...)) -> Dict[str, Any]:
    if not time_machine:
        raise HTTPException(status_code=400, detail="No active workspace for snapshots.")
    try:
        return time_machine.diff(from_id, to_id)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.get("/api/timemachine/file")
def snapshot_file(snapshot_id: str = Query(...), path: str = Query(...)) -> Dict[str, Any]:
    if not time_machine:
        raise HTTPException(status_code=400, detail="No active workspace for snapshots.")
    try:
        return time_machine.file_at(snapshot_id, path)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.post("/api/timemachine/restore")
def restore_snapshot(req: RestoreRequest) -> Dict[str, Any]:
    if not time_machine:
        raise HTTPException(status_code=400, detail="No active workspace for snapshots.")
    try:
        return time_machine.restore(req.snapshot_id, dry_run=req.dry_run)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.post("/api/timemachine/prune")
def prune_snapshots(keep: int = 100) -> Dict[str, Any]:
    if not time_machine:
        return {"removed": 0}
    return time_machine.prune(keep=keep)


# -- terminal -----------------------------------------------------------------

@app.get("/api/terminal/sessions")
def terminal_sessions() -> Dict[str, Any]:
    return {"sessions": shells.list_sessions()}


@app.post("/api/terminal")
def run_command(req: CommandRequest) -> Dict[str, Any]:
    """Blocking command execution, for callers that do not want the websocket."""
    return shells.get(req.session_id).run(req.command, timeout=float(req.timeout))


@app.delete("/api/terminal/{session_id}")
def close_terminal(session_id: str) -> Dict[str, Any]:
    return {"closed": shells.close(session_id)}


@app.websocket("/ws/terminal")
async def terminal_socket(websocket: WebSocket) -> None:
    """
    Live terminal. Output streams line by line as the command produces it,
    instead of arriving in one block when the process exits.
    """
    await websocket.accept()
    session_id = websocket.query_params.get("session", "default")
    session = shells.get(session_id)
    await websocket.send_json({"type": "ready", "info": session.info()})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON."})
                continue

            action = message.get("action", "run")

            if action == "run":
                command = (message.get("command") or "").strip()
                if not command:
                    continue

                blocked = check_command(command)
                if blocked:
                    await websocket.send_json({
                        "type": "exit", "code": -1, "success": False,
                        "error": f"Refused: matches a blocked pattern ({blocked}).",
                    })
                    continue

                await websocket.send_json({"type": "started", "command": command})
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def pump() -> None:
                    for event in session.stream(command, timeout=float(message.get("timeout", 600))):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    loop.call_soon_threadsafe(queue.put_nowait, None)

                asyncio.create_task(asyncio.to_thread(pump))
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    await websocket.send_json(event)

            elif action == "restart":
                session.restart()
                await websocket.send_json({"type": "ready", "info": session.info()})

            elif action == "info":
                await websocket.send_json({"type": "info", "info": session.info()})

            elif action == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        return
    except Exception as err:
        try:
            await websocket.send_json({"type": "error", "message": str(err)})
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
    await websocket.accept()
    session_id = websocket.query_params.get("session", "default")
    running: Optional[asyncio.Task] = None
    pending_approvals: Dict[str, asyncio.Future[bool]] = {}

    async def handle_approval(call: Any) -> bool:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        pending_approvals[call.id] = fut
        try:
            await websocket.send_json({
                "type": "tool_approval_prompt",
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            })
            approved = await asyncio.wait_for(fut, timeout=300)
            return approved
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        finally:
            pending_approvals.pop(call.id, None)

    async def drive(message: Dict[str, Any]) -> None:
        active = settings.get("active", {})
        provider_id = message.get("provider") or active.get("provider", "local")
        model = message.get("model") or active.get("model") or ""

        if not model and provider_id != "local":
            await websocket.send_json({
                "type": "error",
                "message": f"No model selected for {provider_id}. Scan models and pick one in Settings.",
            })
            return

        try:
            provider = provider_for(provider_id)
        except ProviderError as err:
            await websocket.send_json({"type": "error", "message": str(err), "status": err.status})
            return

        agent = get_agent(session_id)
        await websocket.send_json({
            "type": "run_start", "provider": provider_id, "model": model or "(local checkpoint)",
        })

        try:
            async for event in agent.stream_task(
                request=message.get("message", ""),
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
                await websocket.send_json(event)
        except asyncio.CancelledError:
            await websocket.send_json({"type": "cancelled"})
            raise
        except Exception as err:
            await websocket.send_json({"type": "error", "message": str(err)})
        finally:
            await provider.aclose()

    try:
        while True:
            raw = await websocket.receive_text()
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
                get_agent(session_id).reset()
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
        print(f"[agent socket] {err}")


@app.exception_handler(ProviderError)
async def provider_error_handler(_request, exc: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=exc.status or 502, content={"detail": str(exc)})


# The compiled frontend is mounted last so it never shadows an /api route.
_dist = PROJECT_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
