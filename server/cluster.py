"""
ClusterOne integration: run Shree fine-tunes on a lab GPU cluster.

ClusterOne is configured once, as the `clusterone` provider: its base URL is
`https://<cluster>/api/v1/openai/v1` and its key is the ClusterOne API key. The
same credential drives two things:

  * chat - the provider catalogue talks to the cluster's OpenAI-compatible
    endpoint directly, so `clusterone connect` (and its wildcard-CORS proxy)
    is never needed;
  * jobs - this router wraps the `clusterone` SDK (pip install -e ../clausterone)
    to list GPU nodes, submit fine-tune jobs and sweeps, stream logs, cancel
    and download outputs.

Clients never send a shell command. They pick a preset and numbers, and the
command is assembled here from validated values only.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from server.auth import Principal, current_principal
from server.ratelimit import rate_limit

PROVIDER_ID = "clusterone"
OPENAI_SUFFIX = "/api/v1/openai/v1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
DOWNLOADS = PROJECT_ROOT / "outputs" / "cluster"

# What a training job needs on the cluster. Everything else in the repository
# (frontend, other checkpoints, kaggle or local outputs, workspaces) stays home.
BUNDLE_DIRS = ("model", "training", "data", "tokenizer", "configs", "datasets")
BUNDLE_FILES = ("requirements.txt",)
BUNDLE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")

JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_SWEEP = 6
MAX_OUTPUT_BYTES = 8 * 2**30  # refuse to unpack more than 8 GB from one job


def _sdk():
    try:
        from clusterone.client import Client, ClusterOneError
    except ImportError:
        raise HTTPException(
            503,
            "The ClusterOne SDK is not installed in this environment. "
            "Run: pip install -e ../clausterone",
        ) from None
    return Client, ClusterOneError


def cluster_root(base_url: str) -> str:
    """`https://host/api/v1/openai/v1` (or just `https://host`) -> `https://host`."""
    url = (base_url or "").strip().rstrip("/")
    if url.endswith(OPENAI_SUFFIX):
        url = url[: -len(OPENAI_SUFFIX)]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return url


def list_base_checkpoints() -> List[str]:
    return sorted(p.name for p in CHECKPOINTS.glob("*.pt") if p.is_file())


def build_bundle(dest: Path, base_checkpoint: str) -> Path:
    """Copies only what train_coding.py needs into `dest` and returns it."""
    for name in BUNDLE_DIRS:
        src = PROJECT_ROOT / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=BUNDLE_IGNORE)
    for name in BUNDLE_FILES:
        src = PROJECT_ROOT / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    (dest / "checkpoints").mkdir()
    shutil.copy2(CHECKPOINTS / base_checkpoint, dest / "checkpoints" / base_checkpoint)
    tokenizer = CHECKPOINTS / "tokenizer.json"
    if tokenizer.is_file():
        shutil.copy2(tokenizer, dest / "checkpoints" / "tokenizer.json")
    return dest


def train_command(req: "FineTuneRequest", lr: float, steps: int) -> str:
    # Every interpolated value is validated by FineTuneRequest: enum, number or
    # a filename that exists in checkpoints/.
    return (
        f"python training/train_coding.py --config configs/{req.config}.yaml "
        f"--base checkpoints/{req.base} --steps {steps} --lr {lr:g} "
        f"--output outputs/coding_best.pt"
    )


def safe_extract(archive: Path, dest: Path) -> None:
    """Unpacks a job's outputs, refusing zip bombs and paths that escape `dest`."""
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > MAX_OUTPUT_BYTES:
            raise HTTPException(413, "Job outputs are larger than 8 GB; download them with the clusterone CLI.")
        root = dest.resolve()
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if not target.is_relative_to(root):
                raise HTTPException(422, f"Refusing unsafe path in job outputs: {info.filename}")
        z.extractall(dest)


def _job_public(job: Any) -> Dict[str, Any]:
    d = dict(job.data)
    return {
        "id": job.id,
        "name": job.name,
        "status": job.status,
        "url": job.url,
        "created_at": d.get("created_at") or d.get("submitted_at"),
        "gpus": d.get("gpus"),
        "nodes": d.get("nodes"),
        "command": d.get("command"),
        "error": d.get("error") or d.get("message"),
    }


class FineTuneRequest(BaseModel):
    name: str = Field("shree-finetune", min_length=1, max_length=80)
    config: Literal["small", "medium", "large"] = "small"
    base: str = "best.pt"
    steps: int = Field(600, ge=10, le=200_000)
    lr: float = Field(2.5e-4, gt=0, le=1e-1)
    gpus: int = Field(1, ge=1, le=8)
    time_min: int = Field(120, ge=5, le=7 * 24 * 60)
    env: str = Field("pytorch-cu128", pattern=r"^[A-Za-z0-9._-]{1,64}$")
    sweep_lr: List[float] = Field(default_factory=list, max_length=MAX_SWEEP)
    sweep_steps: List[int] = Field(default_factory=list, max_length=MAX_SWEEP)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9 ._-]+", v):
            raise ValueError("Use letters, numbers, spaces, dots, dashes or underscores.")
        return v.strip()

    @field_validator("base")
    @classmethod
    def _base(cls, v: str) -> str:
        if v not in list_base_checkpoints():
            raise ValueError(f"Unknown checkpoint {v!r}. Pick one from checkpoints/.")
        return v

    @field_validator("sweep_lr")
    @classmethod
    def _sweep_lr(cls, v: List[float]) -> List[float]:
        if any(not (0 < x <= 1e-1) for x in v):
            raise ValueError("Sweep learning rates must be between 0 and 0.1.")
        return v

    @field_validator("sweep_steps")
    @classmethod
    def _sweep_steps(cls, v: List[int]) -> List[int]:
        if any(not (10 <= x <= 200_000) for x in v):
            raise ValueError("Sweep steps must be between 10 and 200000.")
        return v


def make_router(settings_for: Callable[[Principal], Any]) -> APIRouter:
    router = APIRouter(prefix="/api/cluster", tags=["cluster"])

    def client_for(principal: Principal):
        Client, _ = _sdk()
        s = settings_for(principal)
        root = cluster_root(s.base_url(PROVIDER_ID))
        key = s.api_key(PROVIDER_ID)
        if not root or not key:
            raise HTTPException(
                409,
                "ClusterOne is not set up. In Settings > Models and keys pick ClusterOne, "
                f"enter https://<your-cluster>{OPENAI_SUFFIX} and your API key.",
            )
        return Client(url=root, key=key)

    async def call(fn: Callable[[], Any]) -> Any:
        """Runs a blocking SDK call off the event loop and maps its errors."""
        _, ClusterOneError = _sdk()
        try:
            return await asyncio.to_thread(fn)
        except ClusterOneError as e:
            status = e.status if 400 <= e.status < 600 else 502
            if status in (401, 403):
                raise HTTPException(status, "ClusterOne rejected the API key. Check it in Settings.") from None
            raise HTTPException(status, f"ClusterOne: {e}") from None

    def job_id(value: str) -> str:
        if not JOB_ID.match(value):
            raise HTTPException(422, "Invalid job id.")
        return value

    @router.get("/status")
    async def status(principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
        try:
            _sdk()
        except HTTPException as e:
            return {"installed": False, "configured": False, "detail": e.detail}
        s = settings_for(principal)
        root = cluster_root(s.base_url(PROVIDER_ID))
        if not root or not s.api_key(PROVIDER_ID):
            return {"installed": True, "configured": False, "checkpoints": list_base_checkpoints()}
        c = client_for(principal)
        me = await call(c.me)
        usage = await call(c.usage)
        return {
            "installed": True, "configured": True, "url": root, "me": me, "usage": usage,
            "checkpoints": list_base_checkpoints(),
        }

    @router.get("/nodes")
    async def nodes(principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
        c = client_for(principal)
        return {"nodes": await call(c.nodes)}

    @router.get("/jobs")
    async def jobs(
        status: str = Query("", max_length=32, pattern=r"^[a-z_]*$"),
        principal: Principal = Depends(current_principal),
    ) -> Dict[str, Any]:
        c = client_for(principal)
        found = await call(lambda: c.jobs(status=status, mine=True, page_size=50))
        return {"jobs": [_job_public(j) for j in found]}

    @router.get("/jobs/{jid}")
    async def job(jid: str, principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
        c = client_for(principal)
        return _job_public(await call(lambda: c.job(job_id(jid))))

    @router.get("/jobs/{jid}/log")
    async def job_log(
        jid: str,
        offset: int = Query(0, ge=0),
        stream: Literal["stdout", "stderr"] = "stdout",
        principal: Principal = Depends(current_principal),
    ) -> Dict[str, Any]:
        c = client_for(principal)
        j = await call(lambda: c.job(job_id(jid)))
        text, next_offset, state = await call(lambda: j.log(stream, offset))
        return {"text": text, "offset": next_offset, "status": state}

    @router.post("/jobs/{jid}/cancel", dependencies=[Depends(rate_limit("cluster", "rate_limit_scan"))])
    async def job_cancel(jid: str, principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
        c = client_for(principal)
        j = await call(lambda: c.job(job_id(jid)))
        await call(j.cancel)
        return {"cancelled": True}

    @router.post("/jobs/{jid}/download", dependencies=[Depends(rate_limit("cluster", "rate_limit_scan"))])
    async def job_download(jid: str, principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
        c = client_for(principal)
        j = await call(lambda: c.job(job_id(jid)))
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", j.name)[:60] or "job"
        dest = DOWNLOADS / f"{safe_name}-{j.id[:8]}"
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            archive = await call(lambda: j.download(tmp, extract=False))
            await asyncio.to_thread(safe_extract, Path(archive), dest)
        return {"path": str(dest.relative_to(PROJECT_ROOT)).replace("\\", "/")}

    @router.post("/finetune", dependencies=[Depends(rate_limit("cluster", "rate_limit_scan"))])
    async def finetune(req: FineTuneRequest, principal: Principal = Depends(current_principal)) -> Dict[str, Any]:
        c = client_for(principal)
        lrs = req.sweep_lr or [req.lr]
        steps = req.sweep_steps or [req.steps]
        if len(lrs) * len(steps) > MAX_SWEEP:
            raise HTTPException(422, f"A sweep can launch at most {MAX_SWEEP} jobs.")
        opts = dict(env=req.env, gpus=req.gpus, time_min=req.time_min, cpus=4, ram_mb=16384)

        def submit_all() -> List[Any]:
            with tempfile.TemporaryDirectory(prefix="shree-cluster-") as tmp:
                bundle = build_bundle(Path(tmp), req.base)
                out = []
                for lr in lrs:
                    for n in steps:
                        tag = f"-lr{lr:g}-s{n}" if len(lrs) * len(steps) > 1 else ""
                        out.append(c.submit(bundle, train_command(req, lr, n),
                                            name=f"{req.name}{tag}"[:120], **opts))
                return out

        submitted = await call(submit_all)
        return {"jobs": [_job_public(j) for j in submitted]}

    return router
