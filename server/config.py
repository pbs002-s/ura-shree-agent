"""
One place that reads the environment.

Every knob a deployment needs is here, resolved once at import. Reading
`os.environ` scattered across modules is how a production setting ends up
honoured in one code path and ignored in the next, so nothing else in the
server package touches the environment directly.

Defaults are the single-user local values the tool has always used, so running
without an env file keeps the developer experience unchanged. Production is
opt-in through explicit variables.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _list(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default or [])
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


# Regenerated on every process start, which is the correct behaviour for a
# single-user dev box: restarting the server logs you out and nothing else.
_EPHEMERAL_DEV_SECRET = secrets.token_urlsafe(48)


# Not frozen: a test needs to flip `mode` or a limit for one case, and the
# alternative - reloading the module and every module that imported the
# singleton - is far more fragile than a plain attribute.
@dataclass
class Config:
    """Deployment configuration, resolved from the environment."""

    # -- mode ---------------------------------------------------------------
    # "local" keeps the single-user behaviour: no login, host execution.
    # "cloud" turns on authentication, the database and container sandboxes.
    mode: str = field(default_factory=lambda: os.environ.get("SHREE_MODE", "local").strip().lower())

    # -- identity -----------------------------------------------------------
    jwt_secret: str = field(default_factory=lambda: os.environ.get("SHREE_JWT_SECRET", "").strip())
    jwt_algorithm: str = field(default_factory=lambda: os.environ.get("SHREE_JWT_ALG", "HS256"))
    access_token_ttl: int = field(default_factory=lambda: _int("SHREE_ACCESS_TOKEN_TTL", 3600))
    refresh_token_ttl: int = field(default_factory=lambda: _int("SHREE_REFRESH_TOKEN_TTL", 30 * 86400))
    oauth_github_client_id: str = field(default_factory=lambda: os.environ.get("SHREE_GITHUB_CLIENT_ID", ""))
    oauth_github_client_secret: str = field(default_factory=lambda: os.environ.get("SHREE_GITHUB_CLIENT_SECRET", ""))
    oauth_google_client_id: str = field(default_factory=lambda: os.environ.get("SHREE_GOOGLE_CLIENT_ID", ""))
    oauth_google_client_secret: str = field(default_factory=lambda: os.environ.get("SHREE_GOOGLE_CLIENT_SECRET", ""))
    oauth_redirect_base: str = field(default_factory=lambda: os.environ.get("SHREE_OAUTH_REDIRECT_BASE", ""))

    # Legacy shared-secret header, kept so existing local scripts keep working.
    legacy_api_key: str = field(default_factory=lambda: os.environ.get("SHREE_API_KEY", "").strip())

    # -- storage ------------------------------------------------------------
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "SHREE_DATABASE_URL", "sqlite:///" + (PROJECT_ROOT / ".shree" / "shree.db").as_posix()
        )
    )
    redis_url: str = field(default_factory=lambda: os.environ.get("SHREE_REDIS_URL", "").strip())
    object_store_url: str = field(default_factory=lambda: os.environ.get("SHREE_OBJECT_STORE_URL", "").strip())
    s3_endpoint_url: str = field(default_factory=lambda: os.environ.get("SHREE_S3_ENDPOINT_URL", "").strip())
    s3_region: str = field(default_factory=lambda: os.environ.get("SHREE_S3_REGION", "us-east-1"))

    # Root under which every tenant workspace volume is created.
    workspaces_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SHREE_WORKSPACES_ROOT", str(PROJECT_ROOT / "workspaces"))
        ).resolve()
    )

    # -- execution ----------------------------------------------------------
    # "local" runs on the host; "container" gives every session its own sandbox.
    execution_driver: str = field(
        default_factory=lambda: os.environ.get("SHREE_EXECUTION_DRIVER", "").strip().lower()
    )
    sandbox_image: str = field(default_factory=lambda: os.environ.get("SHREE_SANDBOX_IMAGE", "shree-sandbox:latest"))
    sandbox_cpus: float = field(default_factory=lambda: float(os.environ.get("SHREE_SANDBOX_CPUS", "1.0")))
    sandbox_memory_mb: int = field(default_factory=lambda: _int("SHREE_SANDBOX_MEMORY_MB", 2048))
    sandbox_pids_limit: int = field(default_factory=lambda: _int("SHREE_SANDBOX_PIDS", 256))
    # Egress is closed by default. A sandbox that can reach the internet can
    # exfiltrate whatever the agent just read.
    sandbox_network: str = field(default_factory=lambda: os.environ.get("SHREE_SANDBOX_NETWORK", "none"))
    session_idle_timeout: int = field(default_factory=lambda: _int("SHREE_SESSION_IDLE_TIMEOUT", 1800))
    reaper_interval: int = field(default_factory=lambda: _int("SHREE_REAPER_INTERVAL", 120))

    # -- limits -------------------------------------------------------------
    rate_limit_scan: str = field(default_factory=lambda: os.environ.get("SHREE_RATE_SCAN", "10/60"))
    rate_limit_chat: str = field(default_factory=lambda: os.environ.get("SHREE_RATE_CHAT", "60/60"))
    rate_limit_ws: str = field(default_factory=lambda: os.environ.get("SHREE_RATE_WS", "120/60"))
    rate_limit_api: str = field(default_factory=lambda: os.environ.get("SHREE_RATE_API", "600/60"))

    # -- web ----------------------------------------------------------------
    allowed_origins: List[str] = field(default_factory=lambda: _list("SHREE_ALLOWED_ORIGINS"))
    # Proxy-aware client IP only when a reverse proxy is actually in front:
    # trusting X-Forwarded-For unconditionally lets a caller forge their own
    # rate-limit bucket.
    trust_proxy_headers: bool = field(default_factory=lambda: _bool("SHREE_TRUST_PROXY", False))
    allow_host_directory_picker: bool = field(
        default_factory=lambda: _bool("SHREE_ALLOW_HOST_PICKER", False)
    )
    # The first account is always allowed, so a new deployment can be claimed.
    # Turning this off after that makes accounts invite-only.
    allow_signup: bool = field(default_factory=lambda: _bool("SHREE_ALLOW_SIGNUP", True))

    # -- observability ------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.environ.get("SHREE_LOG_LEVEL", "INFO").upper())
    log_json: bool = field(default_factory=lambda: _bool("SHREE_LOG_JSON", False))
    otel_endpoint: str = field(default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())
    service_name: str = field(default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "ura-shree"))

    @property
    def cloud(self) -> bool:
        return self.mode == "cloud"

    @property
    def driver_kind(self) -> str:
        """Container sandboxes are the default in cloud mode, host in local."""
        if self.execution_driver:
            return self.execution_driver
        return "container" if self.cloud else "local"

    @property
    def default_origins(self) -> List[str]:
        if self.allowed_origins:
            return self.allowed_origins
        return [
            "http://127.0.0.1:8000", "http://localhost:8000",
            "http://127.0.0.1:5173", "http://localhost:5173",
        ]

    def resolved_jwt_secret(self) -> str:
        """
        The signing key.

        Cloud mode refuses to start without an explicit secret: a generated one
        would silently invalidate every token whenever a worker restarts, and
        with several Gunicorn workers no two of them would ever agree on it.
        """
        if self.jwt_secret:
            return self.jwt_secret
        if self.cloud:
            raise RuntimeError(
                "SHREE_JWT_SECRET must be set in cloud mode. Generate one with: "
                "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return _EPHEMERAL_DEV_SECRET


config = Config()
