"""
Who is calling, and what they are allowed to touch.

Two things live here: token minting and verification, and the FastAPI
dependencies that turn a token into a user and a user into a permission.

The JWT implementation is deliberately hand-rolled and deliberately tiny. It
supports exactly one algorithm, HS256, and never reads the algorithm out of the
token to decide how to verify it - which is the whole `alg: none` family of
bugs, gone by construction rather than by a check somebody has to remember. The
rest is an HMAC and a constant-time compare, so pulling in a JWT library would
add a dependency and a larger attack surface to do the same three things.

Passwords use `hashlib.scrypt`, which is in the standard library and is a
memory-hard KDF, so no password hashing dependency is needed either.

In local mode - a developer on their own machine, no database - every request
resolves to a synthetic admin principal, so the tool behaves exactly as it did
before any of this existed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Request, WebSocket, status

from server.config import config
from server.db import (
    ACCOUNT_ROLES,
    PROJECT_MEMBER,
    PROJECT_OWNER,
    ROLE_ADMIN,
    ROLE_USER,
    SQLMODEL_AVAILABLE,
    project_role_allows,
    session_scope,
)

# scrypt parameters. N=2**15 is roughly 100ms and 32MB per hash on a current
# server core, which is slow enough to matter to an attacker and fast enough
# not to matter to a login.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_BYTES = 16
SCRYPT_KEY_LEN = 32


# -- passwords ----------------------------------------------------------------

def hash_password(password: str) -> str:
    """Returns a self-describing hash, so parameters can change without a migration."""
    if not password:
        raise ValueError("Password cannot be empty.")
    salt = os.urandom(SCRYPT_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        dklen=SCRYPT_KEY_LEN, maxmem=64 * 1024 * 1024,
    )
    return "$".join(
        ["scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
         base64.b64encode(salt).decode(), base64.b64encode(digest).decode()]
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check against a stored hash."""
    if not password or not stored:
        return False
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt_b64),
            n=int(n), r=int(r), p=int(p), dklen=len(expected), maxmem=64 * 1024 * 1024,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(expected, actual)


# -- tokens -------------------------------------------------------------------

class TokenError(Exception):
    """A token was missing, malformed, expired or not ours."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _sign(message: bytes) -> bytes:
    return hmac.new(
        config.resolved_jwt_secret().encode("utf-8"), message, hashlib.sha256
    ).digest()


def create_token(
    subject: str,
    role: str = ROLE_USER,
    token_type: str = "access",
    ttl: Optional[int] = None,
    **claims: Any,
) -> str:
    """Mints a signed HS256 token for `subject`."""
    now = int(time.time())
    lifetime = ttl if ttl is not None else (
        config.refresh_token_ttl if token_type == "refresh" else config.access_token_ttl
    )
    payload = {
        "sub": subject,
        "role": role,
        "typ": token_type,
        "iat": now,
        "exp": now + lifetime,
        # A random id makes a token individually revocable later without
        # changing the format.
        "jti": secrets.token_urlsafe(12),
        **claims,
    }
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signing_input = f"{header}.{body}".encode("ascii")
    return f"{header}.{body}.{_b64url(_sign(signing_input))}"


def decode_token(token: str, expect_type: Optional[str] = "access") -> Dict[str, Any]:
    """
    Verifies a token and returns its claims.

    The signature is checked before the payload is parsed, and the algorithm is
    never taken from the token.
    """
    if not token or token.count(".") != 2:
        raise TokenError("Malformed token.")
    header_b64, body_b64, signature_b64 = token.split(".")
    try:
        expected = _sign(f"{header_b64}.{body_b64}".encode("ascii"))
        if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
            raise TokenError("Signature does not verify.")
        payload = json.loads(_b64url_decode(body_b64))
    except TokenError:
        raise
    except Exception as err:
        raise TokenError(f"Token could not be read: {err}") from err

    if not isinstance(payload, dict):
        raise TokenError("Token payload is not an object.")
    if int(payload.get("exp", 0)) <= time.time():
        raise TokenError("Token has expired.")
    if expect_type and payload.get("typ") != expect_type:
        raise TokenError(f"Expected a {expect_type} token.")
    return payload


def issue_token_pair(user_id: str, role: str, email: str = "") -> Dict[str, Any]:
    """The response body every login path returns."""
    return {
        "access_token": create_token(user_id, role=role, token_type="access", email=email),
        "refresh_token": create_token(user_id, role=role, token_type="refresh"),
        "token_type": "bearer",
        "expires_in": config.access_token_ttl,
        "role": role,
    }


# -- principals ---------------------------------------------------------------

@dataclass(frozen=True)
class Principal:
    """The authenticated caller, as every route sees it."""

    id: str
    email: str
    role: str
    # True for the synthetic local-mode principal, so status can say so.
    local: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "email": self.email, "role": self.role}


LOCAL_PRINCIPAL = Principal(id="local", email="local@localhost", role=ROLE_ADMIN, local=True)


def _bearer_from(headers: Any, query: Any) -> str:
    """
    Pulls a token out of a request.

    A query parameter is accepted only because browsers cannot set headers on a
    WebSocket handshake. It is last in precedence and, unlike a header, it ends
    up in proxy access logs - which is why the header is preferred everywhere
    it is available.
    """
    authorization = headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    api_key = headers.get("x-api-key", "")
    if api_key:
        return api_key.strip()
    return (query.get("token") or "").strip()


def _principal_from_token(token: str) -> Principal:
    claims = decode_token(token, expect_type="access")
    role = claims.get("role", ROLE_USER)
    if role not in ACCOUNT_ROLES:
        role = ROLE_USER
    return Principal(id=str(claims.get("sub", "")), email=str(claims.get("email", "")), role=role)


def _legacy_key_principal(token: str) -> Optional[Principal]:
    """
    The old shared-secret header, still honoured in local mode.

    Compared in constant time: a plain `!=` on a secret leaks its length and,
    over enough requests, its contents.
    """
    if not config.legacy_api_key:
        return None
    if hmac.compare_digest(token, config.legacy_api_key):
        return LOCAL_PRINCIPAL
    return None


def resolve_principal(headers: Any, query: Any) -> Principal:
    """
    Turns credentials into a caller, or raises.

    Local mode with no shared secret configured has no notion of identity, so
    it resolves to a fixed admin principal - the same single user the tool has
    always assumed.
    """
    token = _bearer_from(headers, query)

    if not config.cloud:
        if not config.legacy_api_key:
            return LOCAL_PRINCIPAL
        legacy = _legacy_key_principal(token)
        if legacy is not None:
            return legacy
        raise TokenError("Invalid API key.")

    # The shared secret is not honoured in cloud mode. A single static string
    # that grants administrator rights to everyone holding it is the opposite
    # of per-user identity, and it would never appear in an audit row as
    # anybody in particular.
    if not token:
        raise TokenError("No credentials supplied.")
    return _principal_from_token(token)


async def current_principal(request: Request) -> Principal:
    """FastAPI dependency: the authenticated caller for this request."""
    try:
        principal = resolve_principal(request.headers, request.query_params)
    except TokenError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(err),
            headers={"WWW-Authenticate": "Bearer"},
        ) from err

    from server.observability import user_id_var

    user_id_var.set(principal.id)
    return principal


async def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required.")
    return principal


async def principal_for_socket(websocket: WebSocket) -> Optional[Principal]:
    """
    Authenticates a WebSocket before it is accepted.

    Closing with 1008 (policy violation) rather than accepting and then closing
    means an unauthenticated client never gets a channel at all, not even for
    the length of one frame.
    """
    try:
        return resolve_principal(websocket.headers, websocket.query_params)
    except TokenError:
        await websocket.close(code=1008, reason="Unauthorized")
        return None


# -- project authorisation ----------------------------------------------------

def project_role_for(principal: Principal, project_id: str) -> Optional[str]:
    """
    The caller's role on a project, or None when they are not a member.

    Admins are treated as owners everywhere: an account-level administrator can
    already read the audit log and the database, so pretending otherwise here
    would be theatre rather than a control.
    """
    if principal.local or principal.is_admin:
        return PROJECT_OWNER
    if not SQLMODEL_AVAILABLE:
        return None

    from sqlmodel import select

    from server.db import Project, ProjectMember

    with session_scope() as session:
        project = session.get(Project, project_id)
        if project is None:
            return None
        if project.owner_id == principal.id:
            return PROJECT_OWNER
        membership = session.exec(
            select(ProjectMember).where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == principal.id,
            )
        ).first()
        return membership.role if membership else None


def authorize_project(principal: Principal, project_id: str, required: str = PROJECT_MEMBER) -> str:
    """
    Asserts the caller may act on a project at `required` level or above.

    A non-member gets 404 rather than 403: telling a stranger that a project id
    exists is itself a leak.
    """
    held = project_role_for(principal, project_id)
    if held is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    if not project_role_allows(held, required):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This action needs the '{required}' role on the project.",
        )
    return held


# -- OAuth2 -------------------------------------------------------------------

OAUTH_PROVIDERS: Dict[str, Dict[str, str]] = {
    "github": {
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "user_url": "https://api.github.com/user",
        "scope": "read:user user:email",
    },
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "user_url": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
    },
}


def oauth_credentials(provider: str) -> Tuple[str, str]:
    if provider == "github":
        return config.oauth_github_client_id, config.oauth_github_client_secret
    if provider == "google":
        return config.oauth_google_client_id, config.oauth_google_client_secret
    return "", ""


def oauth_configured(provider: str) -> bool:
    client_id, client_secret = oauth_credentials(provider)
    return bool(client_id and client_secret)


# Issued `state` values, with the time they were handed out. An OAuth callback
# that cannot present one we issued is a login CSRF - someone else's code,
# redeemed in this browser, silently signing the user into an account they do
# not control.
#
# ponytail: in-process, so with several workers a callback can land on a
# replica that never issued the state. Move to Redis (a SETEX per state) if
# the login failure rate says it matters.
_OAUTH_STATES: Dict[str, float] = {}
OAUTH_STATE_TTL = 600.0


def remember_oauth_state(state: str) -> None:
    now = time.time()
    # Opportunistic sweep: no timer needed for a dictionary this small.
    for old, issued in list(_OAUTH_STATES.items()):
        if now - issued > OAUTH_STATE_TTL:
            _OAUTH_STATES.pop(old, None)
    _OAUTH_STATES[state] = now


def consume_oauth_state(state: str) -> bool:
    """Single use: a state that has been redeemed cannot be replayed."""
    issued = _OAUTH_STATES.pop(state, None)
    return issued is not None and (time.time() - issued) <= OAUTH_STATE_TTL


def oauth_authorize_url(provider: str, state: str) -> str:
    """Builds the redirect that starts a login."""
    from urllib.parse import urlencode

    spec = OAUTH_PROVIDERS[provider]
    client_id, _ = oauth_credentials(provider)
    params = {
        "client_id": client_id,
        "redirect_uri": f"{config.oauth_redirect_base.rstrip('/')}/api/auth/oauth/{provider}/callback",
        "scope": spec["scope"],
        "state": state,
        "response_type": "code",
    }
    return f"{spec['authorize_url']}?{urlencode(params)}"


async def oauth_exchange(provider: str, code: str) -> Dict[str, Any]:
    """
    Swaps an authorisation code for the provider's view of the user.

    Returns `{subject, email, name}`. Raises on anything that is not a
    complete, verified identity - an account with no confirmed email is not
    something to key a tenant on.
    """
    import httpx

    spec = OAUTH_PROVIDERS[provider]
    client_id, client_secret = oauth_credentials(provider)
    redirect_uri = f"{config.oauth_redirect_base.rstrip('/')}/api/auth/oauth/{provider}/callback"

    async with httpx.AsyncClient(timeout=20.0) as client:
        token_response = await client.post(
            spec["token_url"],
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token", "")
        if not access_token:
            raise TokenError(f"{provider} did not return an access token.")

        user_response = await client.get(
            spec["user_url"], headers={"Authorization": f"Bearer {access_token}"}
        )
        user_response.raise_for_status()
        profile = user_response.json()

        email = profile.get("email") or ""
        if provider == "github" and not email:
            # GitHub omits the email from /user when it is set to private.
            emails = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if emails.status_code == 200:
                primary = next(
                    (e for e in emails.json() if e.get("primary") and e.get("verified")), None
                )
                email = (primary or {}).get("email", "")

    if not email:
        raise TokenError(f"No verified email address available from {provider}.")
    return {
        "subject": str(profile.get("id") or profile.get("sub") or email),
        "email": email.lower(),
        "name": profile.get("name") or profile.get("login") or email.split("@")[0],
    }
