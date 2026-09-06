"""
Rate limiting for REST routes and WebSocket frames.

The expensive things this server does are expensive for someone else: a
provider scan is an outbound API call against the user's own quota, a chat turn
is tokens they are paying for, and a WebSocket can push frames as fast as a
loop can write them. All three need a ceiling that is enforced per caller
rather than per process.

The bucket is a token bucket in Redis, refilled continuously rather than reset
on a boundary, so a caller cannot save up a whole window and spend it in one
burst at the turn of the minute. The Lua script makes read-decide-write one
atomic step, which a GET/SET pair across several workers is not.

Without Redis it falls back to an in-process bucket. That is honest but weaker:
with N workers the effective limit is N times the configured one. Fine for a
single-process local run, not a substitute for Redis in production.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, Request, status

from server.config import config

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    REDIS_AVAILABLE = False

# KEYS[1] bucket, ARGV: capacity, refill_per_second, now, cost.
# Returns {allowed, tokens_left, retry_after_seconds}.
_TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local refill    = tonumber(ARGV[2])
local now       = tonumber(ARGV[3])
local cost      = tonumber(ARGV[4])

local bucket = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts     = tonumber(bucket[2])
if tokens == nil then
  tokens = capacity
  ts = now
end

tokens = math.min(capacity, tokens + (now - ts) * refill)

local allowed = 0
local retry = 0
if tokens >= cost then
  allowed = 1
  tokens = tokens - cost
else
  retry = (cost - tokens) / refill
end

redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
-- Expire once a full refill has elapsed; an idle bucket is indistinguishable
-- from a fresh one, so keeping it costs memory for nothing.
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill) + 1)
return {allowed, tostring(tokens), tostring(retry)}
"""


@dataclass(frozen=True)
class Limit:
    """`count` requests per `window` seconds."""

    count: int
    window: float

    @property
    def refill_rate(self) -> float:
        return self.count / self.window if self.window else float(self.count)

    @classmethod
    def parse(cls, spec: str, default: str = "60/60") -> "Limit":
        """Reads the `"requests/seconds"` form used in configuration."""
        raw = (spec or default).strip()
        try:
            count, window = raw.split("/")
            return cls(count=int(count), window=float(window))
        except (ValueError, AttributeError):
            count, window = default.split("/")
            return cls(count=int(count), window=float(window))


class _LocalBuckets:
    """The in-process fallback. Per worker, which is its whole limitation."""

    def __init__(self) -> None:
        self._state: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def consume(self, key: str, limit: Limit, cost: float = 1.0) -> Tuple[bool, float, float]:
        now = time.monotonic()
        with self._lock:
            tokens, stamp = self._state.get(key, (float(limit.count), now))
            tokens = min(limit.count, tokens + (now - stamp) * limit.refill_rate)
            if tokens >= cost:
                self._state[key] = (tokens - cost, now)
                return True, tokens - cost, 0.0
            self._state[key] = (tokens, now)
            return False, tokens, (cost - tokens) / limit.refill_rate

    def clear(self) -> None:
        with self._lock:
            self._state.clear()


class RateLimiter:
    """One limiter for the process, backed by Redis when one is configured."""

    def __init__(self, redis_url: str = ""):
        self.redis_url = redis_url
        self._redis: Optional[Any] = None
        self._script = None
        self._local = _LocalBuckets()

    async def _client(self):
        if not self.redis_url or not REDIS_AVAILABLE:
            return None
        if self._redis is None:
            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)
        return self._redis

    async def consume(self, key: str, limit: Limit, cost: float = 1.0) -> Tuple[bool, float, float]:
        """Returns `(allowed, tokens_left, retry_after_seconds)`."""
        client = await self._client()
        if client is None:
            return self._local.consume(key, limit, cost)
        try:
            allowed, tokens, retry = await self._script(
                keys=[f"ratelimit:{key}"],
                args=[limit.count, limit.refill_rate, time.time(), cost],
            )
            return bool(int(allowed)), float(tokens), float(retry)
        except Exception:
            # A Redis outage must not take the API down with it. Degrading to
            # the local bucket keeps a ceiling in place, just a looser one.
            return self._local.consume(key, limit, cost)

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if (self.redis_url and REDIS_AVAILABLE) else "in-process"


limiter = RateLimiter(config.redis_url)


def client_identity(request: Request, principal_id: str = "") -> str:
    """
    The bucket a request counts against.

    An authenticated caller is keyed by user id, so rotating IP addresses does
    not multiply their quota. Anonymous traffic falls back to the peer address,
    and `X-Forwarded-For` is only believed when a proxy is declared - otherwise
    any client could mint a fresh bucket per request by sending a new header.
    """
    if principal_id:
        return f"user:{principal_id}"
    if config.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


async def enforce(bucket: str, identity: str, spec: str) -> None:
    """Raises 429 with a Retry-After header when the caller is over budget."""
    limit = Limit.parse(spec)
    allowed, _remaining, retry_after = await limiter.consume(f"{bucket}:{identity}", limit)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {bucket}. Try again in {retry_after:.0f}s.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.5)))},
        )


def rate_limit(bucket: str, spec_attr: str):
    """
    Builds a FastAPI dependency that enforces one named limit.

    The spec is read from config at call time rather than at import, so a
    deployment can change a limit without the decorator having captured the old
    value.
    """

    async def dependency(request: Request) -> None:
        from server.auth import current_principal

        try:
            principal = await current_principal(request)
            identity = client_identity(request, principal.id)
        except HTTPException:
            identity = client_identity(request)
        await enforce(bucket, identity, getattr(config, spec_attr))

    return dependency


class SocketLimiter:
    """
    Per-connection frame budget.

    A WebSocket has no middleware in front of it, and an unbounded client loop
    is the cheapest way to make a server do expensive work. This is checked in
    the receive loop, before the frame is acted on.
    """

    def __init__(self, identity: str, spec: str):
        self.identity = identity
        self.limit = Limit.parse(spec)

    async def allow(self, cost: float = 1.0) -> Tuple[bool, float]:
        allowed, _remaining, retry_after = await limiter.consume(
            f"ws:{self.identity}", self.limit, cost
        )
        return allowed, retry_after
