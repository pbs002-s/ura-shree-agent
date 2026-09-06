"""
Structured logs, correlation IDs and traces.

A single-user tool can get away with `print`. A hosted one cannot: when a
request goes wrong you need to pull every line it produced out of a stream
carrying a hundred other sessions, and free-text lines do not join. So every
log line is a JSON object, and every line emitted while handling a request
carries the same `request_id`, `session_id` and `user_id` without the call site
having to pass them down.

The context lives in `contextvars`, which follow an `await` and are copied into
tasks - so a tool call running inside an agent turn logs under the request that
started it, several layers below the middleware that set it.

structlog and OpenTelemetry are both optional. Without them this degrades to
the standard library and no-op spans rather than failing to import, because a
developer running the tool locally should not have to install a tracing stack.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

from server.config import config

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    STRUCTLOG_AVAILABLE = False

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
project_id_var: ContextVar[str] = ContextVar("project_id", default="")

_configured = False


def current_context() -> Dict[str, str]:
    """The correlation fields in scope right now."""
    return {
        "request_id": request_id_var.get(),
        "session_id": session_id_var.get(),
        "user_id": user_id_var.get(),
        "project_id": project_id_var.get(),
    }


@contextmanager
def bind_context(**fields: str) -> Iterator[None]:
    """
    Sets correlation fields for the duration of a block.

    Used by the WebSocket handlers, which have no HTTP middleware to do it for
    them but are exactly where the long-lived work happens.
    """
    tokens = []
    mapping = {
        "request_id": request_id_var,
        "session_id": session_id_var,
        "user_id": user_id_var,
        "project_id": project_id_var,
    }
    for key, value in fields.items():
        var = mapping.get(key)
        if var is not None and value:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _inject_context(_logger, _method, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in current_context().items():
        if value and key not in event_dict:
            event_dict[key] = value
    return event_dict


class _JsonFormatter(logging.Formatter):
    """Fallback formatter for when structlog is not installed."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.time(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
            **{k: v for k, v in current_context().items() if v},
            **getattr(record, "extra_fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _StdlibLogger:
    """A structlog-shaped facade over `logging`, so call sites are identical."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        self._log.log(level, event, extra={"extra_fields": fields})

    def debug(self, event: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._log.exception(event, extra={"extra_fields": fields})


def configure_logging() -> None:
    """Installs the JSON pipeline. Idempotent; safe to call from every worker."""
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, config.log_level, logging.INFO)
    # JSON in the cloud, human-readable on a terminal, unless overridden.
    as_json = config.log_json or config.cloud

    handler = logging.StreamHandler(sys.stdout)
    if not STRUCTLOG_AVAILABLE and as_json:
        handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)

    if not STRUCTLOG_AVAILABLE:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if as_json
            else structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "shree"):
    """A logger that takes an event name plus keyword fields."""
    configure_logging()
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return _StdlibLogger(name)


# -- tracing ------------------------------------------------------------------

_tracer = None


def setup_tracing(app=None) -> bool:
    """
    Wires OpenTelemetry when an OTLP endpoint is configured.

    Returns whether tracing is live, so `/api/status` can report it honestly
    rather than implying traces exist when nothing is collecting them.
    """
    global _tracer
    if not config.otel_endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover - depends on the install
        get_logger("otel").warning(
            "otel_endpoint_set_but_sdk_missing", endpoint=config.otel_endpoint
        )
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": config.service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{config.otel_endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(config.service_name)

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:
            pass
    return True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """
    A traced block that costs nothing when tracing is off.

    Also emits a duration log line either way, so LLM and tool latency is
    measurable from logs alone on a deployment with no collector.
    """
    started = time.perf_counter()
    if _tracer is None:
        try:
            yield
        finally:
            get_logger("span").debug(
                name, duration_ms=round((time.perf_counter() - started) * 1000, 2), **attributes
            )
        return

    with _tracer.start_as_current_span(name) as otel_span:
        for key, value in attributes.items():
            otel_span.set_attribute(key, value if isinstance(value, (str, int, float, bool)) else str(value))
        try:
            yield
        finally:
            otel_span.set_attribute(
                "duration_ms", round((time.perf_counter() - started) * 1000, 2)
            )


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class CorrelationMiddleware:
    """
    Pure-ASGI correlation and access logging.

    Written against the raw ASGI interface rather than `BaseHTTPMiddleware`
    because that one wraps the request in a task group, which breaks streaming
    responses and adds a hop to every request for no benefit here.
    """

    def __init__(self, app):
        self.app = app
        self.log = get_logger("http")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        # An upstream proxy may already have assigned one; reusing it keeps a
        # single trace across the whole hop chain.
        request_id = headers.get("x-request-id") or new_request_id()
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_holder: Dict[str, int] = {"status": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            request_id_var.reset(token)
            path = scope.get("path", "")
            # Static asset noise buries the lines that matter.
            if not path.startswith("/assets/"):
                self.log.info(
                    "request",
                    method=scope.get("method", ""),
                    path=path,
                    status=status_holder["status"],
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    request_id=request_id,
                )
