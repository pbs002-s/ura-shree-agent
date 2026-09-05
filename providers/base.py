"""
The provider interface, plus the neutral message and event types every backend
is normalised into.

Keeping one shape in the middle means the agent loop is written once and works
against a local checkpoint, Anthropic, OpenAI or anything OpenAI-compatible
without branching on which one is in play.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, AsyncGenerator, Dict, List, Optional


@dataclass
class ToolCall:
    """A request from the model to run one tool."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    thought_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "name": self.name, "arguments": self.arguments}
        if self.thought_signature:
            out["thought_signature"] = self.thought_signature
        return out


@dataclass
class Message:
    """
    One conversation turn.

    role is "system" | "user" | "assistant" | "tool".
    A tool result carries `tool_call_id` and puts the output in `content`.
    """

    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class ToolSpec:
    """A tool offered to the model, in JSON Schema terms."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StreamEvent:
    """
    One item out of a streaming completion.

    type is one of:
      "text"       - `text` holds a chunk of the visible reply
      "thinking"   - `text` holds a chunk of reasoning, where the model exposes it
      "tool_call"  - `tool_call` holds a fully assembled call
      "usage"      - `usage` holds token counts
      "error"      - `text` holds the failure message
      "done"       - end of the turn; `stop_reason` says why
    """

    type: str
    text: str = ""
    tool_call: Optional[ToolCall] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type}
        if self.text:
            out["text"] = self.text
        if self.tool_call:
            out["tool_call"] = self.tool_call.to_dict()
        if self.usage:
            out["usage"] = self.usage
        if self.stop_reason:
            out["stop_reason"] = self.stop_reason
        return out


@dataclass
class ModelInfo:
    """One selectable model, as returned by a provider scan."""

    id: str
    label: str = ""
    context_window: int = 0
    supports_tools: bool = True
    owned_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label or self.id,
            "context_window": self.context_window,
            "supports_tools": self.supports_tools,
            "owned_by": self.owned_by,
        }


class ProviderError(RuntimeError):
    """Raised when a provider rejects a request, with the status attached."""

    def __init__(self, message: str, status: int = 0, provider: str = ""):
        super().__init__(message)
        self.status = status
        self.provider = provider


class ChatProvider:
    """Base class. Subclasses implement `list_models` and `stream`."""

    protocol = "abstract"

    def __init__(self, spec, api_key: str = "", base_url: str = "", timeout: float = 300.0):
        self.spec = spec
        self.api_key = api_key or ""
        self.base_url = (base_url or spec.base_url).rstrip("/")
        self.timeout = timeout

    async def list_models(self) -> List[ModelInfo]:
        raise NotImplementedError

    async def stream(
        self,
        model: str,
        messages: List[Message],
        system: str = "",
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamEvent, None]:
        raise NotImplementedError
        yield  # pragma: no cover - makes the signature an async generator

    async def aclose(self) -> None:
        """Release any held connections."""
        return None
