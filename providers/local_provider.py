"""
The locally trained URA-Shree checkpoint, exposed through the same provider
interface as the network backends.

Two things are worth being straight about. First, generation is synchronous
PyTorch, so it runs on a worker thread and reaches the event loop through a
queue; otherwise a single reply would block every other websocket on the
server. Second, a 14M-parameter model with a 2k vocabulary has not learned
structured function calling. The tool interface is still wired up - the model
emits a `<|tool_call|>` marker followed by JSON and it is parsed here - but
expect it to fire rarely. For real agent work, point the UI at an API key.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional

from providers.base import (
    ChatProvider,
    Message,
    ModelInfo,
    ProviderError,
    StreamEvent,
    ToolCall,
    ToolSpec,
)

TOOL_CALL_MARKER = "<|tool_call|>"
_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_MARKER) + r"\s*(\{.*?\})\s*(?:<\|/tool_call\|>|$)", re.DOTALL
)

_SENTINEL = object()


class LocalProvider(ChatProvider):
    """Streams from an `inference.engine.InferenceEngine` instance."""

    protocol = "local"

    def __init__(self, spec, engine=None, engine_factory=None, **kwargs):
        super().__init__(spec, **kwargs)
        self._engine = engine
        self._engine_factory = engine_factory

    @property
    def engine(self):
        if self._engine is None:
            if self._engine_factory is None:
                raise ProviderError("No local checkpoint is loaded.", provider="local")
            self._engine = self._engine_factory()
            if self._engine is None:
                raise ProviderError(
                    "No local checkpoint is loaded. Train one, or select an API provider.",
                    provider="local",
                )
        return self._engine

    async def list_models(self) -> List[ModelInfo]:
        """Lists the checkpoints on disk; each is a selectable 'model'."""
        from pathlib import Path

        checkpoints = sorted(Path("checkpoints").glob("*.pt")) if Path("checkpoints").exists() else []
        labels = {
            "coding_best.pt": "Shree coding (fine-tuned)",
            "best.pt": "Shree base (best validation)",
            "last.pt": "Shree base (latest step)",
        }
        return [
            ModelInfo(
                id=str(path.as_posix()),
                label=labels.get(path.name, path.stem),
                context_window=getattr(self._engine, "model_config", None).max_seq_len
                if self._engine
                else 0,
                supports_tools=False,
                owned_by="local",
            )
            for path in checkpoints
        ]

    @staticmethod
    def _flatten(messages: List[Message], system: str, tools: Optional[List[ToolSpec]]) -> str:
        """Renders the conversation into the chat-marker format the model saw in training."""
        parts: List[str] = []
        if system:
            parts.append(f"<|system|>\n{system}\n")
        if tools:
            listing = "\n".join(f"- {t.name}: {t.description}" for t in tools)
            parts.append(
                "<|system|>\nAvailable tools:\n" + listing + "\n"
                f"To use one, emit {TOOL_CALL_MARKER} followed by "
                '{"name": "<tool>", "arguments": {...}}\n'
            )
        for msg in messages:
            if msg.role == "tool":
                parts.append(f"<|tool_result|>\n{msg.content}\n")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}\n")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    async def stream(
        self,
        model: str,
        messages: List[Message],
        system: str = "",
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncGenerator[StreamEvent, None]:
        engine = self.engine
        prompt = self._flatten(messages, system, tools)

        # Bounded queue so a fast GPU cannot outrun a slow websocket and grow
        # an unbounded backlog in memory.
        bridge: "queue.Queue[Any]" = queue.Queue(maxsize=256)
        failure: List[BaseException] = []

        def produce() -> None:
            try:
                for piece in engine.generate_stream(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                ):
                    bridge.put(piece)
            except BaseException as err:  # surfaced on the consumer side
                failure.append(err)
            finally:
                bridge.put(_SENTINEL)

        worker = threading.Thread(target=produce, name="shree-local-generate", daemon=True)
        worker.start()

        buffered = ""
        emitted_upto = 0
        loop = asyncio.get_running_loop()

        while True:
            item = await loop.run_in_executor(None, bridge.get)
            if item is _SENTINEL:
                break
            buffered += item

            # Hold back text once a tool-call marker starts, so the raw JSON
            # never reaches the transcript.
            marker_at = buffered.find(TOOL_CALL_MARKER, emitted_upto)
            visible_end = marker_at if marker_at >= 0 else len(buffered)
            if visible_end > emitted_upto:
                yield StreamEvent(type="text", text=buffered[emitted_upto:visible_end])
                emitted_upto = visible_end

        if failure:
            raise ProviderError(f"Local generation failed: {failure[0]}", provider="local")

        for match in _TOOL_CALL_RE.finditer(buffered):
            try:
                spec = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            name = spec.get("name") or spec.get("tool")
            if not name:
                continue
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(
                    id=f"local_{match.start()}",
                    name=name,
                    arguments=spec.get("arguments") or spec.get("args") or {},
                ),
            )

        stats = getattr(engine, "last_stats", None)
        if stats is not None:
            yield StreamEvent(type="usage", usage=stats.to_dict())
        yield StreamEvent(type="done", stop_reason="end_turn")
