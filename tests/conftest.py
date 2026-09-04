"""
Shared fixtures.

The important one is `ScriptedProvider`: a `ChatProvider` whose replies are
written by the test rather than sampled from a model. Testing an agent against
a live model tests the model; testing it against a scripted one tests the loop,
which is the part this repository owns.
"""

import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from providers.base import ChatProvider, Message, ModelInfo, StreamEvent, ToolCall, ToolSpec
from providers.catalog import get_spec


class ScriptedProvider(ChatProvider):
    """
    Replays a fixed list of turns.

    Each turn is `{"text": str, "tools": [(name, args), ...]}`. The loop should
    call the provider once per turn and stop when a turn asks for no tools.
    """

    protocol = "scripted"

    def __init__(self, turns: List[Dict[str, Any]]):
        super().__init__(get_spec("local"))
        self.turns = turns
        self.calls: List[Dict[str, Any]] = []
        self.seen_messages: List[List[Message]] = []

    async def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(id="scripted-1")]

    async def stream(
        self,
        model: str,
        messages: List[Message],
        system: str = "",
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.seen_messages.append(list(messages))
        index = len(self.calls)
        self.calls.append({"model": model, "system": system, "tools": tools})

        turn = self.turns[index] if index < len(self.turns) else {"text": "Done.", "tools": []}

        for chunk in turn.get("text", ""):
            yield StreamEvent(type="text", text=chunk)

        for n, (name, args) in enumerate(turn.get("tools", [])):
            yield StreamEvent(
                type="tool_call",
                tool_call=ToolCall(id=f"call-{index}-{n}", name=name, arguments=args),
            )

        yield StreamEvent(type="done", stop_reason="end_turn")


@pytest.fixture
def scripted() -> Callable[[List[Dict[str, Any]]], ScriptedProvider]:
    return lambda turns: ScriptedProvider(turns)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A small throwaway project to run agents and snapshots against."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "calc.py").write_text(
        "def divide(a, b):\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Sample\n", encoding="utf-8")
    return tmp_path
