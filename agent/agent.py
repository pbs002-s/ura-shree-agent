"""
The Shree coding agent.

This is the assembly point: it wires a workspace to its filesystem sandbox,
persistent shell, AST index, SQLite memory and Time Machine, then hands that
toolkit to `agent.loop.AgentSession` and lets the selected model drive.

There is deliberately no hand-written intent matching here. An agent whose
answers come from a table of keyword branches is a demo, not an agent; the
model decides what to do and this class makes sure it can actually do it.
"""

from __future__ import annotations

import asyncio
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.indexer import CodebaseIndexer
from agent.loop import AgentSession
from agent.memory import ProjectMemory
from agent.timemachine import TimeMachine
from agent.toolkit import Toolkit
from providers import ChatProvider, build_provider
from tools.filesystem import FileSystemTool
from tools.git import GitTool
from tools.shell import ShellManager


class CodingAgent:
    """A workspace plus everything needed to act on it."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        memory_db: Optional[str] = None,
        shell_manager: Optional[ShellManager] = None,
        time_machine: Optional[TimeMachine] = None,
        max_turns: int = 24,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        session_id: str = "agent",
    ):
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.session_id = session_id
        self.event_callback = event_callback
        self.max_turns = max_turns

        self.fs = FileSystemTool(str(self.workspace_root))
        self.git = GitTool(str(self.workspace_root))
        self.shells = shell_manager or ShellManager(str(self.workspace_root))
        self.indexer = CodebaseIndexer(str(self.workspace_root))

        memory_path = memory_db or str(self.workspace_root / "checkpoints" / "shree_memory.db")
        Path(memory_path).parent.mkdir(parents=True, exist_ok=True)
        self.memory = ProjectMemory(memory_path)

        self.time_machine = time_machine or TimeMachine(str(self.workspace_root))

        self.toolkit = Toolkit(
            workspace_root=str(self.workspace_root),
            filesystem=self.fs,
            shell_manager=self.shells,
            indexer=self.indexer,
            memory=self.memory,
            git=self.git,
            time_machine=self.time_machine,
            shell_session_id=session_id,
        )

        self._session: Optional[AgentSession] = None

    # -- session ------------------------------------------------------------

    def session(self, refresh_context: bool = False) -> AgentSession:
        """The running conversation, created on first use."""
        if self._session is None or refresh_context:
            try:
                self.indexer.scan_and_index()
                tree = self.indexer.get_tree_summary(max_depth=2)
            except Exception:
                tree = ""
            try:
                memory_context = self.memory.get_summary_context()
            except Exception:
                memory_context = ""

            self._session = AgentSession(
                workspace_root=str(self.workspace_root),
                toolkit=self.toolkit,
                max_turns=self.max_turns,
                tree_summary=tree,
                memory_context=memory_context,
                platform_name=f"{platform.system()} {platform.release()}",
            )
        return self._session

    def reset(self) -> None:
        if self._session is not None:
            self._session.reset()

    def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception:
                pass

    # -- running ------------------------------------------------------------

    async def stream_task(
        self,
        request: str,
        provider: ChatProvider,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_tools: bool = True,
        auto_approve: bool = True,
        attachments: Optional[List[Dict[str, Any]]] = None,
        fresh: bool = False,
    ):
        """Streams the agent's events for one user request."""
        session = self.session(refresh_context=fresh)
        if fresh:
            session.reset()
        session.add_user_message(request, attachments=attachments)

        async for event in session.run(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            use_tools=use_tools,
            auto_approve=auto_approve,
        ):
            self.emit(event.get("type", "event"), event)
            yield event

    def run_task(
        self,
        request: str,
        provider_id: str = "local",
        model: str = "",
        api_key: str = "",
        base_url: str = "",
        engine_factory: Optional[Callable[[], Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Blocking convenience wrapper around `stream_task`, for scripts and tests.

        Collects the reply text, the tool calls that ran, and how the turn ended.
        """
        provider = build_provider(
            provider_id, api_key=api_key, base_url=base_url, engine_factory=engine_factory
        )
        return asyncio.run(self._collect(request, provider, model, **kwargs))

    async def _collect(self, request: str, provider, model: str, **kwargs) -> Dict[str, Any]:
        reply: List[str] = []
        actions: List[Dict[str, Any]] = []
        error = ""
        outcome: Dict[str, Any] = {}

        async for event in self.stream_task(request, provider, model, **kwargs):
            kind = event.get("type")
            if kind == "text":
                reply.append(event["text"])
            elif kind == "tool_end":
                actions.append({
                    "tool": event["name"],
                    "ok": event["ok"],
                    "summary": event["text"][:400],
                })
            elif kind == "error":
                error = event["message"]
            elif kind == "done":
                outcome = event

        summary = "".join(reply).strip()
        if summary:
            try:
                self.memory.record_decision(
                    topic=request[:60],
                    decision=summary[:400],
                    rationale=f"{len(actions)} tool call(s) via the agent loop.",
                )
            except Exception:
                pass

        return {
            "success": not error,
            "task": request,
            "summary": summary,
            "error": error,
            "actions": actions,
            "turns": outcome.get("turns", 0),
            "stop_reason": outcome.get("stop_reason", "error" if error else "end_turn"),
            "duration_ms": outcome.get("duration_ms", 0),
            "usage": outcome.get("usage", {}),
        }

    def close(self) -> None:
        self.shells.close(self.session_id)
