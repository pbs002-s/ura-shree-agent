"""
The agent loop.

One turn is: send the conversation to the model, stream back text and tool
calls, run the tools, append the results, repeat until the model stops asking
for tools or the turn budget runs out. That is the whole idea - the intelligence
is in the model, and this file's job is to be a correct, observable harness
around it rather than to fake reasoning with hand-written branches.

Everything is emitted as events so the UI can render the turn as it happens
instead of waiting for a final blob.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from providers.base import ChatProvider, Message, ProviderError, ToolCall
from agent.prompts import build_agent_prompt
from agent.toolkit import MUTATING_TOOLS, Toolkit

MAX_TOOL_RESULT_CHARS = 12_000


class AgentSession:
    """
    A conversation with tools attached, bound to one workspace.

    The session owns the message history, so a follow-up question keeps the
    context of what was already read and run.
    """

    def __init__(
        self,
        workspace_root: str,
        toolkit: Toolkit,
        max_turns: int = 24,
        tree_summary: str = "",
        memory_context: str = "",
        platform_name: str = "",
    ):
        self.workspace_root = str(Path(workspace_root).resolve())
        self.toolkit = toolkit
        self.max_turns = max_turns
        self.messages: List[Message] = []
        self.system_prompt = build_agent_prompt(
            workspace_root=self.workspace_root,
            tree_summary=tree_summary,
            memory_context=memory_context,
            platform_name=platform_name,
        )
        self.total_usage: Dict[str, int] = {}

    def reset(self) -> None:
        self.messages = []
        self.total_usage = {}

    def add_user_message(self, text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Appends a user turn.

        Attachments are inlined as fenced blocks rather than uploaded anywhere;
        the model only ever sees text, and the file already lives in the
        workspace where the tools can reach it.
        """
        body = text
        for item in attachments or []:
            name = item.get("path") or item.get("name") or "attachment"
            content = item.get("content", "")
            if content:
                body += f"\n\nAttached file `{name}`:\n```\n{content}\n```"
            else:
                body += f"\n\nAttached: `{name}`"
        self.messages.append(Message(role="user", content=body))

    def transcript(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    def _merge_usage(self, usage: Dict[str, Any]) -> None:
        for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                self.total_usage[key] = self.total_usage.get(key, 0) + value

    async def run(
        self,
        provider: ChatProvider,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_tools: bool = True,
        auto_approve: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Drives the conversation to completion, yielding events.

        Event types: `turn_start`, `thinking`, `text`, `tool_start`, `tool_end`,
        `usage`, `error`, `done`.
        """
        tools = self.toolkit.specs() if use_tools else None
        started = time.perf_counter()

        for turn in range(1, self.max_turns + 1):
            yield {"type": "turn_start", "turn": turn}

            assistant_text: List[str] = []
            calls: List[ToolCall] = []
            stop_reason = "end_turn"

            try:
                async for event in provider.stream(
                    model=model,
                    messages=self.messages,
                    system=self.system_prompt,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if event.type == "text":
                        assistant_text.append(event.text)
                        yield {"type": "text", "text": event.text}
                    elif event.type == "thinking":
                        yield {"type": "thinking", "text": event.text}
                    elif event.type == "tool_call" and event.tool_call:
                        calls.append(event.tool_call)
                    elif event.type == "usage":
                        self._merge_usage(event.usage)
                        yield {"type": "usage", "usage": event.usage}
                    elif event.type == "done":
                        stop_reason = event.stop_reason or stop_reason
            except ProviderError as err:
                yield {"type": "error", "message": str(err), "status": getattr(err, "status", 0)}
                return
            except asyncio.CancelledError:
                yield {"type": "error", "message": "Cancelled."}
                raise
            except Exception as err:
                yield {"type": "error", "message": f"Provider stream failed: {err}"}
                return

            reply = "".join(assistant_text)
            self.messages.append(Message(role="assistant", content=reply, tool_calls=calls))

            if not calls:
                yield {
                    "type": "done",
                    "turns": turn,
                    "stop_reason": stop_reason,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "usage": self.total_usage,
                }
                return

            for call in calls:
                needs_approval = (not auto_approve) and call.name in MUTATING_TOOLS
                yield {
                    "type": "tool_start",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "mutating": call.name in MUTATING_TOOLS,
                }

                if needs_approval:
                    result = {
                        "ok": False,
                        "text": "Refused: this tool changes the workspace and approval is required.",
                        "data": {},
                    }
                else:
                    # Tools are blocking (subprocesses, disk IO). Running them on
                    # the loop thread would stall every other websocket.
                    result = await asyncio.to_thread(
                        self.toolkit.execute, call.name, call.arguments
                    )

                text = result.get("text", "")
                if len(text) > MAX_TOOL_RESULT_CHARS:
                    text = text[:MAX_TOOL_RESULT_CHARS] + "\n... result truncated"

                self.messages.append(
                    Message(role="tool", content=text, tool_call_id=call.id, name=call.name)
                )
                yield {
                    "type": "tool_end",
                    "id": call.id,
                    "name": call.name,
                    "ok": result.get("ok", False),
                    "text": text,
                    "data": result.get("data", {}),
                }

        yield {
            "type": "done",
            "turns": self.max_turns,
            "stop_reason": "max_turns",
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "usage": self.total_usage,
        }
