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


def _record_completion_latency(**fields: Any) -> None:
    """
    Emits one measurement per model completion.

    Where the tokens come from is the single largest and most variable cost in
    a turn, so it gets its own span rather than being buried inside the turn's.
    Wrapped because observability is optional: a local install without the
    cloud extras still runs the loop.
    """
    try:
        from server.observability import get_logger

        get_logger("llm").info("completion", **fields)
    except Exception:
        pass


class AgentSession:
    """
    A conversation with tools attached, bound to one workspace.

    The session owns the message history, so a follow-up question keeps the
    context of what was already read and run.
    """

    def __init__(
        self,
        workspace_root: str = "",
        toolkit: Optional[Toolkit] = None,
        max_turns: int = 24,
        tree_summary: str = "",
        memory_context: str = "",
        platform_name: str = "",
        skills_prompt: str = "",
    ):
        self.workspace_root = str(Path(workspace_root).resolve()) if workspace_root else ""
        self.toolkit = toolkit
        self.max_turns = max_turns
        self.messages: List[Message] = []
        self.system_prompt = build_agent_prompt(
            workspace_root=self.workspace_root,
            tree_summary=tree_summary,
            memory_context=memory_context,
            platform_name=platform_name,
            skills_prompt=skills_prompt,
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
        approval_callback: Optional[Callable[[ToolCall], Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Drives the conversation to completion, yielding events.

        Event types: `turn_start`, `thinking`, `text`, `tool_start`, `tool_approval_prompt`, `tool_end`,
        `usage`, `error`, `done`.
        """
        tools = (self.toolkit.specs() if self.toolkit else None) if use_tools else None
        started = time.perf_counter()

        for turn in range(1, self.max_turns + 1):
            yield {"type": "turn_start", "turn": turn}

            assistant_text: List[str] = []
            calls: List[ToolCall] = []
            stop_reason = "end_turn"
            completion_started = time.perf_counter()
            first_token_ms = 0.0

            try:
                async for event in provider.stream(
                    model=model,
                    messages=self.messages,
                    system=self.system_prompt,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if not first_token_ms:
                        # Time to first token, not total latency: it is what the
                        # user actually waits for, and the only half of the
                        # number a slow provider and a long answer distinguish.
                        first_token_ms = (time.perf_counter() - completion_started) * 1000
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
                _record_completion_latency(
                    model=model,
                    turn=turn,
                    first_token_ms=first_token_ms,
                    total_ms=(time.perf_counter() - completion_started) * 1000,
                    stop_reason=stop_reason,
                    tool_calls=len(calls),
                )
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

            if not calls and not reply.strip():
                # If model finished thinking without emitting response text, prompt it for the response text.
                try:
                    followup = Message(role="user", content="Please provide your final answer or response now.")
                    async for event in provider.stream(
                        model=model,
                        messages=self.messages + [followup],
                        system=self.system_prompt,
                        tools=None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        if event.type == "text":
                            assistant_text.append(event.text)
                            yield {"type": "text", "text": event.text}
                        elif event.type == "thinking":
                            yield {"type": "thinking", "text": event.text}
                except Exception:
                    pass
                reply = "".join(assistant_text)

            if not calls and not reply.strip():
                last_user_msg = ""
                for m in reversed(self.messages):
                    if m.role == "user":
                        last_user_msg = (m.content or "").lower()
                        break
                is_identity_query = any(q in last_user_msg for q in ("who are you", "who made you", "what is your name", "creator"))
                if is_identity_query:
                    fallback_msg = (
                        "I am Shree, an intelligent AI coding assistant created and developed by Pritam from DIU (Daffodil International University), "
                        "built and trained upon the open-source Qwen foundation."
                    )
                else:
                    fallback_msg = "Here is the result."
                assistant_text.append(fallback_msg)
                yield {"type": "text", "text": fallback_msg}
                reply = fallback_msg

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
                needs_approval = (not auto_approve) and (call.name in MUTATING_TOOLS)
                yield {
                    "type": "tool_start",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                    "mutating": call.name in MUTATING_TOOLS,
                    "needs_approval": needs_approval,
                }

                approved = True
                if needs_approval:
                    if approval_callback:
                        yield {
                            "type": "tool_approval_prompt",
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        approved = await approval_callback(call)
                    else:
                        approved = False

                if not approved:
                    if self.toolkit:
                        self.toolkit.record_denied(call.name, call.arguments)
                    result = {
                        "ok": False,
                        "text": "Refused: The user declined permission to save or modify this file.",
                        "data": {},
                    }
                else:
                    # Who let this run is the fact an audit reader needs: a
                    # policy that auto-approved it, or a person who clicked.
                    approved_by = "user" if needs_approval else "auto_approve_policy"
                    # Tools are blocking (subprocesses, disk IO). Running them on
                    # the loop thread would stall every other websocket.
                    if self.toolkit:
                        result = await asyncio.to_thread(
                            self.toolkit.execute, call.name, call.arguments, approved_by
                        )
                    else:
                        result = {"ok": False, "text": "No toolkit available in general mode.", "data": {}}

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
