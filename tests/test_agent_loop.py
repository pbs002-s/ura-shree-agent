"""
Agent loop tests.

The loop is driven by a scripted provider, so what is under test is the harness
around the model: does it run the tool the model asked for, does it feed the
result back, does it stop when the model stops asking, and does it survive a
failing tool without derailing the turn.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import CodingAgent
from agent.loop import AgentSession
from agent.toolkit import Toolkit
from tools.filesystem import FileSystemTool
from tools.shell import ShellManager


def build_agent(workspace: Path) -> CodingAgent:
    return CodingAgent(
        workspace_root=str(workspace),
        memory_db=str(workspace / ".shree" / "memory.db"),
        session_id="test",
    )


async def collect(agent, request, provider):
    events = []
    async for event in agent.stream_task(request, provider, model="scripted-1"):
        events.append(event)
    return events


def test_tool_call_runs_and_result_returns_to_the_model(workspace, scripted):
    provider = scripted([
        {"text": "Reading the file.", "tools": [("read_file", {"path": "src/calc.py"})]},
        {"text": "It divides a by b with no zero guard."},
    ])
    agent = build_agent(workspace)
    try:
        events = asyncio.run(collect(agent, "What does calc.py do?", provider))
    finally:
        agent.close()

    kinds = [e["type"] for e in events]
    assert "tool_start" in kinds and "tool_end" in kinds

    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert tool_end["ok"] and "def divide" in tool_end["text"]

    # The second request must carry the tool result, or the model is answering blind.
    second_turn = provider.seen_messages[1]
    assert any(m.role == "tool" and "def divide" in m.content for m in second_turn)

    done = next(e for e in events if e["type"] == "done")
    assert done["stop_reason"] == "end_turn" and done["turns"] == 2


def test_edit_is_applied_and_snapshotted(workspace, scripted):
    provider = scripted([
        {"text": "", "tools": [("edit_file", {
            "path": "src/calc.py",
            "old_string": "    return a / b",
            "new_string": "    if b == 0:\n        return 0\n    return a / b",
        })]},
        {"text": "Added a zero guard."},
    ])
    agent = build_agent(workspace)
    try:
        events = asyncio.run(collect(agent, "Guard against division by zero", provider))
        timeline = agent.time_machine.timeline()
    finally:
        agent.close()

    assert (workspace / "src" / "calc.py").read_text(encoding="utf-8").count("if b == 0") == 1

    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert tool_end["ok"], tool_end["text"]
    # Mutating tools snapshot first, so the edit is reversible without being asked.
    assert tool_end["data"].get("snapshot_before"), "no snapshot was taken before the edit"
    assert timeline["count"] >= 1


def test_ambiguous_edit_is_refused_rather_than_guessed(workspace, scripted):
    (workspace / "src" / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    provider = scripted([
        {"text": "", "tools": [("edit_file", {
            "path": "src/dup.py", "old_string": "x = 1", "new_string": "x = 2",
        })]},
        {"text": "That string was not unique."},
    ])
    agent = build_agent(workspace)
    try:
        events = asyncio.run(collect(agent, "Change x", provider))
    finally:
        agent.close()

    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert not tool_end["ok"]
    assert "2 times" in tool_end["text"]
    # The file must be untouched: a rejected edit that half-applied is worse
    # than no edit at all.
    assert (workspace / "src" / "dup.py").read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_failing_tool_does_not_end_the_turn(workspace, scripted):
    provider = scripted([
        {"text": "", "tools": [("read_file", {"path": "does/not/exist.py"})]},
        {"text": "That file does not exist."},
    ])
    agent = build_agent(workspace)
    try:
        events = asyncio.run(collect(agent, "Read a missing file", provider))
    finally:
        agent.close()

    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert not tool_end["ok"]
    assert next(e for e in events if e["type"] == "done")["turns"] == 2


def test_auto_approve_off_blocks_mutating_tools(workspace, scripted):
    provider = scripted([
        {"text": "", "tools": [("write_file", {"path": "new.py", "content": "print(1)"})]},
        {"text": "I was not allowed to write."},
    ])
    agent = build_agent(workspace)
    try:
        events = []

        async def run():
            async for event in agent.stream_task(
                "Write a file", provider, model="scripted-1", auto_approve=False
            ):
                events.append(event)

        asyncio.run(run())
    finally:
        agent.close()

    tool_end = next(e for e in events if e["type"] == "tool_end")
    assert not tool_end["ok"] and "declined permission" in tool_end["text"].lower()
    assert not (workspace / "new.py").exists()


def test_turn_budget_is_enforced(workspace, scripted):
    # A provider that always asks for another tool would otherwise loop forever.
    provider = scripted([{"text": "", "tools": [("list_dir", {"path": "."})]}] * 20)
    agent = CodingAgent(
        workspace_root=str(workspace),
        memory_db=str(workspace / ".shree" / "memory.db"),
        max_turns=3,
        session_id="test-budget",
    )
    try:
        events = asyncio.run(collect(agent, "Loop forever", provider))
    finally:
        agent.close()

    done = next(e for e in events if e["type"] == "done")
    assert done["stop_reason"] == "max_turns" and done["turns"] == 3


def test_conversation_history_persists_across_requests(workspace, scripted):
    provider = scripted([{"text": "First."}, {"text": "Second."}])
    agent = build_agent(workspace)
    try:
        asyncio.run(collect(agent, "one", provider))
        asyncio.run(collect(agent, "two", provider))
    finally:
        agent.close()

    # The second call must see the whole prior exchange, not just the new question.
    second = provider.seen_messages[1]
    assert [m.role for m in second] == ["user", "assistant", "user"]
    assert second[0].content == "one" and second[2].content == "two"


def test_system_prompt_forbids_padded_answers(workspace):
    toolkit = Toolkit(
        workspace_root=str(workspace),
        filesystem=FileSystemTool(str(workspace)),
        shell_manager=ShellManager(str(workspace)),
    )
    session = AgentSession(workspace_root=str(workspace), toolkit=toolkit)
    prompt = session.system_prompt.lower()

    assert "no preamble" in prompt
    assert "default to a few sentences" in prompt
    assert "do not use emoji" in prompt
    assert "read a file before editing it" in prompt


def test_attachments_are_inlined_into_the_user_turn(workspace):
    toolkit = Toolkit(
        workspace_root=str(workspace),
        filesystem=FileSystemTool(str(workspace)),
        shell_manager=ShellManager(str(workspace)),
    )
    session = AgentSession(workspace_root=str(workspace), toolkit=toolkit)
    session.add_user_message("Review this", attachments=[{"path": "a.py", "content": "print(1)"}])

    assert "Attached file `a.py`" in session.messages[0].content
    assert "print(1)" in session.messages[0].content
