"""
Self-healing: the agent runs a failing test, sees the traceback, patches the
source and re-runs to verify.

The model's decisions are scripted so the test is deterministic, but nothing
else is faked - pytest really runs in the persistent shell, the file really
gets edited, and the second run really has to pass for the test to.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import CodingAgent

PYTHON = sys.executable


@pytest.fixture
def broken_project(tmp_path: Path) -> Path:
    (tmp_path / "calc.py").write_text(
        "def safe_divide(a, b):\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    (tmp_path / "test_calc.py").write_text(
        "from calc import safe_divide\n\n\n"
        "def test_divides():\n"
        "    assert safe_divide(10, 2) == 5\n\n\n"
        "def test_handles_zero():\n"
        "    assert safe_divide(10, 0) == 0\n",
        encoding="utf-8",
    )
    return tmp_path


def test_agent_diagnoses_and_repairs_a_failing_test(broken_project, scripted):
    run_tests = f'& "{PYTHON}" -m pytest test_calc.py -q'

    provider = scripted([
        {"text": "Running the tests.", "tools": [("run_command", {"command": run_tests})]},
        {"text": "ZeroDivisionError. Adding a guard.",
         "tools": [("edit_file", {
             "path": "calc.py",
             "old_string": "    return a / b",
             "new_string": "    if b == 0:\n        return 0\n    return a / b",
         })]},
        {"text": "Re-running.", "tools": [("run_command", {"command": run_tests})]},
        {"text": "Both tests pass now."},
    ])

    agent = CodingAgent(
        workspace_root=str(broken_project),
        memory_db=str(broken_project / ".shree" / "memory.db"),
        session_id="heal",
    )
    events = []
    try:
        async def run():
            async for event in agent.stream_task(
                "The zero-division test fails. Find out why and fix it.",
                provider, model="scripted-1",
            ):
                events.append(event)

        asyncio.run(run())
        timeline = agent.time_machine.timeline()
    finally:
        agent.close()

    runs = [e for e in events if e["type"] == "tool_end" and e["name"] == "run_command"]
    assert len(runs) == 2, [e["name"] for e in events if e["type"] == "tool_end"]

    # The first run must genuinely fail, and the failure must reach the model.
    assert not runs[0]["ok"]
    assert "ZeroDivisionError" in runs[0]["text"] or "failed" in runs[0]["text"].lower()

    # After the patch, the same command must pass.
    assert runs[1]["ok"], runs[1]["text"]
    assert "2 passed" in runs[1]["text"]

    assert "if b == 0" in (broken_project / "calc.py").read_text(encoding="utf-8")

    # The pre-edit state stays reachable, so a wrong fix is one click back.
    assert timeline["count"] >= 1
    edit = next(e for e in events if e["type"] == "tool_end" and e["name"] == "edit_file")
    before = edit["data"]["snapshot_before"]
    restored = agent.time_machine.file_at(before, "calc.py")
    assert restored["success"] and "if b == 0" not in restored["content"]
