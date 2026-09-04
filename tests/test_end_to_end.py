"""
Full workflow: the agent explores a project, writes a module and a test for it,
runs the suite, and records what it decided - with every step reversible.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import CodingAgent

PYTHON = sys.executable

MODULE = '''"""Small metrics helpers."""


def compute_metrics(values):
    """Returns the count, total and mean of a list of numbers."""
    if not values:
        return {"count": 0, "total": 0, "mean": 0.0}
    total = sum(values)
    return {"count": len(values), "total": total, "mean": total / len(values)}
'''

TEST = '''from metrics import compute_metrics


def test_computes_mean():
    assert compute_metrics([1, 2, 3])["mean"] == 2


def test_handles_empty():
    assert compute_metrics([])["count"] == 0
'''


def test_full_agent_workflow(tmp_path, scripted):
    (tmp_path / "README.md").write_text("# Metrics project\n", encoding="utf-8")

    provider = scripted([
        {"text": "Looking at the project.", "tools": [("list_dir", {"path": "."})]},
        {"text": "Writing the module and its test.", "tools": [
            ("write_file", {"path": "metrics.py", "content": MODULE}),
            ("write_file", {"path": "test_metrics.py", "content": TEST}),
        ]},
        {"text": "Running the suite.", "tools": [
            ("run_command", {"command": f'& "{PYTHON}" -m pytest test_metrics.py -q'}),
        ]},
        {"text": "Recording the decision.", "tools": [
            ("remember", {
                "category": "architecture",
                "key": "metrics-module",
                "value": "compute_metrics returns count, total and mean; empty input yields zeros.",
            }),
        ]},
        {"text": "Both tests pass."},
    ])

    agent = CodingAgent(
        workspace_root=str(tmp_path),
        memory_db=str(tmp_path / ".shree" / "memory.db"),
        session_id="e2e",
    )
    events = []
    try:
        async def run():
            async for event in agent.stream_task(
                "Add a metrics module with tests and run them.", provider, model="scripted-1",
            ):
                events.append(event)

        asyncio.run(run())

        symbols = agent.indexer.scan_and_index() and agent.indexer.find_symbols("compute_metrics")
        facts = agent.memory.get_facts_by_category("architecture")
        timeline = agent.time_machine.timeline()
    finally:
        agent.close()

    assert (tmp_path / "metrics.py").exists()
    assert (tmp_path / "test_metrics.py").exists()

    test_run = next(e for e in events if e["type"] == "tool_end" and e["name"] == "run_command")
    assert test_run["ok"], test_run["text"]
    assert "2 passed" in test_run["text"]

    assert any(s["name"] == "compute_metrics" for s in symbols)
    assert "metrics-module" in facts

    done = next(e for e in events if e["type"] == "done")
    assert done["stop_reason"] == "end_turn" and done["turns"] == 5

    # Every write left a restore point behind it.
    writes = [e for e in events if e["type"] == "tool_end" and e["name"] == "write_file"]
    assert all(w["data"].get("snapshot_before") for w in writes)
    assert timeline["count"] >= 1
