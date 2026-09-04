"""
Integration & Unit Tests for URA-Shree Autonomous Coding Agent (Shree).
Verifies local SQLite memory persistence, codebase AST indexing,
execution planning, and the end-to-end autonomous agent loop.
"""

import os
import sys
import tempfile
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.memory import ProjectMemory
from agent.indexer import CodebaseIndexer
from agent.agent import CodingAgent


def test_sqlite_project_memory():
    """Verify persistent SQLite facts, decisions, and error resolution retrieval."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_memory.db")
        memory = ProjectMemory(db_path=db_path)

        # 1. Facts
        memory.set_fact("architecture", "Decoder-only Transformer with Pre-LN", category="arch")
        memory.set_fact("indentation", "4 spaces strictly", category="style")
        assert memory.get_fact("architecture") == "Decoder-only Transformer with Pre-LN"
        assert len(memory.get_facts_by_category("style")) == 1

        # 2. Decisions
        d_id = memory.record_decision(
            topic="Tokenizer selection",
            decision="Byte-level BPE",
            rationale="Eliminates out-of-vocabulary tokens on code",
        )
        assert d_id > 0
        decisions = memory.get_recent_decisions(limit=1)
        assert len(decisions) == 1
        assert decisions[0]["topic"] == "Tokenizer selection"

        # 3. Error Resolutions & Pattern Matching
        memory.record_error_resolution(
            error_signature="IndexError: list index out of range",
            root_cause="Off-by-one indexing on pivot",
            fix_strategy="Use integer division for middle element len(arr)//2",
        )
        # Search with partial traceback
        match = memory.find_error_resolution("Traceback ... IndexError: list index out of range at line 14")
        assert match is not None
        assert match["fix_strategy"] == "Use integer division for middle element len(arr)//2"

        # 4. Re-open connection to test disk persistence
        memory_reloaded = ProjectMemory(db_path=db_path)
        assert memory_reloaded.get_fact("architecture") == "Decoder-only Transformer with Pre-LN"


def test_codebase_indexer():
    """Verify AST parsing and symbol extraction on repository files."""
    indexer = CodebaseIndexer()
    stats = indexer.scan_and_index()

    assert stats["total_files"] > 10
    assert stats["indexed_python_files"] > 5
    assert stats["total_symbols"] > 15

    # Check that core model classes were discovered
    symbols = indexer.find_symbols("ShreeTransformerLM")
    assert len(symbols) > 0
    assert any(s["file"] == "model/model.py" for s in symbols)

    # Check that tokenizer was discovered
    tok_syms = indexer.find_symbols("BPETokenizer")
    assert len(tok_syms) > 0

    # Context synthesis
    context = indexer.get_relevant_context("ShreeTransformerLM")
    assert "class `ShreeTransformerLM`" in context


def test_coding_agent_event_callback_receives_the_loop_stream(workspace, scripted):
    """The event callback must see the same stream the websocket forwards to the UI."""
    import asyncio

    seen = []
    agent = CodingAgent(
        workspace_root=str(workspace),
        memory_db=str(workspace / ".shree" / "memory.db"),
        event_callback=lambda kind, data: seen.append(kind),
        session_id="events",
    )
    provider = scripted([
        {"text": "Listing.", "tools": [("list_dir", {"path": "."})]},
        {"text": "Two entries."},
    ])

    async def run():
        async for _ in agent.stream_task("What is here?", provider, model="scripted-1"):
            pass

    try:
        asyncio.run(run())
    finally:
        agent.close()

    for expected in ("turn_start", "text", "tool_start", "tool_end", "done"):
        assert expected in seen, f"{expected} was never emitted; saw {sorted(set(seen))}"
