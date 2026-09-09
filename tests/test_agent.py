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


def test_codebase_indexer_typescript_javascript():
    """Verify regex-based symbol extraction on TS/TSX/JS/JSX source files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        (
            open(os.path.join(tmp_dir, "widget.tsx"), "w", encoding="utf-8")
        ).write(
            "export interface WidgetProps {\n"
            "  label: string;\n"
            "}\n\n"
            "export class Widget {\n"
            "  render(props: WidgetProps): string {\n"
            "    return props.label;\n"
            "  }\n"
            "}\n\n"
            "export function formatLabel(label: string) {\n"
            "  return label.trim();\n"
            "}\n\n"
            "export const useWidget = (id: string) => {\n"
            "  return id;\n"
            "};\n"
        )
        (open(os.path.join(tmp_dir, "helpers.js"), "w", encoding="utf-8")).write(
            "function add(a, b) {\n  return a + b;\n}\n"
        )

        indexer = CodebaseIndexer(workspace_root=tmp_dir)
        stats = indexer.scan_and_index()

        assert stats["total_files"] == 2

        interfaces = indexer.find_symbols("WidgetProps")
        assert any(s["kind"] == "interface" for s in interfaces)

        classes = indexer.find_symbols("Widget")
        assert any(s["kind"] == "class" and s["name"] == "Widget" for s in classes)

        methods = indexer.find_symbols("render")
        assert any(s["kind"] == "method" for s in methods)

        functions = indexer.find_symbols("formatLabel")
        assert any(s["kind"] == "function" for s in functions)

        arrow_fns = indexer.find_symbols("useWidget")
        assert any(s["kind"] == "function" for s in arrow_fns)

        js_functions = indexer.find_symbols("add")
        assert any(s["file"] == "helpers.js" for s in js_functions)


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
