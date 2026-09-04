"""
Unit Tests for Coding Dataset Pipeline and Fine-Tuning.
Verifies trace formatting, JSON tool call syntax validity, and tokenization.
"""

import os
import sys
import json
import re
import tempfile
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.coding_corpus import CODING_TRACES, make_agent_trace, get_all_coding_documents
from data.prepare_coding_dataset import prepare_coding_dataset


def test_coding_corpus_format_and_json_validity():
    """Verify that all agent traces contain required special tags and valid JSON tool calls."""
    assert len(CODING_TRACES) >= 5

    for trace in CODING_TRACES:
        assert "<|bos|>" in trace
        assert "<|system|>" in trace
        assert "<|user|>" in trace
        assert "<|assistant|>" in trace
        assert "<|eos|>" in trace

        # Extract all <|tool_call|> blocks and ensure they are valid JSON
        tool_call_blocks = re.findall(r"<\|tool_call\|>\n(.*?)\n<\|tool_result\|>", trace, re.DOTALL)
        for block in tool_call_blocks:
            call_dict = json.loads(block.strip())
            assert "tool" in call_dict
            assert "action" in call_dict
            assert "args" in call_dict
            assert isinstance(call_dict["args"], dict)


def test_make_agent_trace_assembly():
    """Verify trace assembler builds properly ordered conversational turns."""
    sample_trace = make_agent_trace(
        task="Test task",
        steps=[
            {
                "thought": "Thinking about the task...",
                "tool_call": {"tool": "terminal", "action": "execute", "args": {"command": "ls"}},
                "tool_result": {"status": "success", "output": "file.txt"},
            }
        ],
        summary="Task complete.",
    )
    assert "<|user|>\nTest task" in sample_trace
    assert "Thinking about the task..." in sample_trace
    assert '"tool": "terminal"' in sample_trace
    assert "Task complete." in sample_trace


def test_prepare_coding_dataset_generation():
    """Verify coding dataset preparation exports valid binary token files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_p, val_p, n_train, n_val = prepare_coding_dataset(
            tokenizer_path="checkpoints/tokenizer.json",
            output_dir=tmp_dir,
            val_ratio=0.2,
            replications=5,
        )

        assert os.path.exists(train_p)
        assert os.path.exists(val_p)
        assert n_train > 0
        assert n_val > 0
        assert os.path.getsize(train_p) > 0
