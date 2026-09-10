"""
Inference validation test for local URA-Shree model.
Evaluates model generation accuracy on basic math queries and GitHub workflows.
"""

import os
import pytest
import torch

from inference.engine import InferenceEngine


@pytest.fixture(scope="module")
def local_engine():
    ckpt_path = "checkpoints/coding_best.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint {ckpt_path} not found")
    return InferenceEngine(
        checkpoint_path=ckpt_path,
        tokenizer_path="checkpoints/tokenizer.json",
    )


def test_basic_math_inference(local_engine):
    """Verify that model accurately calculates basic arithmetic."""
    math_tests = [
        ("What is 25 + 17?", "42"),
        ("Calculate 15 * 8", "120"),
        ("What is 100 / 4?", "25"),
        ("What is 50 - 18?", "32"),
    ]

    for question, expected_ans in math_tests:
        prompt = f"<|bos|><|user|>\n{question}\n<|assistant|>\n"
        output = local_engine.generate(
            prompt,
            max_new_tokens=30,
            temperature=0.0,  # greedy deterministic decode
        )
        assert expected_ans in output, f"Expected '{expected_ans}' in response to '{question}', got: {output}"


def test_github_push_inference(local_engine):
    """Verify that model explains GitHub push procedures correctly."""
    prompt = "<|bos|><|user|>\nHow do I push my code to GitHub?\n<|assistant|>\n"
    output = local_engine.generate(
        prompt,
        max_new_tokens=45,
        temperature=0.0,
    )
    # Check that git commands are mentioned
    lower_out = output.lower()
    assert "git" in lower_out or "push" in lower_out or "add" in lower_out, (
        f"Expected git workflow instructions in output, got: {output}"
    )
