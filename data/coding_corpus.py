"""
Coding Dataset & Agent Instruction Corpus for URA-Shree.
Generates structured multi-turn agent interaction traces with tool invocations,
test execution, bug repairs, and Git workflows.
"""

import json
from typing import List, Dict, Any


def make_agent_trace(
    task: str,
    steps: List[Dict[str, Any]],
    summary: str,
) -> str:
    """
    Assembles a complete multi-turn coding agent trace with standard boundary tokens.
    """
    trace = (
        "<|bos|><|system|>\n"
        "You are Shree, an autonomous AI coding agent developed under URA.\n"
        "You inspect project workspaces, execute safe tools, write robust tests, and fix bugs.\n"
        "When taking an action, output a JSON tool call enclosed between <|tool_call|> and <|tool_result|> tags.\n"
        f"<|user|>\n{task.strip()}\n"
    )

    for step in steps:
        trace += f"<|assistant|>\n{step.get('thought', '').strip()}\n"
        if "tool_call" in step:
            trace += f"<|tool_call|>\n{json.dumps(step['tool_call'], ensure_ascii=False)}\n"
            trace += f"<|tool_result|>\n{json.dumps(step['tool_result'], ensure_ascii=False)}\n"

    trace += f"<|assistant|>\n{summary.strip()}\n<|eos|>"
    return trace


# Diverse coding agent workflows across Python, TypeScript, Shell, and Git
CODING_TRACES: List[str] = [
    # Trace 1: File Inspection and Math Utility Creation
    make_agent_trace(
        task="Create a utility module `utils/math_helpers.py` with is_prime and gcd functions, plus unit tests in `tests/test_math.py`.",
        steps=[
            {
                "thought": "I will first inspect the repository layout to confirm the directory structure.",
                "tool_call": {"tool": "filesystem", "action": "list", "args": {"path": "."}},
                "tool_result": {"status": "success", "files": ["README.md", "model/", "tests/"]},
            },
            {
                "thought": "I will create `utils/math_helpers.py` with efficient `is_prime` and `gcd` implementations.",
                "tool_call": {
                    "tool": "filesystem",
                    "action": "write",
                    "args": {
                        "path": "utils/math_helpers.py",
                        "content": "import math\n\ndef is_prime(n: int) -> bool:\n    if n <= 1: return False\n    if n <= 3: return True\n    if n % 2 == 0 or n % 3 == 0: return False\n    for i in range(5, int(math.isqrt(n)) + 1, 6):\n        if n % i == 0 or n % (i + 2) == 0: return False\n    return True\n\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)\n"
                    }
                },
                "tool_result": {"status": "success", "bytes_written": 348},
            },
            {
                "thought": "Now I will write unit tests in `tests/test_math.py` to verify correctness.",
                "tool_call": {
                    "tool": "filesystem",
                    "action": "write",
                    "args": {
                        "path": "tests/test_math.py",
                        "content": "import pytest\nfrom utils.math_helpers import is_prime, gcd\n\ndef test_is_prime():\n    assert not is_prime(0)\n    assert not is_prime(1)\n    assert is_prime(2)\n    assert is_prime(17)\n    assert not is_prime(18)\n\ndef test_gcd():\n    assert gcd(48, 18) == 6\n    assert gcd(101, 10) == 1\n"
                    }
                },
                "tool_result": {"status": "success", "bytes_written": 286},
            },
            {
                "thought": "I will execute the test suite using the terminal tool to ensure tests pass.",
                "tool_call": {"tool": "terminal", "action": "execute", "args": {"command": "pytest tests/test_math.py -v"}},
                "tool_result": {"status": "success", "returncode": 0, "stdout": "2 passed in 0.05s"},
            }
        ],
        summary="Successfully created `utils/math_helpers.py` and verified 100% test coverage with `tests/test_math.py`."
    ),

    # Trace 2: Bug Detection, Traceback Analysis, and Patching
    make_agent_trace(
        task="The user reported an IndexError in `algorithms/quick_sort.py`. Locate the bug and fix it.",
        steps=[
            {
                "thought": "Let me read `algorithms/quick_sort.py` to analyze the implementation.",
                "tool_call": {"tool": "filesystem", "action": "read", "args": {"path": "algorithms/quick_sort.py"}},
                "tool_result": {
                    "status": "success",
                    "content": "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)]  # BUG: index out of bounds\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + mid + quick_sort(right)\n"
                },
            },
            {
                "thought": "The bug is on line 3: `pivot = arr[len(arr)]` causes an IndexError. It should be `arr[len(arr) // 2]`.",
                "tool_call": {
                    "tool": "filesystem",
                    "action": "write",
                    "args": {
                        "path": "algorithms/quick_sort.py",
                        "content": "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + mid + quick_sort(right)\n"
                    }
                },
                "tool_result": {"status": "success", "bytes_written": 235},
            },
            {
                "thought": "Now I will run pytest to ensure the fix resolves the error.",
                "tool_call": {"tool": "terminal", "action": "execute", "args": {"command": "pytest tests/test_sort.py"}},
                "tool_result": {"status": "success", "returncode": 0, "stdout": "1 passed in 0.02s"},
            }
        ],
        summary="Fixed the `IndexError` in `algorithms/quick_sort.py` by indexing the middle element `len(arr) // 2` as the pivot. All tests pass."
    ),

    # Trace 3: Git Status, Commit, and Clean Staging
    make_agent_trace(
        task="Check git status and commit our changes with a conventional commit message.",
        steps=[
            {
                "thought": "I will check git status to inspect modified files.",
                "tool_call": {"tool": "git", "action": "status", "args": {}},
                "tool_result": {"status": "success", "modified": ["model/transformer.py", "training/train.py"]},
            },
            {
                "thought": "I will stage the modified files.",
                "tool_call": {"tool": "git", "action": "add", "args": {"files": ["model/transformer.py", "training/train.py"]}},
                "tool_result": {"status": "success", "staged_count": 2},
            },
            {
                "thought": "I will commit the staged changes with an informative message.",
                "tool_call": {"tool": "git", "action": "commit", "args": {"message": "feat(model): add Pre-LN RMSNorm and AdamW training loop"}},
                "tool_result": {"status": "success", "commit_hash": "c7f1a89", "files_changed": 2},
            }
        ],
        summary="Changes have been committed as `feat(model): add Pre-LN RMSNorm and AdamW training loop` (commit c7f1a89)."
    ),

    # Trace 4: Codebase Search & Refactoring
    make_agent_trace(
        task="Find all occurrences of hardcoded '1e-5' and replace with config.norm_eps.",
        steps=[
            {
                "thought": "I will search the codebase for '1e-5'.",
                "tool_call": {"tool": "filesystem", "action": "search", "args": {"pattern": "1e-5"}},
                "tool_result": {"status": "success", "matches": [{"file": "model/transformer.py", "line": 28}]},
            },
            {
                "thought": "I will update `model/transformer.py` to use `config.norm_eps` instead of the hardcoded constant.",
                "tool_call": {
                    "tool": "filesystem",
                    "action": "replace",
                    "args": {"path": "model/transformer.py", "old": "eps=1e-5", "new": "eps=config.norm_eps"}
                },
                "tool_result": {"status": "success", "replacements": 1},
            }
        ],
        summary="Found 1 instance of hardcoded epsilon in `model/transformer.py` and refactored it to use `config.norm_eps`."
    ),

    # Trace 5: Safe Execution with Destructive Command Protection
    make_agent_trace(
        task="Clean temporary cache files.",
        steps=[
            {
                "thought": "I will safely remove pytest cache and python bytecode without deleting critical project assets.",
                "tool_call": {"tool": "terminal", "action": "execute", "args": {"command": "powershell -Command \"Remove-Item -Recurse -Force .pytest_cache\""}},
                "tool_result": {"status": "success", "returncode": 0},
            }
        ],
        summary="Cleaned temporary `.pytest_cache` directory safely."
    )
]


def get_all_coding_documents(replications: int = 40) -> List[str]:
    """
    Returns the replicated coding corpus for fine-tuning.
    """
    corpus = []
    for _ in range(replications):
        corpus.extend(CODING_TRACES)
    return corpus
