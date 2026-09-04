"""
Dataset Preparation CLI for URA-Shree.
Collects, cleans, deduplicates, tokenizes, and serializes training and validation
datasets into memory-mapped binary (.bin) files.
"""

import os
import sys
import argparse
from typing import Tuple, List, Dict, Optional
import numpy as np

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import BPETokenizer
from data.preprocessing import clean_text, deduplicate_corpus, DataFilter, format_agent_document


# ==============================================================================
# Diverse, Permissive Seed Training Corpus:
# Algorithms, Architecture, Agent Actions, Tool Traces, and Multi-language Code
# ==============================================================================
SEED_DOCUMENTS = [
    # 1. Python Algorithm: Binary Search
    """
    def binary_search(arr: list[int], target: int) -> int:
        \"\"\"
        Perform binary search on a sorted list to locate the target integer.
        Returns the zero-based index of target if found, otherwise -1.
        Time Complexity: O(log N). Space Complexity: O(1).
        \"\"\"
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_val = arr[mid]
            if mid_val == target:
                return mid
            elif mid_val < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1
    """,

    # 2. Python Data Structure: LRU Cache
    """
    class Node:
        def __init__(self, key: int, value: int):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

    class LRUCache:
        def __init__(self, capacity: int):
            self.capacity = capacity
            self.cache = {}
            self.head = Node(0, 0)
            self.tail = Node(0, 0)
            self.head.next = self.tail
            self.tail.prev = self.head

        def _remove(self, node: Node) -> None:
            prev_node = node.prev
            next_node = node.next
            prev_node.next = next_node
            next_node.prev = prev_node

        def _add(self, node: Node) -> None:
            prev_tail = self.tail.prev
            prev_tail.next = node
            node.prev = prev_tail
            node.next = self.tail
            self.tail.prev = node

        def get(self, key: int) -> int:
            if key in self.cache:
                node = self.cache[key]
                self._remove(node)
                self._add(node)
                return node.value
            return -1

        def put(self, key: int, value: int) -> None:
            if key in self.cache:
                self._remove(self.cache[key])
            node = Node(key, value)
            self._add(node)
            self.cache[key] = node
            if len(self.cache) > self.capacity:
                lru = self.head.next
                self._remove(lru)
                del self.cache[lru.key]
    """,

    # 3. Transformer Math: Scaled Dot-Product Attention
    """
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def causal_scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
        \"\"\"
        Compute Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + M) @ V
        where M is the upper-triangular causal mask with -inf above the main diagonal.
        \"\"\"
        B, H, T, d_k = q.shape
        scale = 1.0 / math.sqrt(d_k)
        # Compute raw query-key similarity scores: shape [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over the key sequence dimension
        attention_weights = F.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            attention_weights = F.dropout(attention_weights, p=dropout_p)

        # Output projection weighted sum: shape [B, H, T, d_k]
        out = torch.matmul(attention_weights, v)
        return out, attention_weights
    """,

    # 4. Agent Dialogue: Inspection and File Editing
    format_agent_document(
        system_prompt="You are Shree, an expert AI coding agent. You inspect projects, execute safe tools, write robust tests, and fix bugs.",
        user_prompt="Add a new endpoint GET /api/v1/status that reports server uptime and system memory.",
        assistant_response="I will check the existing routes, add the status endpoint to server/api.py, and verify with pytest.",
        tool_calls=[
            {
                "invocation": 'filesystem.read(path="server/api.py")',
                "result": 'from fastapi import FastAPI\napp = FastAPI()\n@app.get("/health")\ndef health(): return {"status": "ok"}'
            },
            {
                "invocation": 'filesystem.write(path="server/api.py", content="from fastapi import FastAPI\\nimport psutil, time\\napp = FastAPI()\\nSTART_TIME = time.time()\\n\\n@app.get(\'/health\')\\ndef health():\\n    return {\'status\': \'ok\'}\\n\\n@app.get(\'/api/v1/status\')\\ndef status():\\n    return {\\n        \'uptime_seconds\': round(time.time() - START_TIME, 2),\\n        \'memory_percent\': psutil.virtual_memory().percent\\n    }\\n")',
                "result": '{"status": "success", "bytes_written": 320}'
            },
            {
                "invocation": 'terminal.execute(command="pytest tests/test_api.py")',
                "result": '=================== 3 passed in 0.42s ==================='
            }
        ]
    ),

    # 5. TypeScript / React Component
    """
    import React, { useState, useEffect } from 'react';

    interface TerminalOutputProps {
        logs: string[];
        onClear: () => void;
    }

    export const TerminalConsole: React.FC<TerminalOutputProps> = ({ logs, onClear }) => {
        const [autoScroll, setAutoScroll] = useState<boolean>(true);

        return (
            <div className="terminal-panel bg-neutral-950 text-neutral-200 font-mono text-sm p-4 rounded-lg border border-neutral-800">
                <div className="flex justify-between items-center pb-2 border-b border-neutral-800 mb-3">
                    <span className="text-xs uppercase tracking-wider text-neutral-400">Agent Terminal</span>
                    <button onClick={onClear} className="text-xs text-neutral-500 hover:text-neutral-300">Clear</button>
                </div>
                <div className="overflow-y-auto max-h-96 space-y-1">
                    {logs.map((log, index) => (
                        <div key={index} className="leading-relaxed whitespace-pre-wrap">{log}</div>
                    ))}
                </div>
            </div>
        );
    };
    """,

    # 6. Database: SQL Migrations and Indices
    """
    CREATE TABLE IF NOT EXISTS agents (
        id VARCHAR(64) PRIMARY KEY,
        name VARCHAR(128) NOT NULL,
        model_version VARCHAR(64) NOT NULL,
        status VARCHAR(32) DEFAULT 'idle',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS tool_executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id VARCHAR(64) REFERENCES agents(id) ON DELETE CASCADE,
        tool_name VARCHAR(64) NOT NULL,
        parameters TEXT NOT NULL,
        result TEXT NOT NULL,
        duration_ms INTEGER NOT NULL,
        executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_tool_agent ON tool_executions(agent_id);
    CREATE INDEX IF NOT EXISTS idx_tool_executed_at ON tool_executions(executed_at DESC);
    """,

    # 7. Systems & Git Workflow
    """
    #!/usr/bin/env bash
    set -euo pipefail

    echo "Running pre-commit pipeline for URA-Shree..."
    pytest tests/ -v
    git add -u
    git commit -m "feat(model): integrate causal self-attention with rotary embeddings"
    echo "Checks passed successfully."
    """
]


def prepare_dataset(
    tokenizer_path: str = "checkpoints/tokenizer.json",
    output_dir: str = "datasets",
    val_ratio: float = 0.1,
    min_copies: int = 1,
    source_root: str = ".",
    use_source_tree: bool = True,
) -> Tuple[str, str, int, int]:
    """
    Executes end-to-end dataset pipeline:
    1. Loads tokenizer.
    2. Cleans and deduplicates source documents.
    3. Tokenizes documents into a unified sequence.
    4. Writes train.bin and val.bin as uint16 arrays.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run Phase 2 first.")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"[Dataset Prep] Loaded tokenizer with vocabulary size: {len(tokenizer)}")

    # 2. Quality Filter & Clean
    data_filter = DataFilter(min_chars=30)
    cleaned_docs: list[str] = []

    documents = list(SEED_DOCUMENTS)
    if use_source_tree:
        # Real files beat replicated samples: a model trained on eight documents
        # repeated fifty times memorises eight documents.
        from data.source_corpus import collect_documents, corpus_stats

        source_docs = collect_documents([source_root])
        stats = corpus_stats(source_docs)
        print(f"[Dataset Prep] {stats['documents']} source documents, "
              f"{stats['characters']:,} characters from {source_root}")
        documents.extend(source_docs)

    for doc in documents:
        cleaned = clean_text(doc)
        valid, reason = data_filter.is_valid(cleaned)
        if valid:
            cleaned_docs.append(cleaned)
        else:
            print(f"[Dataset Prep] Filtered out document: {reason}")

    # Deduplicate
    unique_docs = deduplicate_corpus(cleaned_docs)
    print(f"[Dataset Prep] Processed {len(unique_docs)} high-quality unique documents.")

    # 3. Replicate documents to create a substantial training sequence
    corpus_tokens: list[int] = []
    for _ in range(min_copies):
        for doc in unique_docs:
            # Wrap each document with BOS and EOS tokens
            doc_with_bounds = f"<|bos|>\n{doc}\n<|eos|>"
            token_ids = tokenizer.encode(doc_with_bounds)
            corpus_tokens.extend(token_ids)

    total_tokens = len(corpus_tokens)
    print(f"[Dataset Prep] Total token count across corpus: {total_tokens:,}")

    # 4. Train / Validation Split
    split_idx = int(total_tokens * (1.0 - val_ratio))
    train_tokens = corpus_tokens[:split_idx]
    val_tokens = corpus_tokens[split_idx:]

    print(f"[Dataset Prep] Train tokens: {len(train_tokens):,}")
    print(f"[Dataset Prep] Val tokens  : {len(val_tokens):,}")

    # 5. Export to uint16 binary files
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"[Dataset Prep] Written {train_path} ({os.path.getsize(train_path) / 1024:.1f} KB)")
    print(f"[Dataset Prep] Written {val_path} ({os.path.getsize(val_path) / 1024:.1f} KB)")

    return train_path, val_path, len(train_tokens), len(val_tokens)


def main():
    parser = argparse.ArgumentParser(description="Prepare and tokenize datasets for URA-Shree.")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="datasets", help="Output directory for binary files")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--copies", type=int, default=1, help="Times to repeat the corpus")
    parser.add_argument("--source-root", type=str, default=".", help="Directory to draw source files from")
    parser.add_argument("--no-source", action="store_true", help="Use only the embedded seed corpus")
    args = parser.parse_args()

    prepare_dataset(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        min_copies=args.copies,
        source_root=args.source_root,
        use_source_tree=not args.no_source,
    )


if __name__ == "__main__":
    main()
