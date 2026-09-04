"""
Preparation script for Coding & Tool-Calling Dataset in URA-Shree.
Serializes instruction traces into datasets/coding_train.bin and datasets/coding_val.bin.
"""

import os
import sys
import argparse
import numpy as np
from typing import Tuple

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.tokenizer import BPETokenizer
from data.coding_corpus import get_all_coding_documents
from data.preprocessing import clean_text


def prepare_coding_dataset(
    tokenizer_path: str = "checkpoints/tokenizer.json",
    output_dir: str = "datasets",
    val_ratio: float = 0.1,
    replications: int = 50,
) -> Tuple[str, str, int, int]:
    """
    Tokenizes coding instruction traces and writes binary datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)

    # 2. Get Corpus Documents
    docs = get_all_coding_documents(replications=replications)
    print(f"[Coding Prep] Loaded {len(docs)} instruction-tuning documents.")

    # 3. Tokenize
    all_tokens: list[int] = []
    for doc in docs:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    print(f"[Coding Prep] Total tokens: {total_tokens:,}")

    # 4. Train / Val Split
    split_idx = int(total_tokens * (1.0 - val_ratio))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"[Coding Prep] Train tokens: {len(train_tokens):,}")
    print(f"[Coding Prep] Val tokens  : {len(val_tokens):,}")

    train_path = os.path.join(output_dir, "coding_train.bin")
    val_path = os.path.join(output_dir, "coding_val.bin")

    train_arr = np.array(train_tokens, dtype=np.uint16)
    val_arr = np.array(val_tokens, dtype=np.uint16)

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"[Coding Prep] Written {train_path} ({os.path.getsize(train_path) / 1024:.1f} KB)")
    print(f"[Coding Prep] Written {val_path} ({os.path.getsize(val_path) / 1024:.1f} KB)")

    return train_path, val_path, len(train_tokens), len(val_tokens)


def main():
    parser = argparse.ArgumentParser(description="Prepare coding fine-tuning dataset.")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--replications", type=int, default=50)
    args = parser.parse_args()

    prepare_coding_dataset(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        replications=args.replications,
    )


if __name__ == "__main__":
    main()
