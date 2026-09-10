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

    # Blend with clean technical dialogues (basic math, GitHub push, identity, coding)
    clean_dialogue_path = os.path.join("dataset", "clean_coding_dialogues.jsonl")
    if os.path.exists(clean_dialogue_path):
        import json
        clean_count = 0
        with open(clean_dialogue_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    user_msg, asst_msg = "", ""
                    for m in messages:
                        if m.get("role") == "user":
                            user_msg = m.get("content", "").strip()
                        elif m.get("role") == "assistant":
                            asst_msg = m.get("content", "").strip()
                    if user_msg and asst_msg:
                        # Include system-prompt and direct user-prompt variants
                        doc_a = (
                            "<|bos|><|system|>\n"
                            "You are Shree, an autonomous AI coding assistant developed under URA.\n"
                            f"<|user|>\n{user_msg}\n"
                            f"<|assistant|>\n{asst_msg}\n<|eos|>"
                        )
                        doc_b = f"<|bos|><|user|>\n{user_msg}\n<|assistant|>\n{asst_msg}\n<|eos|>"
                        is_priority = any(kw in user_msg.lower() for kw in [
                            "+", "-", "*", "/", "=", "math", "calculate", "average", "square", "percent", "git", "push", "github"
                        ])
                        multiplier = 75 if is_priority else 25
                        docs.extend([doc_a, doc_b] * multiplier)
                        clean_count += 1
                except Exception:
                    continue
        print(f"[Coding Prep] Ingested {clean_count} dialogues from {clean_dialogue_path}.")

    # Direct high-frequency pairs for math and git push
    PRIORITY_PAIRS = [
        ("What is 25 + 17?", "25 + 17 = 42."),
        ("Calculate 25 + 17", "25 + 17 = 42."),
        ("What is 15 * 8?", "15 * 8 = 120."),
        ("Calculate 15 * 8", "15 * 8 = 120."),
        ("What is 100 / 4?", "100 / 4 = 25."),
        ("Calculate 100 / 4", "100 / 4 = 25."),
        ("What is 50 - 18?", "50 - 18 = 32."),
        ("Calculate 50 - 18", "50 - 18 = 32."),
        ("What is 7 * 6?", "7 * 6 = 42."),
        ("What is 9 * 9?", "9 * 9 = 81."),
        ("What is 12 + 15?", "12 + 15 = 27."),
        ("Calculate 144 / 12", "144 / 12 = 12."),
        ("How do I push my code to GitHub?", "To push your code to GitHub:\n1. Stage changes: `git add .`\n2. Commit: `git commit -m \"Your descriptive commit message\"`\n3. Push: `git push origin main` (or your branch name)."),
        ("How to push to github", "Run:\n```bash\ngit add .\ngit commit -m \"feat: update project\"\ngit push origin main\n```"),
        ("How to push changes to git", "Run `git push origin <branch_name>` to upload your local commits to the remote repository."),
    ]
    for q, a in PRIORITY_PAIRS:
        d1 = f"<|bos|><|user|>\n{q}\n<|assistant|>\n{a}\n<|eos|>"
        d2 = f"<|bos|><|system|>\nYou are Shree, an autonomous AI coding assistant developed under URA.\n<|user|>\n{q}\n<|assistant|>\n{a}\n<|eos|>"
        docs.extend([d1, d2] * 50)

    # Crucial: Shuffle documents so tokens are uniformly distributed across training steps
    import random
    random.seed(42)
    random.shuffle(docs)

    print(f"[Coding Prep] Loaded and shuffled {len(docs)} total training documents.")

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
