"""
Builds a training corpus out of real source files.

The seed corpus in `data/prepare_dataset.py` is a handful of documents repeated
fifty times. That is enough to prove the pipeline runs and nothing more: a model
trained on it memorises eight files rather than learning the shape of code.

Walking an actual source tree gives thousands of distinct lines with real
imports, real indentation and real naming, which is what a byte-level BPE needs
to learn useful merges and what the model needs to learn structure. It is still
a small corpus - a model this size wants orders of magnitude more - but it is
diverse instead of duplicated, and diversity is the part that cannot be faked.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Optional, Set

SOURCE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".md",
    ".yaml", ".yml", ".json", ".toml", ".sh", ".sql", ".rs", ".go",
}

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    "dist", "build", "datasets", "checkpoints", ".timemachine", ".shree",
    ".mypy_cache", ".ruff_cache", "site-packages", ".idea", ".vscode",
}

# Below this a file is boilerplate; above it, one file would dominate the mix.
MIN_CHARS = 200
MAX_CHARS = 60_000


def iter_source_files(root: str, suffixes: Optional[Set[str]] = None) -> Iterable[Path]:
    """Yields source files under `root`, skipping build and dependency trees."""
    allowed = suffixes or SOURCE_SUFFIXES
    root_path = Path(root).resolve()

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in allowed:
                yield path


def collect_documents(
    roots: Iterable[str],
    max_files: int = 2000,
    include_paths: bool = True,
) -> List[str]:
    """
    Reads source files into documents, deduplicated by content hash.

    With `include_paths` each document is prefixed with its path, which teaches
    the model the association between a filename and the code that belongs in it
    at effectively no token cost.
    """
    documents: List[str] = []
    seen: Set[str] = set()

    for root in roots:
        root_path = Path(root).resolve()
        for path in iter_source_files(str(root_path)):
            if len(documents) >= max_files:
                return documents
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue

            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)

            if include_paths:
                try:
                    relative = path.relative_to(root_path)
                except ValueError:
                    relative = path.name
                text = f"# file: {str(relative).replace(os.sep, '/')}\n{text}"

            documents.append(text)

    return documents


def corpus_stats(documents: List[str]) -> dict:
    total_chars = sum(len(doc) for doc in documents)
    return {
        "documents": len(documents),
        "characters": total_chars,
        "mean_chars": round(total_chars / max(1, len(documents))),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the source corpus")
    parser.add_argument("--root", default=".", help="Directory to walk")
    args = parser.parse_args()

    docs = collect_documents([args.root])
    stats = corpus_stats(docs)
    print(f"{stats['documents']} documents, {stats['characters']:,} characters "
          f"(mean {stats['mean_chars']:,})")
