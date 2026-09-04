"""
Codebase Indexing & AST Semantic Analysis for URA-Shree.
Scans the workspace, extracts Python AST definitions (classes, functions, signatures),
and builds a compact architecture map so the agent can quickly locate relevant files.
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set


class CodebaseIndexer:
    """
    Analyzes and maps the local project codebase.
    Parses Python ASTs and builds searchable symbol tables without external dependencies.
    """

    EXCLUDED_DIRS: Set[str] = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "checkpoints",
        "node_modules",
        ".pytest_cache",
        "dist",
        "build",
        ".eggs",
    }

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            self.workspace_root = Path.cwd().resolve()
        else:
            self.workspace_root = Path(workspace_root).resolve()

        self.files: List[str] = []
        self.symbols: Dict[str, List[Dict[str, Any]]] = {}
        self.file_symbols: Dict[str, List[Dict[str, Any]]] = {}

    def scan_and_index(self) -> Dict[str, Any]:
        """
        Recursively scans workspace, catalogs files, and extracts AST definitions.
        """
        self.files = []
        self.symbols = {}
        self.file_symbols = {}

        for root, dirs, files in os.walk(self.workspace_root):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in self.EXCLUDED_DIRS and not d.startswith(".")]

            for file in files:
                fpath = Path(root) / file
                rel_path = str(fpath.relative_to(self.workspace_root)).replace("\\", "/")
                self.files.append(rel_path)

                if file.endswith(".py"):
                    self._index_python_file(fpath, rel_path)

        return {
            "total_files": len(self.files),
            "indexed_python_files": len(self.file_symbols),
            "total_symbols": sum(len(locs) for locs in self.symbols.values()),
        }

    def _index_python_file(self, fpath: Path, rel_path: str) -> None:
        """Parses a Python file with AST and records classes, functions, and imports."""
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
            tree = ast.parse(code, filename=str(fpath))
        except Exception:
            return

        file_defs: List[Dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                first_line = doc.splitlines()[0] if doc else ""
                entry = {
                    "kind": "class",
                    "name": node.name,
                    "file": rel_path,
                    "line": node.lineno,
                    "doc": first_line,
                }
                file_defs.append(entry)
                self.symbols.setdefault(node.name.lower(), []).append(entry)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Format argument signature
                args_list = [a.arg for a in node.args.args]
                sig = f"{node.name}({', '.join(args_list)})"
                doc = ast.get_docstring(node) or ""
                first_line = doc.splitlines()[0] if doc else ""
                entry = {
                    "kind": "function",
                    "name": node.name,
                    "signature": sig,
                    "file": rel_path,
                    "line": node.lineno,
                    "doc": first_line,
                }
                file_defs.append(entry)
                self.symbols.setdefault(node.name.lower(), []).append(entry)

        if file_defs:
            self.file_symbols[rel_path] = file_defs

    def get_tree_summary(self, max_depth: int = 3) -> str:
        """Returns an indented directory hierarchy tree of the workspace."""
        lines: List[str] = [f"Project: {self.workspace_root.name}/"]

        for rel_path in sorted(self.files):
            parts = rel_path.split("/")
            if len(parts) <= max_depth:
                indent = "  " * (len(parts))
                lines.append(f"{indent}├── {parts[-1]}")

        return "\n".join(lines[:60]) # Cap lines for prompt economy

    def find_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Finds matching classes and functions by name (exact or substring)."""
        q = query.strip().lower()
        results = []
        for name, defs in self.symbols.items():
            if q in name:
                results.extend(defs)
        return results

    def get_relevant_context(self, query: str) -> str:
        """
        Synthesizes a relevant architectural context snippet for a query or task prompt.
        """
        matches = self.find_symbols(query)
        if not matches:
            # Fall back to file name matching
            matched_files = [f for f in self.files if any(word in f.lower() for word in query.lower().split())]
            if matched_files:
                return f"[Relevant Files Found]: {', '.join(matched_files[:5])}"
            return ""

        context_lines = ["[Relevant Codebase Symbols]"]
        for m in matches[:6]:
            kind = m["kind"]
            name = m.get("signature", m["name"])
            f = m["file"]
            line = m["line"]
            doc = f" - {m['doc']}" if m.get("doc") else ""
            context_lines.append(f"- {kind} `{name}` ({f}:{line}){doc}")

        return "\n".join(context_lines)
