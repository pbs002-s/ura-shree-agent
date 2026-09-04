"""
Confined Workspace File System Tools for URA-Shree.
Strictly confines all read, write, list, search, diff, and replace operations
to the designated workspace root, preventing directory traversal and unauthorized OS access.
"""

import os
import re
import difflib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


class FileSystemTool:
    """
    Sandboxed filesystem manager confined to a root workspace directory.
    Rejects any access outside workspace boundary.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            self.workspace_root = Path.cwd().resolve()
        else:
            self.workspace_root = Path(workspace_root).resolve()

        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: str) -> Path:
        """
        Resolves path and enforces strict containment within workspace_root.
        Raises PermissionError if path attempts directory traversal.
        """
        # Convert path to pure Path
        target = (self.workspace_root / relative_path).resolve()

        try:
            # In Python 3.9+, is_relative_to checks boundary containment
            if not target.is_relative_to(self.workspace_root):
                raise PermissionError(
                    f"Access Denied: Path '{relative_path}' resolves outside workspace boundary '{self.workspace_root}'"
                )
        except AttributeError:
            # Fallback for older Path implementations
            if not str(target).startswith(str(self.workspace_root)):
                raise PermissionError(
                    f"Access Denied: Path '{relative_path}' resolves outside workspace boundary '{self.workspace_root}'"
                )

        return target

    def read(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reads text content from a file within the workspace.
        Supports 1-indexed start_line and end_line slicing.
        """
        safe_path = self._resolve_safe_path(path)
        if not safe_path.exists():
            return {"success": False, "error": f"File does not exist: {path}"}
        if not safe_path.is_file():
            return {"success": False, "error": f"Target is a directory, not a file: {path}"}

        try:
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            if start_line is not None or end_line is not None:
                s = max(1, start_line) if start_line is not None else 1
                e = min(total_lines, end_line) if end_line is not None else total_lines
                selected_lines = lines[s - 1 : e]
                content = "".join(selected_lines)
            else:
                s = 1
                e = total_lines
                content = "".join(lines)

            return {
                "success": True,
                "path": str(safe_path.relative_to(self.workspace_root)),
                "content": content,
                "total_lines": total_lines,
                "start_line": s,
                "end_line": e,
                "size_bytes": safe_path.stat().st_size,
            }
        except Exception as err:
            return {"success": False, "error": f"Failed to read file: {str(err)}"}

    def write(
        self,
        path: str,
        content: str,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Writes content to a file inside the workspace, creating parent directories if needed.
        """
        safe_path = self._resolve_safe_path(path)

        if safe_path.exists() and not overwrite:
            return {"success": False, "error": f"File already exists and overwrite is False: {path}"}

        try:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(content)

            return {
                "success": True,
                "path": str(safe_path.relative_to(self.workspace_root)),
                "bytes_written": len(content.encode("utf-8")),
                "lines": len(content.splitlines()),
            }
        except Exception as err:
            return {"success": False, "error": f"Failed to write file: {str(err)}"}

    def list(
        self,
        path: str = ".",
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """
        Lists files and subdirectories in the specified workspace path.
        """
        safe_path = self._resolve_safe_path(path)
        if not safe_path.exists():
            return {"success": False, "error": f"Directory does not exist: {path}"}
        if not safe_path.is_dir():
            return {"success": False, "error": f"Path is a file, not a directory: {path}"}

        entries = []
        try:
            if recursive:
                for root, dirs, files in os.walk(safe_path):
                    rel_root = Path(root).relative_to(self.workspace_root)
                    for d in dirs:
                        entries.append({"name": str(rel_root / d), "type": "directory"})
                    for f in files:
                        file_path = Path(root) / f
                        entries.append({
                            "name": str(rel_root / f),
                            "type": "file",
                            "size_bytes": file_path.stat().st_size,
                        })
            else:
                for item in safe_path.iterdir():
                    entries.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size_bytes": item.stat().st_size if item.is_file() else None,
                    })

            return {
                "success": True,
                "path": str(safe_path.relative_to(self.workspace_root)),
                "count": len(entries),
                "entries": entries,
            }
        except Exception as err:
            return {"success": False, "error": f"Failed to list directory: {str(err)}"}

    def search(
        self,
        pattern: str,
        path: str = ".",
        is_regex: bool = False,
    ) -> Dict[str, Any]:
        """
        Searches files in the workspace for text or regex pattern matches.
        """
        safe_path = self._resolve_safe_path(path)
        if not safe_path.exists():
            return {"success": False, "error": f"Search path does not exist: {path}"}

        matches = []
        regex = re.compile(pattern) if is_regex else None

        search_files = [safe_path] if safe_path.is_file() else safe_path.rglob("*")

        for fpath in search_files:
            if not fpath.is_file():
                continue
            # Skip hidden, venv, and git directories
            parts = fpath.parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__") for p in parts):
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, start=1):
                        matched = False
                        if is_regex and regex.search(line):
                            matched = True
                        elif not is_regex and pattern in line:
                            matched = True

                        if matched:
                            matches.append({
                                "file": str(fpath.relative_to(self.workspace_root)),
                                "line": line_num,
                                "text": line.strip(),
                            })
                            if len(matches) >= 100:
                                break
            except Exception:
                continue

            if len(matches) >= 100:
                break

        return {
            "success": True,
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches,
        }

    def diff(self, path: str, new_content: str) -> Dict[str, Any]:
        """
        Generates a unified diff between current file content and proposed new_content.
        """
        safe_path = self._resolve_safe_path(path)
        old_content = ""
        if safe_path.exists() and safe_path.is_file():
            with open(safe_path, "r", encoding="utf-8", errors="replace") as f:
                old_content = f.read()

        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        ))

        return {
            "success": True,
            "path": path,
            "has_changes": len(diff_lines) > 0,
            "diff": "".join(diff_lines),
        }

    def replace(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> Dict[str, Any]:
        """
        Replaces an exact string inside a workspace file.

        An ambiguous target is rejected rather than guessed at. Silently
        rewriting every match of a string the caller believed was unique is how
        an edit turns into a scattered, unreviewable change.
        """
        safe_path = self._resolve_safe_path(path)
        if not safe_path.exists() or not safe_path.is_file():
            return {"success": False, "error": f"File not found: {path}"}
        if old == new:
            return {"success": False, "error": "The old and new strings are identical."}

        try:
            with open(safe_path, "r", encoding="utf-8") as f:
                content = f.read()

            occurrences = content.count(old)
            if occurrences == 0:
                return {"success": False, "error": f"Target string not found in {path}"}
            if occurrences > 1 and not replace_all:
                return {
                    "success": False,
                    "error": (
                        f"Target string appears {occurrences} times in {path}. "
                        "Include surrounding lines to make it unique, or pass replace_all."
                    ),
                    "occurrences": occurrences,
                }

            new_content = content.replace(old, new) if replace_all else content.replace(old, new, 1)

            with open(safe_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return {
                "success": True,
                "path": str(safe_path.relative_to(self.workspace_root)),
                "replacements": occurrences if replace_all else 1,
            }
        except Exception as err:
            return {"success": False, "error": f"Replace failed: {str(err)}"}
