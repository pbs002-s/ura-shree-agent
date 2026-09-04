"""
The tools the agent can call, and the dispatcher that runs them.

Two things happen here that matter beyond plumbing:

  * Every mutating call is bracketed by a Time Machine snapshot, so any edit the
    agent makes is reversible from the timeline without the user having thought
    to ask for it first.
  * Results are rendered to compact text before going back to the model. Feeding
    raw JSON blobs back into the context is how agents run out of window halfway
    through a task.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from providers.base import ToolSpec

# Tools that change the workspace. Used to decide when to snapshot and when the
# UI should ask for confirmation.
MUTATING_TOOLS = {"write_file", "edit_file", "run_command"}


def _string(description: str) -> Dict[str, Any]:
    return {"type": "string", "description": description}


TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="read_file",
        description=(
            "Read a text file from the workspace. Returns the content with line numbers. "
            "Read a file before editing it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": _string("Workspace-relative path, for example 'model/model.py'."),
                "start_line": {"type": "integer", "description": "First line to return, 1-indexed."},
                "end_line": {"type": "integer", "description": "Last line to return, inclusive."},
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="write_file",
        description=(
            "Create a file or replace its entire contents. For changing part of an "
            "existing file use edit_file instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": _string("Workspace-relative path."),
                "content": _string("The complete new contents of the file."),
            },
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        name="edit_file",
        description=(
            "Replace an exact string in a file. The old string must appear exactly once "
            "unless replace_all is set; include surrounding lines to make it unique."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": _string("Workspace-relative path."),
                "old_string": _string("Exact text to replace, including indentation."),
                "new_string": _string("Replacement text."),
                "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    ),
    ToolSpec(
        name="list_dir",
        description="List the files and folders in a workspace directory.",
        parameters={
            "type": "object",
            "properties": {
                "path": _string("Workspace-relative directory, defaults to the root."),
                "recursive": {"type": "boolean", "description": "Walk subdirectories too."},
            },
        },
    ),
    ToolSpec(
        name="search",
        description=(
            "Search file contents across the workspace. Use this to find where something "
            "is defined or used before reading whole files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": _string("Text or regular expression to find."),
                "path": _string("Directory to search under, defaults to the root."),
                "is_regex": {"type": "boolean", "description": "Treat the pattern as a regex."},
            },
            "required": ["pattern"],
        },
    ),
    ToolSpec(
        name="run_command",
        description=(
            "Run a shell command in the persistent workspace terminal. Working directory, "
            "environment variables and activated virtualenvs carry over between calls."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": _string("The command line to run."),
                "timeout": {"type": "integer", "description": "Seconds before giving up, default 120."},
            },
            "required": ["command"],
        },
    ),
    ToolSpec(
        name="find_symbols",
        description=(
            "Look up Python functions and classes by name across the indexed codebase. "
            "Faster than searching when you know roughly what the symbol is called."
        ),
        parameters={
            "type": "object",
            "properties": {"query": _string("Symbol name or fragment.")},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="git_status",
        description="Show the working tree status and the current diff summary.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="remember",
        description=(
            "Record a durable fact about this project so it survives into later sessions. "
            "Use for decisions and constraints, not for restating the code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "category": _string("Grouping, for example 'architecture' or 'conventions'."),
                "key": _string("Short identifier for the fact."),
                "value": _string("The fact itself, in one or two sentences."),
            },
            "required": ["category", "key", "value"],
        },
    ),
    ToolSpec(
        name="snapshot",
        description=(
            "Save a named point in the workspace Time Machine before attempting something "
            "risky, so it can be restored or compared later."
        ),
        parameters={
            "type": "object",
            "properties": {"label": _string("Short description of the current state.")},
            "required": ["label"],
        },
    ),
]

TOOL_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def _number_lines(content: str, start: int = 1) -> str:
    lines = content.splitlines()
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{start + i:>{width}} | {line}" for i, line in enumerate(lines))


class Toolkit:
    """Binds the tool schemas to concrete implementations for one workspace."""

    def __init__(
        self,
        workspace_root: str,
        filesystem,
        shell_manager,
        indexer=None,
        memory=None,
        git=None,
        time_machine=None,
        shell_session_id: str = "agent",
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.fs = filesystem
        self.shells = shell_manager
        self.indexer = indexer
        self.memory = memory
        self.git = git
        self.time_machine = time_machine
        self.shell_session_id = shell_session_id
        self.on_event = on_event

    def specs(self) -> List[ToolSpec]:
        available = []
        for spec in TOOL_SPECS:
            if spec.name == "find_symbols" and self.indexer is None:
                continue
            if spec.name == "git_status" and self.git is None:
                continue
            if spec.name == "remember" and self.memory is None:
                continue
            if spec.name == "snapshot" and self.time_machine is None:
                continue
            available.append(spec)
        return available

    # -- execution ----------------------------------------------------------

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs one tool and returns `{"ok": bool, "text": str, "data": dict}`.

        `text` is what goes back to the model; `data` is the structured result
        the UI renders.
        """
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            known = ", ".join(spec.name for spec in self.specs())
            return {"ok": False, "text": f"No such tool '{name}'. Available: {known}", "data": {}}

        snapshot_id = None
        if name in MUTATING_TOOLS and self.time_machine is not None:
            try:
                snap = self.time_machine.snapshot(f"Before {name}", kind="auto")
                snapshot_id = snap["id"]
            except Exception:
                # A failure to snapshot must not block the edit itself.
                snapshot_id = None

        try:
            result = handler(**arguments)
        except TypeError as err:
            return {"ok": False, "text": f"Bad arguments for {name}: {err}", "data": {}}
        except PermissionError as err:
            return {"ok": False, "text": f"Refused: {err}", "data": {}}
        except Exception as err:
            return {"ok": False, "text": f"{name} failed: {err}", "data": {}}

        if snapshot_id:
            result.setdefault("data", {})["snapshot_before"] = snapshot_id
        return result

    # -- individual tools ---------------------------------------------------

    def _tool_read_file(
        self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        result = self.fs.read(path, start_line=start_line, end_line=end_line)
        if not result.get("success"):
            return {"ok": False, "text": result.get("error", "read failed"), "data": result}

        body = _number_lines(result["content"], start=result.get("start_line", 1))
        header = f"{result['path']} ({result['total_lines']} lines)"
        return {"ok": True, "text": f"{header}\n{body}", "data": result}

    def _tool_write_file(self, path: str, content: str) -> Dict[str, Any]:
        preview = self.fs.diff(path, content)
        result = self.fs.write(path, content)
        if not result.get("success"):
            return {"ok": False, "text": result.get("error", "write failed"), "data": result}
        result["diff"] = preview.get("diff", "")
        return {
            "ok": True,
            "text": f"Wrote {result['path']} ({result['lines']} lines, {result['bytes_written']} bytes).",
            "data": result,
        }

    def _tool_edit_file(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> Dict[str, Any]:
        before = self.fs.read(path)
        result = self.fs.replace(path, old_string, new_string, replace_all=replace_all)
        if not result.get("success"):
            return {"ok": False, "text": result.get("error", "edit failed"), "data": result}

        if before.get("success"):
            after = before["content"].replace(
                old_string, new_string, -1 if replace_all else 1
            )
            result["diff"] = self.fs.diff(path, after).get("diff", "")
        return {
            "ok": True,
            "text": f"Edited {result['path']} ({result['replacements']} replacement(s)).",
            "data": result,
        }

    def _tool_list_dir(self, path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        result = self.fs.list(path, recursive=recursive)
        if not result.get("success"):
            return {"ok": False, "text": result.get("error", "list failed"), "data": result}

        lines = [
            f"{entry['name']}/" if entry["type"] == "directory" else entry["name"]
            for entry in result["entries"][:200]
        ]
        text = f"{result['path']} ({result['count']} entries)\n" + "\n".join(lines)
        if result["count"] > 200:
            text += f"\n... {result['count'] - 200} more"
        return {"ok": True, "text": text, "data": result}

    def _tool_search(self, pattern: str, path: str = ".", is_regex: bool = False) -> Dict[str, Any]:
        result = self.fs.search(pattern, path=path, is_regex=is_regex)
        matches = result.get("matches", [])
        if not matches:
            return {"ok": True, "text": f"No matches for '{pattern}'.", "data": result}
        lines = [f"{m['file']}:{m['line']}: {m['text'][:200]}" for m in matches[:60]]
        text = f"{result['match_count']} match(es) for '{pattern}'\n" + "\n".join(lines)
        return {"ok": True, "text": text, "data": result}

    def _tool_run_command(self, command: str, timeout: int = 120) -> Dict[str, Any]:
        session = self.shells.get(self.shell_session_id)
        result = session.run(command, timeout=float(timeout))

        status = "exit 0" if result["success"] else f"exit {result['returncode']}"
        output = result["output"] or "(no output)"
        # Long output is trimmed from the middle: the first lines say what ran
        # and the last lines carry the error, and the middle rarely matters.
        if len(output) > 8000:
            output = output[:4000] + "\n... trimmed ...\n" + output[-3000:]
        text = f"$ {command}\n[{status} in {result['duration_ms']}ms, cwd {result['cwd']}]\n{output}"
        if result.get("error"):
            text += f"\n{result['error']}"
        return {"ok": result["success"], "text": text, "data": result}

    def _tool_find_symbols(self, query: str) -> Dict[str, Any]:
        if not getattr(self.indexer, "file_symbols", None):
            self.indexer.scan_and_index()
        found = self.indexer.find_symbols(query)
        if not found:
            return {"ok": True, "text": f"No symbol matching '{query}'.", "data": {"symbols": []}}
        lines = [
            f"{s['file']}:{s['line']}  {s.get('kind', 'symbol')} {s['name']}{s.get('signature', '')}"
            for s in found[:40]
        ]
        return {"ok": True, "text": "\n".join(lines), "data": {"symbols": found[:40]}}

    def _tool_git_status(self) -> Dict[str, Any]:
        status = self.git.status()
        diff = self.git.diff()
        text = json.dumps(status, indent=2)[:2000]
        patch = (diff.get("diff") or "")[:4000]
        if patch:
            text += f"\n\n{patch}"
        return {"ok": True, "text": text, "data": {"status": status, "diff": diff}}

    def _tool_remember(self, category: str, key: str, value: str) -> Dict[str, Any]:
        self.memory.set_fact(key=key, value=value, category=category)
        return {
            "ok": True,
            "text": f"Remembered {category}/{key}.",
            "data": {"category": category, "key": key, "value": value},
        }

    def _tool_snapshot(self, label: str) -> Dict[str, Any]:
        snap = self.time_machine.snapshot(label, kind="agent")
        return {
            "ok": True,
            "text": f"Snapshot {snap['id']} saved ({snap['file_count']} files).",
            "data": snap,
        }
