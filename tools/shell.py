"""
A persistent shell session.

The difference between this and `tools.terminal` is that one process stays
alive across commands, so `cd`, activated virtualenvs, exported variables and
shell history all persist the way they do in a real terminal. A fresh
subprocess per command cannot do that, which is why one-shot runners feel
broken the moment you type `cd ..`.

Completion is detected with a sentinel: after each command the shell is asked
to echo a unique marker and the exit code. Everything before the marker is the
command's output, and the marker carries the status - which is the standard
trick, because a pipe gives no other signal that a command has finished.
"""

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# Patterns refused outright. These are the ones with no plausible recovery,
# not a general safety net - anything narrower belongs to the caller's approval.
BANNED_PATTERNS: List[str] = [
    r"rm\s+-rf\s+/(?:\s|$)",
    r"rm\s+-rf\s+/\*",
    r"rm\s+-rf\s+~(?:\s|$)",
    r"\bmkfs\b",
    r"dd\s+if=/dev/(?:zero|u?random)\s+of=/dev/",
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;:",
    r"format\s+[a-z]:",
    r"del\s+/[fsq]\s+.*[a-z]:\\\\?(?:\s|$)",
    r"Remove-Item\s+.*-Recurse.*\s+[a-z]:\\\\?(?:\s|$)",
    r"\bshutdown\b",
    r"\breboot\b",
    r"Stop-Computer",
    r"Restart-Computer",
]
_BANNED = [re.compile(p, re.IGNORECASE) for p in BANNED_PATTERNS]

MARKER_PREFIX = "__SHREE_DONE_"
_MARKER_RE = re.compile(re.escape(MARKER_PREFIX) + r"([0-9a-f]{12})__:(-?\d+|):(True|False)")


def check_command(command: str) -> Optional[str]:
    """Returns the matched banned pattern, or None when the command is allowed."""
    for pattern in _BANNED:
        if pattern.search(command):
            return pattern.pattern
    return None


class ShellSession:
    """One long-lived shell process confined to a workspace directory."""

    def __init__(
        self,
        workspace_root: str,
        session_id: Optional[str] = None,
        max_output_chars: int = 200_000,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.max_output_chars = max_output_chars
        self.created_at = time.time()
        self.cwd = str(self.workspace_root)

        self._process: Optional[subprocess.Popen] = None
        self._output: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start()

    # -- lifecycle ----------------------------------------------------------

    def _shell_argv(self) -> List[str]:
        if sys.platform == "win32":
            pwsh = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
            # "-Command -" reads a command stream from stdin and keeps running.
            return [pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "-"]
        shell = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
        return [shell, "-i"] if shell.endswith("bash") else [shell]

    def _start(self) -> None:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["TERM"] = "dumb"          # no cursor escapes to strip out later
        env["NO_COLOR"] = "1"

        self._process = subprocess.Popen(
            self._shell_argv(),
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Merging streams preserves the real interleaving of output, which
            # is what a terminal shows. Split streams reorder under buffering.
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        self._output = queue.Queue()
        self._reader = threading.Thread(
            target=self._pump, name=f"shell-{self.session_id}", daemon=True
        )
        self._reader.start()

    def _pump(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self._output.put(line)
        except (ValueError, OSError):
            pass
        finally:
            self._output.put(None)

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def restart(self) -> None:
        """Kills the shell and starts a fresh one at the same working directory."""
        self.close()
        self._start()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    # -- execution ----------------------------------------------------------

    def _wrap(self, command: str, marker: str) -> str:
        """Appends the sentinel echo that reports completion and exit status."""
        cmd_stripped = command.strip()
        if sys.platform == "win32":
            if cmd_stripped == "claude":
                command = "Start-Process claude ; Write-Output '[Ura-Shree] Launched Claude Code in an interactive console window.'"
            elif cmd_stripped == "aider":
                command = "Start-Process cmd -ArgumentList '/c', 'aider' ; Write-Output '[Ura-Shree] Launched Aider in an interactive console window.'"

            return (
                f"{command}\n"
                f"$__ok = $?\n"
                f'Write-Output "{MARKER_PREFIX}{marker}__:$($LASTEXITCODE):$($__ok)"\n'
                f"$global:LASTEXITCODE = 0\n"
            )
        return (
            f"{command}\n"
            f"__code=$?\n"
            f'echo "{MARKER_PREFIX}{marker}__:$__code:$([ $__code -eq 0 ] && echo True || echo False)"\n'
        )

    def stream(
        self,
        command: str,
        timeout: float = 300.0,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Runs `command` and yields events as output arrives.

        Events are `{"type": "output", "text": ...}` while running, then a final
        `{"type": "exit", "code": int, "success": bool, "duration_ms": int,
          "cwd": str, "truncated": bool}`.
        """
        banned = check_command(command)
        if banned:
            yield {
                "type": "exit",
                "code": -1,
                "success": False,
                "error": f"Refused: the command matches a blocked pattern ({banned}).",
                "duration_ms": 0,
                "cwd": self.cwd,
                "truncated": False,
            }
            return

        with self._lock:
            if not self.alive:
                self._start()

            process = self._process
            if process is None or process.stdin is None:
                yield {"type": "exit", "code": -1, "success": False,
                       "error": "Shell process is not running.", "duration_ms": 0,
                       "cwd": self.cwd, "truncated": False}
                return

            marker = uuid.uuid4().hex[:12]
            started = time.perf_counter()

            try:
                process.stdin.write(self._wrap(command, marker))
                process.stdin.flush()
            except (OSError, ValueError) as err:
                yield {"type": "exit", "code": -1, "success": False,
                       "error": f"Could not write to the shell: {err}", "duration_ms": 0,
                       "cwd": self.cwd, "truncated": False}
                return

            emitted = 0
            truncated = False
            code = -1
            success = False
            deadline = started + timeout

            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    # A hung command poisons the session, since the sentinel for
                    # it may still arrive later and desynchronise the next read.
                    self.restart()
                    yield {"type": "exit", "code": -1, "success": False,
                           "error": f"Timed out after {timeout:.0f}s; the shell was restarted.",
                           "duration_ms": int((time.perf_counter() - started) * 1000),
                           "cwd": self.cwd, "truncated": truncated}
                    return

                try:
                    line = self._output.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue

                if line is None:
                    yield {"type": "exit", "code": -1, "success": False,
                           "error": "The shell exited unexpectedly.",
                           "duration_ms": int((time.perf_counter() - started) * 1000),
                           "cwd": self.cwd, "truncated": truncated}
                    return

                match = _MARKER_RE.search(line)
                if match and match.group(1) == marker:
                    raw_code, ok_flag = match.group(2), match.group(3)
                    success = ok_flag == "True"
                    code = int(raw_code) if raw_code else (0 if success else 1)
                    if success and code != 0:
                        # A successful cmdlet leaves a stale native exit code.
                        code = 0
                    if not success and code == 0:
                        code = 1
                    # Emit anything that shared the marker's line.
                    head = line[: match.start()]
                    if head.strip():
                        yield {"type": "output", "text": head}
                    break

                if emitted >= self.max_output_chars:
                    truncated = True
                    continue

                emitted += len(line)
                if on_output:
                    on_output(line)
                yield {"type": "output", "text": line}

            self._refresh_cwd()
            yield {
                "type": "exit",
                "code": code,
                "success": success and code == 0,
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "cwd": self.cwd,
                "truncated": truncated,
            }

    def run(self, command: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Blocking convenience wrapper returning the collected output."""
        chunks: List[str] = []
        final: Dict[str, Any] = {}
        for event in self.stream(command, timeout=timeout):
            if event["type"] == "output":
                chunks.append(event["text"])
            else:
                final = event
        output = "".join(chunks).strip()
        if final.get("truncated"):
            output += f"\n... output truncated at {self.max_output_chars} characters"
        return {
            "success": final.get("success", False),
            "command": command,
            "returncode": final.get("code", -1),
            "output": output,
            "error": final.get("error", ""),
            "duration_ms": final.get("duration_ms", 0),
            "cwd": self.relative_cwd(),
        }

    def _refresh_cwd(self) -> None:
        """Asks the shell where it is now, so the UI prompt tracks `cd`."""
        process = self._process
        if process is None or process.stdin is None:
            return
        marker = uuid.uuid4().hex[:12]
        probe = "$PWD.Path" if sys.platform == "win32" else "pwd"
        try:
            process.stdin.write(self._wrap(probe, marker))
            process.stdin.flush()
        except (OSError, ValueError):
            return

        lines: List[str] = []
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            try:
                line = self._output.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                return
            match = _MARKER_RE.search(line)
            if match and match.group(1) == marker:
                break
            lines.append(line.strip())

        for candidate in reversed(lines):
            if candidate and Path(candidate).is_dir():
                self.cwd = candidate
                return

    def relative_cwd(self) -> str:
        """Current directory as a workspace-relative path, for the prompt line."""
        try:
            rel = Path(self.cwd).resolve().relative_to(self.workspace_root)
            if str(rel) == ".":
                return self.workspace_root.name
            return f"{self.workspace_root.name}/{str(rel).replace('\\', '/')}"
        except ValueError:
            return Path(self.cwd).name or self.cwd

    def info(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "relative_cwd": self.relative_cwd(),
            "alive": self.alive,
            "workspace_root": str(self.workspace_root),
            "uptime_sec": round(time.time() - self.created_at, 1),
            "shell": Path(self._shell_argv()[0]).name,
        }


class ShellManager:
    """Keeps one `ShellSession` per session id."""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self._sessions: Dict[str, ShellSession] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str = "default") -> ShellSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.alive:
                if session is not None:
                    session.close()
                session = ShellSession(self.workspace_root, session_id=session_id)
                self._sessions[session_id] = session
            return session

    def close(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [s.info() for s in self._sessions.values()]
