"""
Where a tool call actually runs.

Until now `run_command` executed on whatever machine served the request, which
is exactly right for a developer running the tool on their own laptop and
exactly wrong for a hosted service: one tenant's `find / -name '*.pem'` reads
another tenant's keys, and a fork bomb takes the API down with it.

An `ExecutionDriver` is the seam. Everything the agent can do to a machine -
run a command, read, write, list, search - goes through one, and the driver
decides where that lands:

  * `LocalDriver`      the host process, with the existing command blocklist
                       and workspace path confinement. Development default.
  * `ContainerDriver`  an ephemeral Docker sandbox per session, with its own
                       CPU, memory, PID and network limits, and only that
                       session's workspace volume mounted.

The blocklist in `tools.shell` stays in force in both. It is a guard rail
against an accidental `rm -rf /`, not a security boundary - a determined
command can always spell itself differently. The container is the boundary.

Filesystem calls in `ContainerDriver` run against the host side of the session's
bind mount rather than through `docker exec`. It is the same bytes either way,
and it keeps reads off the container round trip; containment still comes from
`FileSystemTool`, which cannot address anything outside that one volume.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from tools.filesystem import FileSystemTool
from tools.shell import ShellSession, check_command

# Where every sandbox mounts the workspace it is allowed to touch.
CONTAINER_WORKSPACE = "/workspace"


class SandboxError(RuntimeError):
    """A sandbox could not be created or reached."""


class ExecutionDriver(ABC):
    """
    One execution environment for one session.

    Implementations own the lifetime of whatever backs them. `touch()` records
    activity so the reaper can tell an idle sandbox from a busy one, and every
    public entry point calls it.
    """

    kind = "abstract"

    def __init__(self, workspace_root: str, session_id: str = "default"):
        self.workspace_root = Path(workspace_root).resolve()
        self.session_id = session_id
        self.created_at = time.time()
        self.last_used = time.time()
        self._fs = FileSystemTool(str(self.workspace_root))

    # -- lifecycle ----------------------------------------------------------

    def touch(self) -> None:
        self.last_used = time.time()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Describes the environment for the UI status line."""

    @abstractmethod
    def restart(self) -> None:
        """Throws away the shell state and starts a clean one."""

    @abstractmethod
    def close(self) -> None:
        """Releases everything this driver holds."""

    # -- commands -----------------------------------------------------------

    @abstractmethod
    def stream(self, command: str, timeout: float = 300.0) -> Iterator[Dict[str, Any]]:
        """Yields `output` events then one terminal `exit` event."""

    def run(self, command: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Blocking form of `stream`, returning the collected output."""
        self.touch()
        chunks: List[str] = []
        final: Dict[str, Any] = {}
        for event in self.stream(command, timeout=timeout):
            if event.get("type") == "output":
                chunks.append(event.get("text", ""))
            else:
                final = event

        output = "".join(chunks).strip()
        if final.get("truncated"):
            output += "\n... output was truncated"
        return {
            "success": final.get("success", False),
            "command": command,
            "returncode": final.get("code", -1),
            "output": output,
            "error": final.get("error", ""),
            "duration_ms": final.get("duration_ms", 0),
            # Workspace-relative, which is what the prompt line and the model
            # both want; the absolute path is a host detail.
            "cwd": final.get("relative_cwd") or final.get("cwd", str(self.workspace_root)),
        }

    # -- filesystem ---------------------------------------------------------
    # These delegate rather than reimplement: FileSystemTool already owns
    # traversal containment, and two copies of that check is one too many.

    def read_file(self, path: str, **kwargs) -> Dict[str, Any]:
        self.touch()
        return self._fs.read(path, **kwargs)

    def write_file(self, path: str, content: str, **kwargs) -> Dict[str, Any]:
        self.touch()
        return self._fs.write(path, content, **kwargs)

    def list_dir(self, path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        self.touch()
        return self._fs.list(path, recursive=recursive)

    def search(self, pattern: str, path: str = ".", is_regex: bool = False) -> Dict[str, Any]:
        self.touch()
        return self._fs.search(pattern, path=path, is_regex=is_regex)

    def diff(self, path: str, new_content: str) -> Dict[str, Any]:
        self.touch()
        return self._fs.diff(path, new_content)

    def replace(self, path: str, old: str, new: str, replace_all: bool = False) -> Dict[str, Any]:
        self.touch()
        return self._fs.replace(path, old, new, replace_all=replace_all)

    @property
    def filesystem(self) -> FileSystemTool:
        """The confined filesystem, for callers that still expect the tool object."""
        return self._fs


class LocalDriver(ExecutionDriver):
    """
    Runs on the host, in a persistent shell rooted at the workspace.

    This is the behaviour the tool has always had. It is safe enough for a
    developer running it against their own machine and is not offered as a
    tenant boundary: use `ContainerDriver` for anything multi-user.
    """

    kind = "local"

    def __init__(self, workspace_root: str, session_id: str = "default"):
        super().__init__(workspace_root, session_id)
        self._shell = ShellSession(str(self.workspace_root), session_id=session_id)

    def info(self) -> Dict[str, Any]:
        info = self._shell.info()
        info.update({"driver": self.kind, "isolated": False, "idle_sec": round(self.idle_seconds, 1)})
        return info

    def stream(self, command: str, timeout: float = 300.0) -> Iterator[Dict[str, Any]]:
        self.touch()
        yield from self._shell.stream(command, timeout=timeout)
        self.touch()

    def restart(self) -> None:
        self.touch()
        self._shell.restart()

    def close(self) -> None:
        self._shell.close()


class ContainerDriver(ExecutionDriver):
    """
    An ephemeral Docker sandbox, one per session.

    The container is created on first use and torn down when the session goes
    idle. It gets the session's workspace volume and nothing else: no host
    filesystem, no Docker socket, no network unless a deployment opts back in,
    a CPU and memory ceiling, and a PID limit so a fork bomb kills only itself.

    The Docker CLI is driven directly instead of through the SDK. It is one
    fewer dependency for the same three commands, and every host that can run
    containers already has it.
    """

    kind = "container"

    def __init__(
        self,
        workspace_root: str,
        session_id: str = "default",
        image: str = "shree-sandbox:latest",
        cpus: float = 1.0,
        memory_mb: int = 2048,
        pids_limit: int = 256,
        network: str = "none",
        docker_bin: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        super().__init__(workspace_root, session_id)
        self.image = image
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.pids_limit = pids_limit
        self.network = network
        self.labels = dict(labels or {})
        self.docker = docker_bin or shutil.which("docker") or "docker"

        self.container_name = f"shree-{session_id}-{uuid.uuid4().hex[:8]}"
        self.container_id: Optional[str] = None
        self._shell: Optional[ShellSession] = None
        self._lock = threading.Lock()

    # -- container lifetime -------------------------------------------------

    def _docker(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.docker, *args], capture_output=True, text=True, timeout=timeout
        )

    def _ensure_container(self) -> str:
        """Creates the sandbox if it is not already up, and returns its id."""
        with self._lock:
            if self.container_id and self._container_running():
                return self.container_id

            self.workspace_root.mkdir(parents=True, exist_ok=True)
            argv = [
                "run", "--detach", "--rm",
                "--name", self.container_name,
                "--workdir", CONTAINER_WORKSPACE,
                # The workspace volume is the only path shared with the host.
                "--mount", f"type=bind,source={self.workspace_root},target={CONTAINER_WORKSPACE}",
                f"--cpus={self.cpus}",
                f"--memory={self.memory_mb}m",
                # Without this, memory pressure spills into host swap and the
                # limit stops meaning anything.
                f"--memory-swap={self.memory_mb}m",
                f"--pids-limit={self.pids_limit}",
                f"--network={self.network}",
                # Nothing in a sandbox needs to gain privileges, and the
                # default capability set is far wider than a build needs.
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--read-only",
                # Writable scratch that vanishes with the container.
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
                "--user", "1000:1000",
            ]
            for key, value in self.labels.items():
                argv += ["--label", f"{key}={value}"]
            argv += [self.image, "sleep", "infinity"]

            result = self._docker(*argv, timeout=180.0)
            if result.returncode != 0:
                raise SandboxError(
                    f"Could not start sandbox for session {self.session_id}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            self.container_id = result.stdout.strip()
            return self.container_id

    def _container_running(self) -> bool:
        if not self.container_id:
            return False
        result = self._docker("inspect", "-f", "{{.State.Running}}", self.container_id, timeout=15.0)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _ensure_shell(self) -> ShellSession:
        container = self._ensure_container()
        if self._shell is not None and self._shell.alive:
            return self._shell
        if self._shell is not None:
            self._shell.close()
        # One long-lived `docker exec` carrying a shell, so `cd`, exported
        # variables and activated virtualenvs survive between tool calls the
        # same way they do for the local driver.
        self._shell = ShellSession(
            CONTAINER_WORKSPACE,
            session_id=self.session_id,
            argv=[
                self.docker, "exec", "-i",
                "--workdir", CONTAINER_WORKSPACE,
                container, "/bin/bash",
            ],
            host_cwd=str(self.workspace_root),
            posix=True,
        )
        return self._shell

    # -- interface ----------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return {
            "driver": self.kind,
            "isolated": True,
            "session_id": self.session_id,
            "container": self.container_name,
            "container_id": (self.container_id or "")[:12],
            "running": self._container_running(),
            "image": self.image,
            "cwd": self._shell.cwd if self._shell else CONTAINER_WORKSPACE,
            "relative_cwd": self._shell.relative_cwd() if self._shell else "workspace",
            "workspace_root": CONTAINER_WORKSPACE,
            "limits": {
                "cpus": self.cpus,
                "memory_mb": self.memory_mb,
                "pids": self.pids_limit,
                "network": self.network,
            },
            "uptime_sec": round(time.time() - self.created_at, 1),
            "idle_sec": round(self.idle_seconds, 1),
        }

    def stream(self, command: str, timeout: float = 300.0) -> Iterator[Dict[str, Any]]:
        self.touch()
        banned = check_command(command)
        if banned:
            yield {
                "type": "exit", "code": -1, "success": False,
                "error": f"Refused: the command matches a blocked pattern ({banned}).",
                "duration_ms": 0, "cwd": CONTAINER_WORKSPACE, "truncated": False,
            }
            return
        try:
            shell = self._ensure_shell()
        except (SandboxError, subprocess.SubprocessError, OSError) as err:
            yield {
                "type": "exit", "code": -1, "success": False, "error": str(err),
                "duration_ms": 0, "cwd": CONTAINER_WORKSPACE, "truncated": False,
            }
            return
        yield from shell.stream(command, timeout=timeout)
        self.touch()

    def restart(self) -> None:
        self.touch()
        if self._shell is not None:
            self._shell.close()
            self._shell = None
        self._ensure_shell()

    def close(self) -> None:
        if self._shell is not None:
            self._shell.close()
            self._shell = None
        if self.container_id:
            try:
                # --rm on `run` means removal follows the stop; the timeout
                # bounds a container that ignores SIGTERM.
                self._docker("stop", "--time", "5", self.container_id, timeout=30.0)
            except (subprocess.SubprocessError, OSError):
                pass
            self.container_id = None


def build_driver(
    kind: str,
    workspace_root: str,
    session_id: str = "default",
    **options: Any,
) -> ExecutionDriver:
    """Constructs the driver a deployment asked for."""
    if kind == "container":
        return ContainerDriver(workspace_root, session_id=session_id, **options)
    if kind == "local":
        return LocalDriver(workspace_root, session_id=session_id)
    raise ValueError(f"Unknown execution driver '{kind}'. Use 'local' or 'container'.")


def docker_available(docker_bin: Optional[str] = None) -> bool:
    """Whether a container driver could start here at all."""
    binary = docker_bin or shutil.which("docker")
    if not binary:
        return False
    try:
        return subprocess.run(
            [binary, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=15.0,
        ).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False
