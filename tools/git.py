"""
Git Version Control Operations Tool for URA-Shree.
Provides structured commands for inspecting git repository state, staging files,
creating commits, and reading diffs inside the workspace.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class GitTool:
    """
    Controlled Git interface for URA-Shree.
    Confined to the workspace repository.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            self.workspace_root = Path.cwd().resolve()
        else:
            self.workspace_root = Path(workspace_root).resolve()

    def _run_git(self, args: List[str]) -> Tuple[int, str, str]:
        """Executes a git command and returns (returncode, stdout, stderr)."""
        cmd = ["git"] + args
        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.workspace_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return res.returncode, res.stdout.strip(), res.stderr.strip()
        except FileNotFoundError:
            return -1, "", "Git executable not found on system PATH."
        except Exception as err:
            return -1, "", str(err)

    def is_git_repo(self) -> bool:
        """Returns True if workspace is inside a valid git repository."""
        code, _, _ = self._run_git(["rev-parse", "--is-inside-work-tree"])
        return code == 0

    def status(self) -> Dict[str, Any]:
        """
        Parses git status into categorized lists of modified, staged, and untracked files.
        """
        code, stdout, stderr = self._run_git(["status", "--porcelain"])
        if code != 0:
            return {
                "success": True,
                "clean": True,
                "message": "Workspace active (Git repository not initialized).",
                "staged": [],
                "unstaged": [],
                "untracked": [],
            }

        staged: List[str] = []
        unstaged: List[str] = []
        untracked: List[str] = []

        for line in stdout.splitlines():
            if len(line) < 3:
                continue
            index_status = line[0]
            work_tree_status = line[1]
            filepath = line[3:].strip()

            if index_status in ("M", "A", "D", "R"):
                staged.append(filepath)
            if work_tree_status in ("M", "D"):
                unstaged.append(filepath)
            if index_status == "?" and work_tree_status == "?":
                untracked.append(filepath)

        return {
            "success": True,
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
            "is_clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
        }

    def diff(self, staged: bool = False, file: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns git diff for working tree or staged changes.
        """
        args = ["diff"]
        if staged:
            args.append("--staged")
        if file:
            args.extend(["--", file])

        code, stdout, stderr = self._run_git(args)
        if code != 0:
            return {"success": False, "error": stderr}

        return {
            "success": True,
            "staged": staged,
            "file": file,
            "diff": stdout,
            "has_diff": len(stdout) > 0,
        }

    def add(self, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Stages specified files or all workspace changes.
        """
        args = ["add"]
        if files:
            args.extend(files)
        else:
            args.append(".")

        code, stdout, stderr = self._run_git(args)
        if code != 0:
            return {"success": False, "error": stderr}

        return {"success": True, "files_staged": files if files else "all"}

    def commit(self, message: str) -> Dict[str, Any]:
        """
        Creates a Git commit with the provided message.
        """
        if not message.strip():
            return {"success": False, "error": "Commit message cannot be empty."}

        code, stdout, stderr = self._run_git(["commit", "-m", message.strip()])
        if code != 0:
            return {"success": False, "error": stderr or stdout}

        # Extract commit hash if available
        commit_hash = ""
        for line in stdout.splitlines():
            if "[" in line and "]" in line:
                parts = line.split("]")
                commit_hash = parts[0].replace("[", "").strip()
                break

        return {
            "success": True,
            "message": message,
            "commit_summary": stdout,
            "commit_hash": commit_hash,
        }

    def log(self, max_count: int = 5) -> Dict[str, Any]:
        """
        Returns recent commit history.
        """
        args = ["log", f"-n{max_count}", "--pretty=format:%h - %an (%ar): %s"]
        code, stdout, stderr = self._run_git(args)
        if code != 0:
            return {"success": False, "error": stderr}

        commits = [line.strip() for line in stdout.splitlines() if line.strip()]
        return {"success": True, "commits": commits}
