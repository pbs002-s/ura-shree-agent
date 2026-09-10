"""
Unit tests for GitTool push and repository management in URA-Shree.
"""

import os
import subprocess
import pytest
from tools.git import GitTool


def test_git_tool_non_repo(tmp_path):
    git = GitTool(str(tmp_path))
    assert not git.is_git_repo()
    status = git.status()
    assert status["success"]
    assert status["is_clean"]


def test_git_tool_workflow_and_push(tmp_path):
    # Initialize a local git repository
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init"], cwd=str(workspace), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(workspace), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(workspace), check=True)

    git = GitTool(str(workspace))
    assert git.is_git_repo()

    # Create a file
    test_file = workspace / "math.py"
    test_file.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    status = git.status()
    assert not status["is_clean"]
    assert "math.py" in status["untracked"]

    # Stage file
    add_res = git.add(["math.py"])
    assert add_res["success"]

    # Commit
    commit_res = git.commit("feat(math): add basic addition")
    assert commit_res["success"]
    assert commit_res["commit_hash"] != ""

    # Log
    log_res = git.log(max_count=1)
    assert log_res["success"]
    assert len(log_res["commits"]) == 1
    assert "add basic addition" in log_res["commits"][0]

    # Test push without remote returns clean failure dictionary
    push_res = git.push(remote="origin", branch="main")
    assert not push_res["success"]
    assert "error" in push_res
    assert push_res["remote"] == "origin"


def test_git_tool_push_to_bare_remote(tmp_path):
    # Set up a bare remote repository
    bare_remote = tmp_path / "remote.git"
    bare_remote.mkdir()
    subprocess.run(["git", "init", "--bare"], cwd=str(bare_remote), check=True, capture_output=True)

    # Set up working repository
    work_repo = tmp_path / "work"
    work_repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=str(work_repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(work_repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(work_repo), check=True)
    subprocess.run(["git", "remote", "add", "origin", str(bare_remote)], cwd=str(work_repo), check=True)

    git = GitTool(str(work_repo))

    file_a = work_repo / "solution.py"
    file_a.write_text("result = 25 + 17\nprint(result)\n", encoding="utf-8")

    git.add(["solution.py"])
    git.commit("feat: calculate basic math addition")

    # Push to bare remote
    push_res = git.push(remote="origin", branch="main", set_upstream=True)
    assert push_res["success"]
    assert push_res["branch"] == "main"
