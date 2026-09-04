"""
The persistent shell.

Every assertion here is about state surviving between commands, which is the
one thing a per-command subprocess cannot do.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.shell import ShellManager, ShellSession, check_command

WINDOWS = sys.platform == "win32"


@pytest.fixture
def session(tmp_path):
    (tmp_path / "sub").mkdir()
    shell = ShellSession(str(tmp_path))
    yield shell
    shell.close()


def test_runs_a_command_and_captures_output(session):
    result = session.run("echo hello" if not WINDOWS else "Write-Output hello")
    assert result["success"] and "hello" in result["output"]
    assert result["returncode"] == 0


def test_working_directory_persists_between_commands(session):
    session.run("cd sub")
    result = session.run("$PWD.Path" if WINDOWS else "pwd")
    assert result["cwd"] == "sub", result


def test_environment_persists_between_commands(session):
    if WINDOWS:
        session.run("$env:SHREE_TEST = 'kept'")
        result = session.run("Write-Output $env:SHREE_TEST")
    else:
        session.run("export SHREE_TEST=kept")
        result = session.run("echo $SHREE_TEST")
    assert result["output"].strip() == "kept"


def test_non_zero_exit_is_reported(session):
    result = session.run("cmd /c exit 3" if WINDOWS else "exit 3")
    assert not result["success"] and result["returncode"] == 3


def test_failing_command_is_not_reported_as_success(session):
    result = session.run("Get-Item .\\nope-xyz" if WINDOWS else "cat nope-xyz")
    assert not result["success"]


def test_destructive_commands_are_refused():
    assert check_command("rm -rf /") is not None
    assert check_command("mkfs.ext4 /dev/sda") is not None
    assert check_command("shutdown /s") is not None
    assert check_command("pytest -q") is None
    assert check_command("git status") is None


def test_refusal_happens_before_execution(session, tmp_path):
    marker = tmp_path / "should-not-exist.txt"
    result = session.run(f"Write-Output x > {marker}; shutdown")
    assert not result["success"] and "blocked pattern" in result["error"]
    assert not marker.exists()


def test_a_timed_out_command_restarts_the_shell(session):
    result = session.run("Start-Sleep -Seconds 10" if WINDOWS else "sleep 10", timeout=1.5)
    assert not result["success"] and "Timed out" in result["error"]
    # The session must still be usable afterwards, not wedged.
    follow_up = session.run("Write-Output alive" if WINDOWS else "echo alive")
    assert follow_up["success"] and "alive" in follow_up["output"]


def test_manager_reuses_a_session_per_id(tmp_path):
    manager = ShellManager(str(tmp_path))
    try:
        first = manager.get("a")
        assert manager.get("a") is first
        assert manager.get("b") is not first
        assert len(manager.list_sessions()) == 2
        assert manager.close("a") and not manager.close("a")
    finally:
        manager.close_all()
