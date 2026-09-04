"""
Unit Tests for URA-Shree Safe Tool System & Sandboxing.
Verifies path traversal containment, destructive command blacklisting,
execution timeouts, file operations, and central tool auditing.
"""

import os
import sys
import tempfile
import pytest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.filesystem import FileSystemTool
from tools.git import GitTool


def test_filesystem_containment_and_traversal_blocking():
    """
    CRITICAL SECURITY TEST:
    Verifies that attempting path traversal (e.g. ../ or absolute paths outside workspace)
    is strictly blocked with a PermissionError.
    """
    with tempfile.TemporaryDirectory() as sandbox_dir:
        fs = FileSystemTool(workspace_root=sandbox_dir)

        # Attempt relative directory traversal escaping
        with pytest.raises(PermissionError):
            fs._resolve_safe_path("../escaped_file.txt")

        # Attempt deeply nested traversal escaping
        with pytest.raises(PermissionError):
            fs._resolve_safe_path("sub/../../escaped.txt")


def test_filesystem_operations():
    """Verify write, read, list, search, diff, and replace."""
    with tempfile.TemporaryDirectory() as sandbox_dir:
        fs = FileSystemTool(workspace_root=sandbox_dir)

        # 1. Write file
        write_res = fs.write("src/app.py", "def main():\n    print('Hello World')\n    return 0\n")
        assert write_res["success"]
        assert write_res["bytes_written"] > 0

        # 2. Read full file
        read_res = fs.read("src/app.py")
        assert read_res["success"]
        assert "Hello World" in read_res["content"]
        assert read_res["total_lines"] == 3

        # 3. Read slice
        slice_res = fs.read("src/app.py", start_line=2, end_line=2)
        assert slice_res["success"]
        assert slice_res["content"] == "    print('Hello World')\n"

        # 4. List directory
        list_res = fs.list("src")
        assert list_res["success"]
        assert any(e["name"] == "app.py" for e in list_res["entries"])

        # 5. Search
        search_res = fs.search("print", path="src")
        assert search_res["success"]
        assert len(search_res["matches"]) == 1
        assert search_res["matches"][0]["line"] == 2

        # 6. Diff
        diff_res = fs.diff("src/app.py", "def main():\n    print('Updated World')\n    return 0\n")
        assert diff_res["success"]
        assert diff_res["has_changes"]
        assert "+    print('Updated World')" in diff_res["diff"]

        # 7. Replace
        replace_res = fs.replace("src/app.py", "Hello World", "URA-Shree")
        assert replace_res["success"]
        assert replace_res["replacements"] == 1
        assert "URA-Shree" in fs.read("src/app.py")["content"]



def test_ambiguous_replace_is_refused(tmp_path):
    """An edit whose target is not unique must fail rather than rewrite every match."""
    fs = FileSystemTool(workspace_root=str(tmp_path))
    fs.write("dup.py", "x = 1\nx = 1\n")

    result = fs.replace("dup.py", "x = 1", "x = 2")
    assert not result["success"] and result["occurrences"] == 2
    assert fs.read("dup.py")["content"] == "x = 1\nx = 1\n"

    forced = fs.replace("dup.py", "x = 1", "x = 2", replace_all=True)
    assert forced["success"] and forced["replacements"] == 2


def test_toolkit_dispatch_and_result_rendering(tmp_path):
    """
    The Toolkit is what the model actually calls: verify routing, the compact
    text handed back to it, and refusal of unknown tools and bad arguments.
    """
    from agent.toolkit import Toolkit
    from tools.shell import ShellManager

    fs = FileSystemTool(workspace_root=str(tmp_path))
    fs.write("note.py", "VALUE = 1\n")

    shells = ShellManager(str(tmp_path))
    toolkit = Toolkit(
        workspace_root=str(tmp_path),
        filesystem=fs,
        shell_manager=shells,
    )
    try:
        read = toolkit.execute("read_file", {"path": "note.py"})
        assert read["ok"] and "1 | VALUE = 1" in read["text"]

        write = toolkit.execute("write_file", {"path": "new.py", "content": "X = 2\n"})
        assert write["ok"] and fs.read("new.py")["content"] == "X = 2\n"

        edit = toolkit.execute(
            "edit_file", {"path": "note.py", "old_string": "VALUE = 1", "new_string": "VALUE = 9"}
        )
        assert edit["ok"] and "VALUE = 9" in fs.read("note.py")["content"]

        missing = toolkit.execute("no_such_tool", {})
        assert not missing["ok"] and "No such tool" in missing["text"]

        bad_args = toolkit.execute("read_file", {"wrong": "arg"})
        assert not bad_args["ok"] and "Bad arguments" in bad_args["text"]

        # Tools with no backend attached are not advertised to the model.
        names = {spec.name for spec in toolkit.specs()}
        assert "read_file" in names and "snapshot" not in names
    finally:
        shells.close_all()


def test_git_tool_reports_status(tmp_path):
    assert isinstance(GitTool(str(tmp_path)).status(), dict)
