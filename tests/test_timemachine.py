"""
Time Machine: snapshot, diff, restore and branch.

The behaviour that matters is that a restore is non-destructive - rewinding
forks the history rather than deleting what came after it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.timemachine import TimeMachine


def test_identical_tree_does_not_create_a_second_node(workspace):
    tm = TimeMachine(str(workspace))
    try:
        first = tm.snapshot("initial")
        again = tm.snapshot("nothing changed")
        assert again["unchanged"] and again["id"] == first["id"]
    finally:
        tm.close()


def test_blobs_are_shared_between_snapshots(workspace):
    """Content addressing means an unchanged file is stored once, not per snapshot."""
    tm = TimeMachine(str(workspace))
    try:
        tm.snapshot("one")
        size_after_first = tm.store_size()

        (workspace / "README.md").write_text("# Sample\nsecond line\n", encoding="utf-8")
        tm.snapshot("two")
        growth = tm.store_size() - size_after_first

        # Only the changed README is new; calc.py must not be stored twice.
        assert 0 < growth < 400, growth
    finally:
        tm.close()


def test_diff_reports_additions_removals_and_edits(workspace):
    tm = TimeMachine(str(workspace))
    try:
        first = tm.snapshot("initial")
        (workspace / "src" / "calc.py").write_text("def divide(a, b):\n    return b\n", encoding="utf-8")
        (workspace / "src" / "extra.py").write_text("y = 2\n", encoding="utf-8")
        (workspace / "README.md").unlink()
        second = tm.snapshot("changed")

        diff = tm.diff(first["id"], second["id"])
        assert diff["summary"] == {"added": 1, "removed": 1, "modified": 1, "truncated": False}

        edited = next(f for f in diff["files"] if f["path"] == "src/calc.py")
        assert "-    return a / b" in edited["diff"]
        assert "+    return b" in edited["diff"]
        assert edited["additions"] == 1 and edited["deletions"] == 1
    finally:
        tm.close()


def test_restore_forks_instead_of_destroying_the_future(workspace):
    tm = TimeMachine(str(workspace))
    try:
        first = tm.snapshot("initial")
        (workspace / "src" / "future.py").write_text("z = 3\n", encoding="utf-8")
        second = tm.snapshot("with future.py")

        tm.restore(first["id"])
        assert not (workspace / "src" / "future.py").exists()

        # The abandoned line is still readable, which is the whole point.
        # Content is compared after normalising newlines: snapshots store the
        # exact bytes, and Python's write_text emits CRLF on Windows.
        recovered = tm.file_at(second["id"], "src/future.py")
        assert recovered["success"]
        assert recovered["content"].replace("\r\n", "\n") == "z = 3\n"

        nodes = {n["id"]: n for n in tm.timeline()["nodes"]}
        assert nodes[first["id"]]["is_branch_point"], "restoring should fork the timeline"

        # And going forward again works, so the rewind was itself reversible.
        tm.restore(second["id"])
        assert (workspace / "src" / "future.py").exists()
    finally:
        tm.close()


def test_dry_run_reports_the_plan_without_touching_disk(workspace):
    tm = TimeMachine(str(workspace))
    try:
        first = tm.snapshot("initial")
        (workspace / "src" / "temp.py").write_text("w = 4\n", encoding="utf-8")
        tm.snapshot("added temp")

        plan = tm.restore(first["id"], dry_run=True)
        assert plan["dry_run"] and "src/temp.py" in plan["will_delete"]
        assert (workspace / "src" / "temp.py").exists(), "a dry run must not modify anything"
    finally:
        tm.close()


def test_prune_drops_old_nodes_and_unreferenced_blobs(workspace):
    tm = TimeMachine(str(workspace))
    try:
        for n in range(6):
            (workspace / "README.md").write_text(f"# Sample {n}\n", encoding="utf-8")
            tm.snapshot(f"step {n}")

        before = tm.store_size()
        result = tm.prune(keep=2)

        assert result["snapshots_removed"] == 4
        assert tm.timeline()["count"] == 2
        assert result["bytes_freed"] > 0 and tm.store_size() < before
    finally:
        tm.close()
