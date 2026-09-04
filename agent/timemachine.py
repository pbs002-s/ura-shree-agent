"""
Workspace Time Machine.

Every agent action that touches the filesystem is bracketed by a snapshot of
the entire workspace, so the history of a session is a navigable tree rather
than a pile of edits you have to reason about after the fact. From any point
you can diff, restore, or branch - restoring does not throw away the future,
it creates a sibling line, so exploring an alternative approach never costs you
the one you already had.

Storage is content-addressed. A snapshot records `path -> sha256`, and each
distinct blob is written once under `objects/`, zlib-compressed. Snapshotting a
1000-file project where one file changed writes exactly one new blob, so
snapshots stay cheap enough to take on every single write.

An mtime and size index short-circuits re-hashing files that have not moved,
which is what keeps a snapshot in the low milliseconds on a warm cache.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Directories never worth versioning: build output, dependency trees, and the
# Time Machine's own store.
DEFAULT_IGNORES: Set[str] = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".next", ".turbo", ".idea",
    ".vscode", ".timemachine", "checkpoints", "datasets", ".cache", "target",
    ".gradle", "coverage", ".DS_Store", ".parcel-cache", "site-packages",
}

# Anything larger is recorded by hash but its content is not stored, so a stray
# model weight or dataset cannot blow up the object store.
MAX_BLOB_BYTES = 2 * 1024 * 1024

BINARY_SUFFIXES = {
    ".pt", ".bin", ".onnx", ".safetensors", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".ico", ".pdf", ".zip", ".gz", ".tar", ".7z", ".exe", ".dll",
    ".so", ".dylib", ".pyc", ".woff", ".woff2", ".ttf", ".mp4", ".mp3", ".db",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id           TEXT PRIMARY KEY,
    parent_id    TEXT,
    label        TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'manual',
    created_at   REAL NOT NULL,
    file_count   INTEGER NOT NULL,
    total_bytes  INTEGER NOT NULL,
    tree_json    TEXT NOT NULL,
    meta_json    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_snapshots_parent  ON snapshots(parent_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_created ON snapshots(created_at);
"""


def _now() -> float:
    return time.time()


class TimeMachine:
    """A branchable, content-addressed history of a workspace directory."""

    def __init__(
        self,
        workspace_root: str,
        store_dir: Optional[str] = None,
        extra_ignores: Optional[Iterable[str]] = None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.store_dir = Path(store_dir or (self.workspace_root / ".timemachine")).resolve()
        self.objects_dir = self.store_dir / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

        self.ignores = set(DEFAULT_IGNORES)
        if extra_ignores:
            self.ignores.update(extra_ignores)

        # Reentrant: restore() holds the lock and then calls snapshot(),
        # which needs it again to take the safety point.
        self._lock = threading.RLock()
        self._db_path = self.store_dir / "history.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

        # path -> (mtime_ns, size, sha256); avoids re-reading untouched files.
        self._hash_cache: Dict[str, Tuple[int, int, str]] = {}
        self.head: Optional[str] = self._latest_id()

    # -- object store -------------------------------------------------------

    def _object_path(self, digest: str) -> Path:
        return self.objects_dir / digest[:2] / digest

    def _store_blob(self, digest: str, data: bytes) -> None:
        path = self._object_path(digest)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp name and rename, so a crash cannot leave a truncated
        # object that would later be trusted as valid content.
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(zlib.compress(data, 6))
        tmp.replace(path)

    def _load_blob(self, digest: str) -> Optional[bytes]:
        path = self._object_path(digest)
        if not path.exists():
            return None
        try:
            return zlib.decompress(path.read_bytes())
        except zlib.error:
            return None

    # -- scanning -----------------------------------------------------------

    def _should_skip_dir(self, name: str) -> bool:
        return name in self.ignores or (name.startswith(".") and name not in {".github"})

    def _iter_files(self) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(self.workspace_root):
            dirnames[:] = [d for d in dirnames if not self._should_skip_dir(d)]
            for name in filenames:
                if name.endswith(".tmp"):
                    continue
                yield Path(dirpath) / name

    def _hash_file(self, path: Path) -> Optional[Tuple[str, str, int]]:
        """Returns (relative_path, sha256, size) or None when unreadable."""
        try:
            stat = path.stat()
        except OSError:
            return None

        rel = str(path.relative_to(self.workspace_root)).replace("\\", "/")
        cached = self._hash_cache.get(rel)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return rel, cached[2], stat.st_size

        try:
            data = path.read_bytes()
        except OSError:
            return None

        digest = hashlib.sha256(data).hexdigest()
        self._hash_cache[rel] = (stat.st_mtime_ns, stat.st_size, digest)

        if stat.st_size <= MAX_BLOB_BYTES:
            self._store_blob(digest, data)
        return rel, digest, stat.st_size

    # -- snapshots ----------------------------------------------------------

    def snapshot(
        self,
        label: str,
        kind: str = "manual",
        parent_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        skip_if_unchanged: bool = True,
    ) -> Dict[str, Any]:
        """
        Captures the current workspace as a new node in the history.

        With `skip_if_unchanged` an identical tree returns the existing snapshot
        rather than adding a duplicate node, so bracketing every tool call does
        not fill the timeline with noise.
        """
        with self._lock:
            tree: Dict[str, str] = {}
            total = 0
            for path in self._iter_files():
                entry = self._hash_file(path)
                if entry is None:
                    continue
                rel, digest, size = entry
                tree[rel] = digest
                total += size

            parent = parent_id if parent_id is not None else self.head

            if skip_if_unchanged and parent:
                previous = self._row(parent)
                if previous and json.loads(previous["tree_json"]) == tree:
                    return self._present(previous, unchanged=True)

            snapshot_id = uuid.uuid4().hex[:12]
            self._conn.execute(
                "INSERT INTO snapshots (id, parent_id, label, kind, created_at, "
                "file_count, total_bytes, tree_json, meta_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    snapshot_id, parent, label, kind, _now(),
                    len(tree), total, json.dumps(tree), json.dumps(meta or {}),
                ),
            )
            self._conn.commit()
            self.head = snapshot_id
            return self._present(self._row(snapshot_id))

    def _row(self, snapshot_id: str) -> Optional[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
        return cur.fetchone()

    def _latest_id(self) -> Optional[str]:
        cur = self._conn.execute("SELECT id FROM snapshots ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        return row["id"] if row else None

    @staticmethod
    def _present(row: sqlite3.Row, unchanged: bool = False) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "parent_id": row["parent_id"],
            "label": row["label"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "file_count": row["file_count"],
            "total_bytes": row["total_bytes"],
            "meta": json.loads(row["meta_json"]),
            "unchanged": unchanged,
        }

    def timeline(self, limit: int = 200) -> Dict[str, Any]:
        """The full history as nodes plus parent links, ready to draw as a tree."""
        cur = self._conn.execute(
            "SELECT * FROM snapshots ORDER BY created_at ASC LIMIT ?", (limit,)
        )
        rows = cur.fetchall()
        nodes = [self._present(r) for r in rows]

        children: Dict[Optional[str], int] = {}
        for node in nodes:
            children[node["parent_id"]] = children.get(node["parent_id"], 0) + 1
        for node in nodes:
            # A node with more than one child is where the history forked.
            node["is_branch_point"] = children.get(node["id"], 0) > 1

        return {
            "nodes": nodes,
            "head": self.head,
            "count": len(nodes),
            "store_bytes": self.store_size(),
        }

    def store_size(self) -> int:
        return sum(p.stat().st_size for p in self.objects_dir.rglob("*") if p.is_file())

    # -- diffing ------------------------------------------------------------

    def _tree(self, snapshot_id: str) -> Dict[str, str]:
        row = self._row(snapshot_id)
        if row is None:
            raise KeyError(f"No snapshot '{snapshot_id}'")
        return json.loads(row["tree_json"])

    def _text(self, digest: str) -> Optional[str]:
        blob = self._load_blob(digest)
        if blob is None:
            return None
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def diff(self, from_id: str, to_id: str, max_files: int = 60) -> Dict[str, Any]:
        """Unified diff between any two snapshots, in either direction."""
        before, after = self._tree(from_id), self._tree(to_id)

        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        modified = sorted(p for p in set(before) & set(after) if before[p] != after[p])

        files: List[Dict[str, Any]] = []
        for path in (modified + added + removed)[:max_files]:
            old_text = self._text(before[path]) if path in before else ""
            new_text = self._text(after[path]) if path in after else ""

            if Path(path).suffix.lower() in BINARY_SUFFIXES or old_text is None or new_text is None:
                files.append({
                    "path": path,
                    "status": "added" if path in added else "removed" if path in removed else "modified",
                    "binary": True,
                    "diff": "",
                    "additions": 0,
                    "deletions": 0,
                })
                continue

            patch = list(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            ))
            files.append({
                "path": path,
                "status": "added" if path in added else "removed" if path in removed else "modified",
                "binary": False,
                "diff": "".join(patch),
                "additions": sum(1 for line in patch if line.startswith("+") and not line.startswith("+++")),
                "deletions": sum(1 for line in patch if line.startswith("-") and not line.startswith("---")),
            })

        return {
            "from": from_id,
            "to": to_id,
            "summary": {
                "added": len(added),
                "removed": len(removed),
                "modified": len(modified),
                "truncated": len(added) + len(removed) + len(modified) > max_files,
            },
            "files": files,
        }

    def file_at(self, snapshot_id: str, path: str) -> Dict[str, Any]:
        """The contents of one file as of a given snapshot."""
        tree = self._tree(snapshot_id)
        digest = tree.get(path)
        if digest is None:
            return {"success": False, "error": f"'{path}' did not exist at {snapshot_id}"}
        text = self._text(digest)
        if text is None:
            return {"success": False, "error": f"'{path}' is binary or was too large to store"}
        return {"success": True, "path": path, "content": text, "snapshot": snapshot_id}

    # -- restore ------------------------------------------------------------

    def restore(self, snapshot_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Puts the workspace back to `snapshot_id`.

        The current state is snapshotted first, so the timeline you are leaving
        stays reachable and the restore itself can be undone. The restored state
        is recorded as a child of the snapshot it came from, which is what turns
        a rewind into a branch instead of a destructive reset.
        """
        target = self._tree(snapshot_id)

        current: Dict[str, str] = {}
        for path in self._iter_files():
            entry = self._hash_file(path)
            if entry:
                current[entry[0]] = entry[1]

        to_write = [p for p, d in target.items() if current.get(p) != d]
        to_delete = [p for p in current if p not in target]
        missing_blobs = [p for p in to_write if self._load_blob(target[p]) is None]

        plan = {
            "snapshot": snapshot_id,
            "will_write": to_write,
            "will_delete": to_delete,
            "unrecoverable": missing_blobs,
        }
        if dry_run:
            return {"success": True, "dry_run": True, **plan}

        with self._lock:
            safety = self.snapshot(
                f"Before restoring {snapshot_id[:8]}", kind="auto-restore-point"
            )

            written, deleted = 0, 0
            for path in to_write:
                blob = self._load_blob(target[path])
                if blob is None:
                    continue
                dest = self.workspace_root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                written += 1

            for path in to_delete:
                try:
                    (self.workspace_root / path).unlink()
                    deleted += 1
                except OSError:
                    pass

            # Files changed underneath us; the mtime index is no longer valid.
            self._hash_cache.clear()

            branch = self.snapshot(
                f"Restored to {snapshot_id[:8]}",
                kind="restore",
                parent_id=snapshot_id,
                meta={"restored_from": snapshot_id, "safety_snapshot": safety["id"]},
                skip_if_unchanged=False,
            )

        return {
            "success": True,
            "dry_run": False,
            "files_written": written,
            "files_deleted": deleted,
            "unrecoverable": missing_blobs,
            "safety_snapshot": safety["id"],
            "new_head": branch["id"],
            **plan,
        }

    def prune(self, keep: int = 100) -> Dict[str, Any]:
        """Drops the oldest snapshots and every object no live tree references."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM snapshots ORDER BY created_at DESC LIMIT -1 OFFSET ?", (keep,)
            )
            stale = [row["id"] for row in cur.fetchall()]
            if stale:
                self._conn.executemany(
                    "DELETE FROM snapshots WHERE id = ?", [(sid,) for sid in stale]
                )
                # Re-parent orphans to keep the tree connected.
                self._conn.execute(
                    "UPDATE snapshots SET parent_id = NULL WHERE parent_id IN "
                    f"({','.join('?' * len(stale))})", stale
                )
                self._conn.commit()

            live: Set[str] = set()
            for row in self._conn.execute("SELECT tree_json FROM snapshots"):
                live.update(json.loads(row["tree_json"]).values())

            freed = 0
            for obj in self.objects_dir.rglob("*"):
                if obj.is_file() and obj.name not in live:
                    freed += obj.stat().st_size
                    obj.unlink()

            self.head = self._latest_id()

        return {"snapshots_removed": len(stale), "bytes_freed": freed, "head": self.head}

    def close(self) -> None:
        self._conn.close()
