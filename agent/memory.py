"""
Local SQLite Project Memory for URA-Shree Coding Agent.
Stores architectural decisions, coding conventions, known bug fixes,
and user preferences locally without external AI memory services.
Guarantees clean connection teardown and Windows file-lock safety.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator


class ProjectMemory:
    """
    Persistent SQLite memory store for project context, architectural decisions,
    past error patterns, and user preferences.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = str(Path("checkpoints") / "shree_memory.db")
        else:
            self.db_path = db_path

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_tables()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that guarantees connection is closed on exit (prevents Windows file locks)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Initializes database schema if not already present."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS project_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    rationale TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_resolutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_signature TEXT UNIQUE NOT NULL,
                    root_cause TEXT NOT NULL,
                    fix_strategy TEXT NOT NULL,
                    success_count INTEGER DEFAULT 1,
                    last_observed TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pref_key TEXT UNIQUE NOT NULL,
                    pref_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def set_fact(self, key: str, value: str, category: str = "general") -> None:
        """Saves or updates a project fact or convention."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO project_facts (key, value, category, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    updated_at = excluded.updated_at
            """, (key, value, category, now))
            conn.commit()

    def get_fact(self, key: str) -> Optional[str]:
        """Retrieves a specific project fact."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM project_facts WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def get_facts_by_category(self, category: Optional[str] = None) -> Dict[str, str]:
        """Returns all facts, optionally filtered by category."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("SELECT key, value FROM project_facts WHERE category = ?", (category,))
            else:
                cursor.execute("SELECT key, value FROM project_facts")
            return {row["key"]: row["value"] for row in cursor.fetchall()}

    def record_decision(self, topic: str, decision: str, rationale: Optional[str] = None) -> int:
        """Logs an architectural or design decision."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO decisions (topic, decision, rationale, created_at)
                VALUES (?, ?, ?, ?)
            """, (topic, decision, rationale, now))
            conn.commit()
            return cursor.lastrowid

    def get_recent_decisions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent architectural decisions."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT topic, decision, rationale, created_at FROM decisions ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def record_error_resolution(self, error_signature: str, root_cause: str, fix_strategy: str) -> None:
        """Stores a past bug pattern and its verified resolution for quick self-healing recall."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute("""
                INSERT INTO error_resolutions (error_signature, root_cause, fix_strategy, success_count, last_observed)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(error_signature) DO UPDATE SET
                    root_cause = excluded.root_cause,
                    fix_strategy = excluded.fix_strategy,
                    success_count = success_count + 1,
                    last_observed = excluded.last_observed
            """, (error_signature, root_cause, fix_strategy, now))
            conn.commit()

    def find_error_resolution(self, error_text: str) -> Optional[Dict[str, Any]]:
        """Finds any known resolution matching substring of error_text."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT error_signature, root_cause, fix_strategy, success_count FROM error_resolutions")
            for row in cursor.fetchall():
                sig = row["error_signature"]
                if sig.lower() in error_text.lower() or error_text.lower() in sig.lower():
                    return dict(row)
            return None

    def get_summary_context(self) -> str:
        """Produces a compact context string summarizing project facts and recent decisions."""
        facts = self.get_facts_by_category()
        decisions = self.get_recent_decisions(limit=3)

        context_lines = []
        if facts:
            context_lines.append("[Project Conventions & Facts]")
            for k, v in facts.items():
                context_lines.append(f"- {k}: {v}")

        if decisions:
            context_lines.append("[Recent Architectural Decisions]")
            for d in decisions:
                context_lines.append(f"- {d['topic']}: {d['decision']}")

        return "\n".join(context_lines)
