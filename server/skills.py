"""
Skills Management for Shree.

Allows users to manage and customize agent capabilities (skills) that inject
expert system directives, domain constraints, or specialized coding workflows
into Shree's prompt dynamically.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SKILLS: List[Dict[str, Any]] = [
    {
        "id": "ui-ux-specialist",
        "name": "Modern UI/UX Specialist",
        "description": "Enforces rich aesthetics, glassmorphism, responsive layouts, micro-animations, and clean typography.",
        "prompt": (
            "UI/UX Focus: Build modern, aesthetic, and interactive interfaces. Use curated color palettes, "
            "subtle borders, glassmorphism accents, responsive CSS layouts, and micro-interactions. Never create basic or raw MVP designs."
        ),
        "enabled": True,
        "built_in": True,
    },
    {
        "id": "security-auditor",
        "name": "Security & Code Auditor",
        "description": "Ensures input sanitization, safe file operations, OWASP best practices, and vulnerability prevention.",
        "prompt": (
            "Security Focus: Thoroughly validate inputs, prevent path traversal, avoid arbitrary code injection, "
            "and protect against sensitive data leaks in code and dependencies."
        ),
        "enabled": True,
        "built_in": True,
    },
    {
        "id": "fullstack-architect",
        "name": "Full-Stack Software Architect",
        "description": "Optimizes modular code architecture, clean API contracts, error boundaries, and typing.",
        "prompt": (
            "Architecture Focus: Write production-grade, modular, and maintainable software. Structure clean layers, "
            "document critical design decisions, use strict types, and include informative error handling."
        ),
        "enabled": True,
        "built_in": True,
    },
    {
        "id": "cli-agent-runner",
        "name": "CLI Coding Agent Runner",
        "description": "Guides terminal execution for Claude Code, Aider, Git, and automated script workflows.",
        "prompt": (
            "CLI Agent Focus: When working with terminal commands or external agents (such as Claude Code, Aider, or package CLIs), "
            "provide exact commands, flag explanations, and concise instructions."
        ),
        "enabled": True,
        "built_in": True,
    },
]


class SkillsManager:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._skills: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    # Ensure default skills exist
                    existing_ids = {s["id"] for s in data if "id" in s}
                    merged = list(data)
                    for default in DEFAULT_SKILLS:
                        if default["id"] not in existing_ids:
                            merged.append(default)
                    self._skills = merged
                    return
            except Exception:
                pass
        self._skills = list(DEFAULT_SKILLS)
        self._save()

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps(self._skills, indent=2), encoding="utf-8"
        )

    def list_skills(self) -> List[Dict[str, Any]]:
        return list(self._skills)

    def add_skill(self, name: str, description: str, prompt: str) -> Dict[str, Any]:
        skill = {
            "id": f"custom-{uuid.uuid4().hex[:8]}",
            "name": name.strip(),
            "description": description.strip(),
            "prompt": prompt.strip(),
            "enabled": True,
            "built_in": False,
        }
        self._skills.append(skill)
        self._save()
        return skill

    def toggle_skill(self, skill_id: str, enabled: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        for skill in self._skills:
            if skill["id"] == skill_id:
                skill["enabled"] = not skill["enabled"] if enabled is None else bool(enabled)
                self._save()
                return skill
        return None

    def delete_skill(self, skill_id: str) -> bool:
        initial = len(self._skills)
        self._skills = [s for s in self._skills if s["id"] != skill_id or s.get("built_in")]
        if len(self._skills) < initial:
            self._save()
            return True
        return False

    def get_active_prompt(self) -> str:
        active = [s["prompt"] for s in self._skills if s.get("enabled")]
        if not active:
            return ""
        return "Active Skills & Guidelines:\n" + "\n\n".join(f"- {p}" for p in active)
