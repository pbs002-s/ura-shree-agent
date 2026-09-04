"""
Persisted user settings, including provider API keys.

Keys live in `.shree/settings.json` next to the workspace, with file
permissions tightened where the OS supports it, and the directory is
gitignored. They are stored so you paste a key once instead of every launch.

Be clear about what this is not: it is obfuscation, not encryption. Anything
that can decrypt a key without asking you for a passphrase can be undone by
anyone who can read the file. The honest protection here is filesystem
permissions plus never sending the key back to the browser - the API returns a
masked preview only.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

SETTINGS_VERSION = 1

DEFAULTS: Dict[str, Any] = {
    "version": SETTINGS_VERSION,
    "active": {
        "provider": "local",
        "model": "",
        "temperature": 0.7,
        "max_tokens": 4096,
        "use_tools": True,
        "auto_approve": True,
    },
    "theme": "system",
    "providers": {},        # provider_id -> {api_key, base_url, models, selected_model}
    "local": {
        "checkpoint": "",
        "device": "",
        "quantize": False,
        "compile": False,
    },
}


def _obfuscate(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _deobfuscate(value: str) -> str:
    try:
        return base64.b64decode(value.encode("ascii")).decode("utf-8")
    except Exception:
        return ""


def mask(key: str) -> str:
    """A preview safe to show in the UI: enough to recognise, not enough to use."""
    if not key:
        return ""
    if len(key) <= 12:
        return key[:2] + "..."
    return f"{key[:6]}...{key[-4:]}"


class Settings:
    """Thread-safe JSON-backed settings with masked key readback."""

    def __init__(self, path: str):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return json.loads(json.dumps(DEFAULTS))
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return json.loads(json.dumps(DEFAULTS))

        merged = json.loads(json.dumps(DEFAULTS))
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return merged

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        try:
            # Owner read/write only. A no-op on Windows, which is why this is
            # not the primary protection.
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    # -- provider credentials ----------------------------------------------

    def set_provider(
        self,
        provider_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        models: Optional[List[Dict[str, Any]]] = None,
        selected_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            entry = self._data["providers"].setdefault(provider_id, {})
            if api_key is not None:
                # An empty string is an explicit "forget this key".
                entry["api_key"] = _obfuscate(api_key) if api_key else ""
            if base_url is not None:
                entry["base_url"] = base_url
            if models is not None:
                entry["models"] = models
            if selected_model is not None:
                entry["selected_model"] = selected_model
            self._save()
            return self.provider_public(provider_id)

    def api_key(self, provider_id: str) -> str:
        stored = self._data["providers"].get(provider_id, {}).get("api_key", "")
        if stored:
            return _deobfuscate(stored)
        # An environment variable is a reasonable fallback and keeps CI working
        # without a settings file on disk.
        return os.environ.get(f"{provider_id.upper()}_API_KEY", "")

    def base_url(self, provider_id: str) -> str:
        return self._data["providers"].get(provider_id, {}).get("base_url", "")

    def provider_public(self, provider_id: str) -> Dict[str, Any]:
        """Provider state with the key replaced by a masked preview."""
        entry = self._data["providers"].get(provider_id, {})
        key = self.api_key(provider_id)
        return {
            "provider": provider_id,
            "has_key": bool(key),
            "key_preview": mask(key),
            "base_url": entry.get("base_url", ""),
            "models": entry.get("models", []),
            "selected_model": entry.get("selected_model", ""),
        }

    def all_providers_public(self) -> Dict[str, Any]:
        ids = set(self._data["providers"]) | {"local"}
        return {pid: self.provider_public(pid) for pid in sorted(ids)}

    def forget_provider(self, provider_id: str) -> bool:
        with self._lock:
            existed = self._data["providers"].pop(provider_id, None) is not None
            if existed:
                self._save()
            return existed

    # -- general settings ---------------------------------------------------

    def update(self, section: str, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if isinstance(self._data.get(section), dict):
                self._data[section].update(values)
            else:
                self._data[section] = values
            self._save()
            return self._data[section]

    def get(self, section: str, default: Any = None) -> Any:
        return self._data.get(section, default)

    def public(self) -> Dict[str, Any]:
        """The whole settings object, safe to send to the browser."""
        return {
            "version": self._data.get("version", SETTINGS_VERSION),
            "active": self._data.get("active", {}),
            "theme": self._data.get("theme", "system"),
            "local": self._data.get("local", {}),
            "providers": self.all_providers_public(),
        }
