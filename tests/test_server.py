"""
Backend API tests.

The app is imported with SHREE_WORKSPACE pointed at a temporary directory, so
nothing here writes into the real repository.
"""

import base64
import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api-workspace")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "README.md").write_text("# API test\n", encoding="utf-8")

    os.environ["SHREE_WORKSPACE"] = str(root)
    import server.api as api
    importlib.reload(api)

    with TestClient(api.app) as test_client:
        test_client.workspace = root  # type: ignore[attr-defined]
        yield test_client

    api.session_manager.close_all()
    os.environ.pop("SHREE_WORKSPACE", None)


def test_status_reports_hardware_and_runtime(client):
    body = client.get("/api/status").json()
    assert body["status"] == "online"
    assert "cuda_available" in body["hardware"]
    assert body["runtime"]["cpu_threads"] >= 1
    assert "head" in body["time_machine"]


def test_provider_catalogue_is_served(client):
    body = client.get("/api/providers").json()
    ids = {p["id"] for p in body["providers"]}
    assert {"local", "openai", "anthropic", "google", "ollama"} <= ids
    assert next(p for p in body["providers"] if p["id"] == "anthropic")["protocol"] == "anthropic"


def test_api_keys_are_stored_but_never_returned(client):
    secret = "sk-test-abcdefghijklmnop"
    saved = client.post("/api/providers/key", json={"provider": "openai", "api_key": secret}).json()

    assert saved["has_key"] and saved["key_preview"] and secret not in json.dumps(saved)
    # And the full settings payload must not leak it either.
    assert secret not in json.dumps(client.get("/api/settings").json())

    assert client.delete("/api/providers/openai").json()["removed"]
    assert not client.get("/api/settings").json()["providers"].get("openai", {}).get("has_key")


def test_scanning_a_bad_key_reports_the_reason_without_failing(client):
    body = client.post("/api/providers/scan", json={
        "provider": "custom", "base_url": "http://127.0.0.1:9/v1", "save": False,
    }).json()
    assert body["source"] in {"fallback", "none"}
    assert body["error"]


def test_local_scan_lists_checkpoints(client):
    body = client.post("/api/providers/scan", json={"provider": "local", "save": False}).json()
    assert isinstance(body["models"], list)


def test_settings_round_trip(client):
    updated = client.post("/api/settings", json={
        "section": "active", "values": {"temperature": 0.25, "use_tools": False},
    }).json()
    assert updated["active"]["temperature"] == 0.25
    assert updated["active"]["use_tools"] is False
    client.post("/api/settings", json={"section": "active", "values": {"use_tools": True}})


def test_file_tree_and_read_write(client):
    tree = client.get("/api/tree").json()["tree"]
    names = {child["name"] for child in tree["children"]}
    assert {"src", "README.md"} <= names

    read = client.get("/api/file", params={"path": "src/app.py"}).json()
    assert read["content"] == "print('hi')\n"

    client.post("/api/file", json={"path": "src/app.py", "content": "print('bye')\n"})
    assert client.get("/api/file", params={"path": "src/app.py"}).json()["content"] == "print('bye')\n"


def test_reads_outside_the_workspace_are_refused(client):
    response = client.get("/api/file", params={"path": "../../../../Windows/System32/drivers/etc/hosts"})
    assert response.status_code in (403, 404)


def test_upload_rejects_executables_and_path_escapes(client):
    payload = {
        "target_dir": "incoming",
        "files": [
            {"path": "notes.md", "content_base64": base64.b64encode(b"# notes").decode()},
            {"path": "tool.exe", "content_base64": base64.b64encode(b"MZ").decode()},
            {"path": "../../escaped.txt", "content_base64": base64.b64encode(b"no").decode()},
        ],
    }
    body = client.post("/api/upload", json=payload).json()

    assert body["count"] == 1
    assert body["written"][0]["path"] == "incoming/notes.md"
    reasons = {item["path"]: item["reason"] for item in body["rejected"]}
    assert "tool.exe" in reasons and "escaped" in " ".join(reasons)
    assert not (client.workspace.parent / "escaped.txt").exists()


def test_terminal_keeps_its_working_directory(client):
    client.post("/api/terminal", json={"command": "cd src"})
    body = client.post("/api/terminal", json={"command": "$PWD.Path"}).json()
    assert body["cwd"].endswith("/src"), body
    client.post("/api/terminal", json={"command": "cd .."})


def test_terminal_refuses_destructive_commands(client):
    body = client.post("/api/terminal", json={"command": "rm -rf /"}).json()
    assert not body["success"] and "blocked pattern" in body["error"]


def test_time_machine_snapshot_diff_and_restore(client):
    first = client.post("/api/timemachine/snapshot", json={"label": "before"}).json()

    client.post("/api/file", json={"path": "src/app.py", "content": "print('changed')\n"})
    second = client.post("/api/timemachine/snapshot", json={"label": "after"}).json()
    assert second["id"] != first["id"]

    diff = client.get("/api/timemachine/diff",
                      params={"from_id": first["id"], "to_id": second["id"]}).json()
    assert diff["summary"]["modified"] >= 1

    plan = client.post("/api/timemachine/restore",
                       json={"snapshot_id": first["id"], "dry_run": True}).json()
    assert plan["dry_run"] and plan["will_write"]

    result = client.post("/api/timemachine/restore", json={"snapshot_id": first["id"]}).json()
    assert result["success"] and result["files_written"] >= 1

    timeline = client.get("/api/timemachine").json()
    assert timeline["count"] >= 2


def test_index_returns_symbols(client):
    client.post("/api/file", json={
        "path": "src/lib.py", "content": "def helper(x):\n    return x\n",
    })
    body = client.post("/api/index").json()
    assert any(symbol["name"] == "helper" for symbol in body["symbols"])


def test_terminal_websocket_streams_output(client):
    with client.websocket_connect("/ws/terminal?session=wstest") as socket:
        assert socket.receive_json()["type"] == "ready"
        socket.send_json({"action": "run", "command": "Write-Output streamed"})

        seen, exited = [], None
        for _ in range(40):
            message = socket.receive_json()
            if message["type"] == "output":
                seen.append(message["text"])
            elif message["type"] == "exit":
                exited = message
                break

        assert exited is not None and exited["success"]
        assert any("streamed" in line for line in seen), seen


def test_agent_websocket_reports_a_missing_model(client):
    client.post("/api/settings", json={
        "section": "active", "values": {"provider": "openai", "model": ""},
    })
    with client.websocket_connect("/ws/agent?session=wstest") as socket:
        socket.send_json({"action": "chat", "message": "hello"})
        message = socket.receive_json()
        assert message["type"] == "error" and "model" in message["message"].lower()

    client.post("/api/settings", json={
        "section": "active", "values": {"provider": "local", "model": ""},
    })
