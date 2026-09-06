"""
Tests for the multi-tenant, hosted half of the server.

The interesting cases here are the ones where getting it wrong is silent: a
token that verifies when it should not, a workspace path that resolves outside
its root, a rate limiter that refills too fast, an audit row that can be edited
without trace. Each of those is a bug you find in an incident report rather
than in a stack trace, so each gets a test.
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import auth
from server.config import config
from server.ratelimit import Limit, RateLimiter
from server.sessions import WorkspaceSessionManager, safe_segment
from server.storage import LocalObjectStore
from tools.drivers import LocalDriver


# -- passwords ----------------------------------------------------------------

def test_password_hash_is_salted_and_verifies():
    stored = auth.hash_password("correct horse battery staple")
    assert auth.verify_password("correct horse battery staple", stored)
    assert not auth.verify_password("Correct horse battery staple", stored)
    # Two hashes of the same password must differ, or the salt is not doing
    # its job and identical passwords become visible in a dump.
    assert stored != auth.hash_password("correct horse battery staple")


def test_password_verify_rejects_junk():
    assert not auth.verify_password("anything", "")
    assert not auth.verify_password("", auth.hash_password("something"))
    assert not auth.verify_password("x", "not-a-stored-hash")


# -- tokens -------------------------------------------------------------------

def test_token_round_trip_carries_role():
    token = auth.create_token("user-1", role="admin", email="a@example.com")
    claims = auth.decode_token(token)
    assert claims["sub"] == "user-1" and claims["role"] == "admin"


def test_tampered_payload_is_rejected():
    token = auth.create_token("user-1", role="user")
    header, body, signature = token.split(".")
    forged_body = auth._b64url(b'{"sub":"user-1","role":"admin","typ":"access","exp":9999999999}')
    with pytest.raises(auth.TokenError):
        auth.decode_token(f"{header}.{forged_body}.{signature}")


def test_alg_none_token_is_rejected():
    """The classic JWT bypass: an unsigned token claiming no algorithm."""
    header = auth._b64url(b'{"alg":"none","typ":"JWT"}')
    body = auth._b64url(b'{"sub":"attacker","role":"admin","typ":"access","exp":9999999999}')
    with pytest.raises(auth.TokenError):
        auth.decode_token(f"{header}.{body}.")


def test_expired_token_is_rejected():
    with pytest.raises(auth.TokenError):
        auth.decode_token(auth.create_token("user-1", ttl=-1))


def test_refresh_token_cannot_be_used_as_access():
    refresh = auth.create_token("user-1", token_type="refresh")
    with pytest.raises(auth.TokenError):
        auth.decode_token(refresh, expect_type="access")
    assert auth.decode_token(refresh, expect_type="refresh")["sub"] == "user-1"


def test_cloud_mode_needs_an_explicit_signing_secret(monkeypatch):
    monkeypatch.setattr(config, "mode", "cloud")
    monkeypatch.setattr(config, "jwt_secret", "")
    with pytest.raises(RuntimeError, match="SHREE_JWT_SECRET"):
        config.resolved_jwt_secret()


# -- principals ---------------------------------------------------------------

class _Headers(dict):
    """Case-insensitive enough for the two lookups resolve_principal makes."""

    def get(self, key, default=""):
        return super().get(key.lower(), default)


def test_local_mode_without_a_key_is_open(monkeypatch):
    monkeypatch.setattr(config, "mode", "local")
    monkeypatch.setattr(config, "legacy_api_key", "")
    assert auth.resolve_principal(_Headers(), {}).local


def test_cloud_mode_refuses_an_anonymous_caller(monkeypatch):
    monkeypatch.setattr(config, "mode", "cloud")
    monkeypatch.setattr(config, "jwt_secret", "test-secret-value")
    with pytest.raises(auth.TokenError):
        auth.resolve_principal(_Headers(), {})


def test_bearer_token_resolves_to_its_subject(monkeypatch):
    monkeypatch.setattr(config, "mode", "cloud")
    monkeypatch.setattr(config, "jwt_secret", "test-secret-value")
    token = auth.create_token("user-9", role="user", email="nine@example.com")
    principal = auth.resolve_principal(_Headers({"authorization": f"Bearer {token}"}), {})
    assert principal.id == "user-9" and not principal.is_admin


def test_a_token_signed_with_another_secret_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "mode", "cloud")
    monkeypatch.setattr(config, "jwt_secret", "secret-one")
    token = auth.create_token("user-9")
    monkeypatch.setattr(config, "jwt_secret", "secret-two")
    with pytest.raises(auth.TokenError):
        auth.decode_token(token)


# -- workspace isolation ------------------------------------------------------

@pytest.mark.parametrize("value", ["..", ".", "../etc", "a/b", "", "a" * 80, "/abs", "x\\y"])
def test_path_segments_that_could_escape_are_refused(value):
    with pytest.raises(ValueError):
        safe_segment(value, "test id")


def test_each_tenant_gets_its_own_volume(tmp_path):
    manager = WorkspaceSessionManager(workspaces_root=tmp_path, driver_kind="local")
    first = manager.volume_for("alice", "proj-1")
    second = manager.volume_for("bob", "proj-1")
    assert first != second
    assert first.is_relative_to(tmp_path) and second.is_relative_to(tmp_path)


def test_sessions_are_keyed_by_user_and_project(tmp_path):
    manager = WorkspaceSessionManager(workspaces_root=tmp_path, driver_kind="local")
    try:
        alice = manager.get("alice", "proj")
        bob = manager.get("bob", "proj")
        assert alice.root != bob.root
        assert alice.agent is not bob.agent
        # The same pair must come back to the same live session, or a page
        # reload would silently start a second sandbox.
        assert manager.get("alice", "proj") is alice
    finally:
        manager.close_all()


def test_idle_sessions_are_reaped(tmp_path):
    manager = WorkspaceSessionManager(workspaces_root=tmp_path, driver_kind="local")
    try:
        session = manager.get("alice", "proj")
        assert manager.reap_idle(timeout=3600) == 0
        # Backdate both clocks: the session takes the lower of the two.
        session.last_used = time.time() - 10_000
        session.driver.last_used = time.time() - 10_000
        assert manager.reap_idle(timeout=1800) == 1
        assert manager.peek("alice", "proj") is None
    finally:
        manager.close_all()


# -- execution driver ---------------------------------------------------------

def test_local_driver_confines_file_access(tmp_path):
    driver = LocalDriver(str(tmp_path), session_id="test")
    try:
        assert driver.write_file("notes.txt", "hello")["success"]
        assert driver.read_file("notes.txt")["content"] == "hello"
        with pytest.raises(PermissionError):
            driver.read_file("../../etc/passwd")
    finally:
        driver.close()


def test_driver_refuses_a_blocked_command(tmp_path):
    driver = LocalDriver(str(tmp_path), session_id="test")
    try:
        result = driver.run("rm -rf /", timeout=10)
        assert not result["success"] and "blocked pattern" in result["error"]
    finally:
        driver.close()


def test_driver_activity_updates_the_idle_clock(tmp_path):
    driver = LocalDriver(str(tmp_path), session_id="test")
    try:
        driver.last_used = time.time() - 500
        driver.read_file("missing.txt")
        assert driver.idle_seconds < 5
    finally:
        driver.close()


# -- object storage -----------------------------------------------------------

def test_object_store_round_trip(tmp_path):
    store = LocalObjectStore(str(tmp_path / "objects"))
    digest = "a" * 64
    store.put(digest, b"file content")
    assert store.get(digest) == b"file content"
    assert store.exists(digest)
    assert store.total_bytes() > 0
    assert store.delete(digest) and store.get(digest) is None


def test_object_store_never_rewrites_an_existing_key(tmp_path):
    """Blobs are content-addressed, so a second write is always redundant."""
    store = LocalObjectStore(str(tmp_path / "objects"))
    store.put("b" * 64, b"first")
    store.put("b" * 64, b"second")
    assert store.get("b" * 64) == b"first"


def test_time_machine_uses_the_object_store(tmp_path):
    from agent.timemachine import TimeMachine

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('one')\n", encoding="utf-8")

    store = LocalObjectStore(str(tmp_path / "blobs"))
    machine = TimeMachine(str(workspace), object_store=store)
    try:
        snapshot = machine.snapshot("first")
        assert snapshot["file_count"] == 1
        # The blob landed in the injected store, not beside the workspace.
        assert store.total_bytes() > 0
        restored = machine.file_at(snapshot["id"], "app.py")["content"]
        # Line endings come back as the platform writes them; the content is
        # what the store is being tested for.
        assert restored.replace("\r\n", "\n") == "print('one')\n"
    finally:
        machine.close()


# -- rate limiting ------------------------------------------------------------

def test_limit_parses_and_falls_back():
    assert Limit.parse("10/60") == Limit(10, 60.0)
    assert Limit.parse("nonsense") == Limit(60, 60.0)


def test_token_bucket_blocks_a_burst_then_refills():
    async def scenario():
        limiter = RateLimiter(redis_url="")
        limit = Limit(count=3, window=1.0)
        for _ in range(3):
            allowed, _left, _retry = await limiter.consume("burst", limit)
            assert allowed
        allowed, _left, retry_after = await limiter.consume("burst", limit)
        assert not allowed and retry_after > 0

        # Continuous refill, not a window reset: after part of the window one
        # token is back, and no more than one.
        await asyncio.sleep(0.4)
        assert (await limiter.consume("burst", limit))[0]
        assert not (await limiter.consume("burst", limit))[0]

    asyncio.run(scenario())


def test_buckets_are_independent_per_identity():
    async def scenario():
        limiter = RateLimiter(redis_url="")
        limit = Limit(count=1, window=60.0)
        assert (await limiter.consume("user:a", limit))[0]
        assert not (await limiter.consume("user:a", limit))[0]
        # One noisy tenant must not spend another's budget.
        assert (await limiter.consume("user:b", limit))[0]

    asyncio.run(scenario())


# -- audit --------------------------------------------------------------------

def test_credentials_are_redacted_before_being_recorded():
    from server.db import redact_arguments

    cleaned = redact_arguments({
        "path": "src/app.py",
        "api_key": "sk-live-should-never-be-logged",
        "nested": {"password": "hunter2", "keep": "visible"},
        "blob": "x" * 9000,
    })
    assert cleaned["api_key"] == "[redacted]"
    assert cleaned["nested"]["password"] == "[redacted]"
    assert cleaned["nested"]["keep"] == "visible"
    assert cleaned["path"] == "src/app.py"
    assert len(cleaned["blob"]) < 9000


@pytest.fixture
def audit_db(tmp_path, monkeypatch):
    """A throwaway SQLite database standing in for Postgres."""
    from server import db

    pytest.importorskip("sqlmodel")
    monkeypatch.setattr(config, "mode", "cloud")
    monkeypatch.setattr(config, "database_url", f"sqlite:///{(tmp_path / 'audit.db').as_posix()}")
    db.reset_engine()
    db.init_db()
    yield db
    db.reset_engine()


def test_audit_entries_form_a_verifiable_chain(audit_db):
    for index in range(4):
        audit_db.record_tool_invocation(
            tool_name="run_command",
            arguments={"command": f"echo {index}"},
            status="ok",
            approved_by="user",
            exit_code=0,
            user_id="alice",
            project_id="proj",
            session_id="alice:proj",
        )
    report = audit_db.verify_audit_chain()
    assert report["ok"] and report["checked"] == 4


def test_editing_an_audit_row_breaks_the_chain(audit_db):
    from sqlmodel import select

    audit_db.record_tool_invocation(
        tool_name="run_command", arguments={"command": "safe"}, status="ok",
        approved_by="user", user_id="alice", project_id="proj",
    )
    audit_db.record_tool_invocation(
        tool_name="run_command", arguments={"command": "also safe"}, status="ok",
        approved_by="user", user_id="alice", project_id="proj",
    )
    assert audit_db.verify_audit_chain()["ok"]

    # Someone rewrites history to hide what really ran.
    with audit_db.session_scope() as session:
        row = session.exec(select(audit_db.AuditEntry).order_by(audit_db.AuditEntry.created_at)).first()
        row.arguments_json = '{"command": "rm -rf /"}'
        session.add(row)

    report = audit_db.verify_audit_chain()
    assert not report["ok"] and report["broken_at"]


def test_denied_tool_calls_are_recorded(audit_db, tmp_path):
    from agent.toolkit import Toolkit

    toolkit = Toolkit(
        workspace_root=str(tmp_path),
        driver=LocalDriver(str(tmp_path), session_id="audit"),
        audit_context={"user_id": "alice", "project_id": "proj", "session_id": "alice:proj"},
    )
    try:
        toolkit.record_denied("write_file", {"path": "secrets.env", "content": "x"})
        from sqlmodel import select

        with audit_db.session_scope() as session:
            rows = list(session.exec(select(audit_db.AuditEntry)))
        assert any(r.status == "denied" and r.tool_name == "write_file" for r in rows)
    finally:
        toolkit.driver.close()


# -- OAuth login CSRF ---------------------------------------------------------

def test_oauth_state_is_single_use_and_required():
    state = "issued-state-value"
    auth.remember_oauth_state(state)
    assert auth.consume_oauth_state(state)
    # Replaying it must fail: a callback is redeemed exactly once.
    assert not auth.consume_oauth_state(state)
    # And a state this server never issued is refused outright.
    assert not auth.consume_oauth_state("planted-by-another-site")


def test_expired_oauth_state_is_refused(monkeypatch):
    monkeypatch.setattr(auth, "OAUTH_STATE_TTL", -1.0)
    auth.remember_oauth_state("stale")
    assert not auth.consume_oauth_state("stale")


def test_shared_api_key_does_not_grant_access_in_cloud_mode(monkeypatch):
    """
    The legacy header is a local convenience, not an identity.

    One static string that makes its holder an administrator would show up in
    the audit log as nobody in particular, which defeats the point of having
    one.
    """
    monkeypatch.setattr(config, "mode", "cloud")
    monkeypatch.setattr(config, "jwt_secret", "test-secret-value")
    monkeypatch.setattr(config, "legacy_api_key", "shared-secret")
    with pytest.raises(auth.TokenError):
        auth.resolve_principal(_Headers({"x-api-key": "shared-secret"}), {})


def test_wrong_api_key_is_refused_in_local_mode(monkeypatch):
    monkeypatch.setattr(config, "mode", "local")
    monkeypatch.setattr(config, "legacy_api_key", "shared-secret")
    with pytest.raises(auth.TokenError):
        auth.resolve_principal(_Headers({"x-api-key": "wrong"}), {})
    assert auth.resolve_principal(_Headers({"x-api-key": "shared-secret"}), {}).local


# -- the deny-by-default gate -------------------------------------------------

@pytest.fixture
def cloud_client(tmp_path, monkeypatch):
    """
    The real app, flipped into cloud mode.

    Config is patched in place rather than reloaded: every module holds a
    reference to the same singleton, and reloading would hand half of them a
    second one.
    """
    from fastapi.testclient import TestClient

    import server.api as api

    monkeypatch.setattr(config, "mode", "cloud")
    monkeypatch.setattr(config, "jwt_secret", "test-secret-value")
    monkeypatch.setattr(config, "workspaces_root", tmp_path)
    monkeypatch.setattr(api.session_manager, "workspaces_root", tmp_path)
    monkeypatch.setattr(api.session_manager, "driver_kind", "local")
    with TestClient(api.app) as client:
        yield client
    api.session_manager.close_all()


def test_protected_routes_refuse_an_unauthenticated_caller(cloud_client):
    for path in ("/api/settings", "/api/tree", "/api/timemachine", "/api/admin/audit"):
        response = cloud_client.get(path)
        assert response.status_code == 401, path


def test_health_and_status_stay_public(cloud_client):
    assert cloud_client.get("/api/health").status_code == 200
    assert cloud_client.get("/api/status").status_code == 200


def test_a_valid_token_reaches_a_protected_route(cloud_client):
    token = auth.create_token("aaaa1111", role="user", email="user@example.com")
    response = cloud_client.get("/api/settings", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    # Provider keys are per user, so a fresh account carries none of another's.
    providers = response.json()["providers"]
    assert not any(entry.get("has_key") for entry in providers.values())


def test_a_plain_user_cannot_read_the_audit_log(cloud_client):
    token = auth.create_token("bbbb2222", role="user")
    response = cloud_client.get("/api/admin/audit", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_workspace_select_is_refused_in_cloud_mode(cloud_client, tmp_path):
    """The endpoint would otherwise serve any directory on the host."""
    token = auth.create_token("cccc3333", role="user")
    response = cloud_client.post(
        "/api/workspace/select",
        json={"path": str(tmp_path)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_a_project_scoped_route_needs_a_project(cloud_client):
    token = auth.create_token("dddd4444", role="user")
    response = cloud_client.get("/api/tree", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 400
    assert "project" in response.json()["detail"].lower()
