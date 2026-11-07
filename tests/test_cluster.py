"""ClusterOne router: URL handling, input validation, safe unzip, and the HTTP surface with a fake SDK."""

import zipfile

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import server.cluster as cluster
from server.auth import Principal, current_principal


# -- pure helpers ---------------------------------------------------------------

@pytest.mark.parametrize("raw, root", [
    ("https://gpu.lab/api/v1/openai/v1", "https://gpu.lab"),
    ("https://gpu.lab/api/v1/openai/v1/", "https://gpu.lab"),
    ("http://10.0.0.5:8000", "http://10.0.0.5:8000"),
    ("", ""),
    ("gpu.lab", ""),
    ("file:///etc/passwd", ""),
    ("javascript:alert(1)", ""),
])
def test_cluster_root_accepts_only_http_urls(raw, root):
    assert cluster.cluster_root(raw) == root


def test_safe_extract_refuses_path_traversal(tmp_path):
    archive = tmp_path / "out.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "nope")
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(HTTPException) as err:
        cluster.safe_extract(archive, dest)
    assert err.value.status_code == 422
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_unpacks_normal_outputs(tmp_path):
    archive = tmp_path / "out.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("outputs/coding_best.pt", "weights")
    dest = tmp_path / "dest"
    dest.mkdir()
    cluster.safe_extract(archive, dest)
    assert (dest / "outputs" / "coding_best.pt").read_text() == "weights"


# -- HTTP surface with a fake SDK -----------------------------------------------

class FakeJob:
    def __init__(self, jid="abc123", name="shree-finetune", status="running", command=""):
        self.data = {"id": jid, "name": name, "status": status, "command": command}
        self.id, self.name, self.status = jid, name, status
        self.url = f"https://gpu.lab/jobs/{jid}"
        self.cancelled = False

    def log(self, stream="stdout", offset=0):
        return "step 10 loss 2.1\n", offset + 17, self.status

    def cancel(self):
        self.cancelled = True


class FakeClient:
    submitted: list = []

    def __init__(self, url, key):
        assert url == "https://gpu.lab" and key == "sk-cluster"

    def me(self):
        return {"username": "pritam"}

    def usage(self):
        return {"gpu_hours": 1.5}

    def nodes(self):
        return [{"name": "lab-pc-1", "gpus": 1}]

    def jobs(self, status="", mine=True, page_size=50):
        return [FakeJob()]

    def job(self, jid):
        return FakeJob(jid)

    def submit(self, path, command, name=None, **opts):
        assert (path / "checkpoints" / "best.pt").exists(), "bundle must carry the base checkpoint"
        assert not (path / "frontend").exists(), "bundle must stay slim"
        FakeClient.submitted.append((command, name, opts))
        return FakeJob(f"job{len(FakeClient.submitted)}", name, "submitting", command)


class FakeError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class FakeSettings:
    def __init__(self, base="https://gpu.lab/api/v1/openai/v1", key="sk-cluster"):
        self._base, self._key = base, key

    def base_url(self, _pid):
        return self._base

    def api_key(self, _pid):
        return self._key


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "best.pt").write_bytes(b"w")
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "train_coding.py").write_text("")
    (tmp_path / "frontend").mkdir()
    monkeypatch.setattr(cluster, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cluster, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(cluster, "DOWNLOADS", tmp_path / "outputs" / "cluster")
    monkeypatch.setattr(cluster, "_sdk", lambda: (FakeClient, FakeError))
    FakeClient.submitted = []

    def build(settings=None):
        app = FastAPI()
        app.include_router(cluster.make_router(lambda _p: settings or FakeSettings()))
        app.dependency_overrides[current_principal] = lambda: Principal(id="u1", email="u1@example.com", role="user", local=True)
        return TestClient(app)

    return build


def test_status_reports_unconfigured_without_contacting_the_cluster(make_client):
    client = make_client(FakeSettings(base="", key=""))
    body = client.get("/api/cluster/status").json()
    assert body["configured"] is False
    assert body["checkpoints"] == ["best.pt"]


def test_jobs_endpoints_refuse_when_not_configured(make_client):
    client = make_client(FakeSettings(base="", key=""))
    assert client.get("/api/cluster/jobs").status_code == 409


def test_status_and_jobs_when_configured(make_client):
    client = make_client()
    status = client.get("/api/cluster/status").json()
    assert status["configured"] and status["url"] == "https://gpu.lab"
    jobs = client.get("/api/cluster/jobs").json()["jobs"]
    assert jobs[0]["id"] == "abc123"
    log = client.get("/api/cluster/jobs/abc123/log").json()
    assert "loss" in log["text"] and log["offset"] == 17


def test_invalid_job_id_is_rejected(make_client):
    assert make_client().get("/api/cluster/jobs/..%2F..%2Fetc/log").status_code in (404, 422)
    assert make_client().get("/api/cluster/jobs/a b/log").status_code == 422


def test_finetune_builds_command_from_validated_values_only(make_client):
    client = make_client()
    r = client.post("/api/cluster/finetune", json={"steps": 300, "lr": 1e-4, "gpus": 1})
    assert r.status_code == 200, r.text
    command, name, opts = FakeClient.submitted[0]
    assert command == (
        "python training/train_coding.py --config configs/small.yaml --base checkpoints/best.pt "
        "--steps 300 --lr 0.0001 --output outputs/coding_best.pt"
    )
    assert name == "shree-finetune" and opts["gpus"] == 1


def test_finetune_sweep_submits_one_job_per_combination(make_client):
    client = make_client()
    r = client.post("/api/cluster/finetune", json={"sweep_lr": [1e-4, 3e-4], "sweep_steps": [100, 200]})
    assert r.status_code == 200, r.text
    assert len(r.json()["jobs"]) == 4
    assert {n for _, n, _ in FakeClient.submitted} == {
        "shree-finetune-lr0.0001-s100", "shree-finetune-lr0.0001-s200",
        "shree-finetune-lr0.0003-s100", "shree-finetune-lr0.0003-s200",
    }


@pytest.mark.parametrize("payload", [
    {"base": "../../secrets.pt"},
    {"base": "missing.pt"},
    {"name": "x; rm -rf /"},
    {"config": "huge"},
    {"env": "bad env"},
    {"steps": 0},
    {"sweep_lr": [1e-4, 2e-4, 3e-4], "sweep_steps": [1, 2, 3]},
])
def test_finetune_rejects_unsafe_or_invalid_input(make_client, payload):
    r = make_client().post("/api/cluster/finetune", json=payload)
    assert r.status_code == 422, payload
    assert FakeClient.submitted == []


def test_sdk_auth_errors_become_a_clear_message(make_client, monkeypatch):
    def boom(self):
        raise FakeError(401, "bad key")

    monkeypatch.setattr(FakeClient, "nodes", boom)
    r = make_client().get("/api/cluster/nodes")
    assert r.status_code == 401
    assert "API key" in r.json()["detail"]

