"""Deployment-specific legacy Judge credentials, using only temporary files."""

import importlib
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rsi_loop.harness import constants


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch, tmp_path):
    monkeypatch.delenv("RSI_ADMIN_SECRET", raising=False)
    monkeypatch.delenv("RSI_ADMIN_SECRET_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


def test_environment_override_is_shared_without_creating_files(monkeypatch, tmp_path):
    value = "test-deployment-A-" + "a" * 40
    monkeypatch.setenv("RSI_ADMIN_SECRET", value)
    assert constants.get_admin_secret() == value
    assert constants.get_admin_secret() == value
    assert list(tmp_path.iterdir()) == []


def test_default_is_private_persisted_and_shared_across_concurrent_callers(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: constants.get_admin_secret(), range(16)))
    assert len(set(values)) == 1
    assert len(values[0]) >= 32
    path = tmp_path / ".rsi-loop" / "admin-secret"
    assert path.read_text().strip() == values[0]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    importlib.reload(constants)
    assert constants.get_admin_secret() == values[0]
    assert list(path.parent.iterdir()) == [path]


def test_import_does_not_create_credentials(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from rsi_loop.harness import constants, judge_server, run_agent",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    assert list(tmp_path.iterdir()) == []


def test_default_is_shared_by_separate_host_processes():
    command = [
        sys.executable,
        "-c",
        "from rsi_loop.harness.constants import get_admin_secret; "
        "print(get_admin_secret())",
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=20
                ),
                range(4),
            )
        )
    assert {result.stdout.strip() for result in results} == {
        constants.get_admin_secret()
    }


def test_explicit_file_reads_shared_credential(monkeypatch, tmp_path):
    directory = tmp_path / "secrets"
    directory.mkdir(mode=0o700)
    path = directory / "judge-key"
    path.write_text("file-deployment-" + "b" * 40 + "\n")
    path.chmod(0o600)
    monkeypatch.setenv("RSI_ADMIN_SECRET_FILE", str(path))
    assert constants.get_admin_secret() == "file-deployment-" + "b" * 40
    assert not (tmp_path / ".rsi-loop").exists()


@pytest.mark.parametrize(
    "value", ["", " ", "too-short", "x" * 31, "x" * 32 + "\n", "é" * 32]
)
def test_invalid_env_fails_closed_without_echoing_value(monkeypatch, value, tmp_path):
    monkeypatch.setenv("RSI_ADMIN_SECRET", value)
    with pytest.raises(ValueError) as caught:
        constants.get_admin_secret()
    if len(value) > 1:
        assert value not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_conflicting_sources_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("RSI_ADMIN_SECRET", "c" * 40)
    monkeypatch.setenv("RSI_ADMIN_SECRET_FILE", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="only one"):
        constants.get_admin_secret()


@pytest.mark.parametrize(
    "kind",
    [
        "file_permissions",
        "directory_permissions",
        "file_symlink",
        "directory_symlink",
        "fifo",
        "missing",
        "empty_path",
    ],
)
def test_insecure_explicit_file_is_rejected(monkeypatch, tmp_path, kind):
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / "key"
    path.write_text("d" * 40)
    path.chmod(0o600)
    if kind == "file_permissions":
        path.chmod(0o644)
    elif kind == "directory_permissions":
        directory.chmod(0o755)
    elif kind == "file_symlink":
        link = directory / "link"
        link.symlink_to(path)
        path = link
    elif kind == "directory_symlink":
        link = tmp_path / "linked"
        link.symlink_to(directory, target_is_directory=True)
        path = link / "key"
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    elif kind == "missing":
        path.unlink()
    monkeypatch.setenv(
        "RSI_ADMIN_SECRET_FILE", "" if kind == "empty_path" else str(path)
    )
    with pytest.raises(ValueError):
        constants.get_admin_secret()


def test_existing_default_directory_must_be_private(tmp_path):
    directory = tmp_path / ".rsi-loop"
    directory.mkdir(mode=0o755)
    with pytest.raises(ValueError):
        constants.get_admin_secret()
    assert list(directory.iterdir()) == []


def test_server_rejects_wrong_key_and_uses_startup_key_for_full_history(monkeypatch):
    from fastapi.testclient import TestClient

    from rsi_loop.harness import judge_server

    key = "server-deployment-" + "e" * 40
    monkeypatch.setenv("RSI_ADMIN_SECRET", key)

    class State:
        tasks = {}

        def __init__(self, config):
            pass

        def load_tasks(self):
            pass

        def register_session(self, task_id, run_id, **kwargs):
            return "dummy-session"

        def resolve_token(self, token):
            return {"run_id": "dummy-run", "task_id": "dummy-task"}

        def get_run_history(self, run_id, task_id):
            return {"entries": [{"round": "auto-1"}], "marker": "host-only"}

    monkeypatch.setattr(judge_server, "JudgeState", State)
    app = judge_server.create_app(SimpleNamespace())
    monkeypatch.setenv("RSI_ADMIN_SECRET", "changed-after-startup-" + "f" * 40)
    with TestClient(app) as client:
        body = {"task_id": "dummy-task", "run_id": "dummy-run"}
        assert client.post("/api/v1/register", json=body).status_code == 403
        assert (
            client.post(
                "/api/v1/register", json={**body, "admin_secret": "wrong"}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/register", json={**body, "admin_secret": "é" * 40}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/register", json={**body, "admin_secret": key}
            ).status_code
            == 200
        )
        rejected = client.post(
            "/api/v1/submit",
            data={"token": "dummy-session", "kind": "auto", "admin_secret": "wrong"},
            files={"archive": ("empty.tar", b"")},
        )
        assert rejected.status_code == 403
        history = client.get(
            "/api/v1/history",
            params={"token": "dummy-session"},
            headers={"X-RSI-Admin-Secret": key},
        )
        assert history.status_code == 200
        assert history.json()["marker"] == "host-only"
        assert key not in str(history.request.url)
        without_key = client.get("/api/v1/history", params={"token": "dummy-session"})
        assert "marker" not in without_key.json()
        assert without_key.json()["entries"] == []
