from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".agents/skills/proposal-agent/scripts/proposal_session.py"
SPEC = importlib.util.spec_from_file_location("proposal_session", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def init_git_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    return checkout


def write_transcript(path: Path, content: bytes = b'{"type":"message"}\n') -> Path:
    path.write_bytes(content)
    return path


def hook(session_id: str, transcript: Path, checkout: Path) -> dict[str, object]:
    return {
        "session_id": session_id,
        "transcript_path": str(transcript),
        "cwd": str(checkout),
        "hook_event_name": "Stop",
    }


def invoke(*args: str, input: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=input,
        capture_output=True,
        text=True,
        check=False,
    )


def test_activate_then_codex_stop_binds_exact_session(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "codex.jsonl")

    state_dir = module.activate(checkout)
    assert state_dir.parent == Path(
        subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--absolute-git-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert module.capture_hook(checkout, "codex", hook("thr_exact", transcript, checkout))

    binding = module.load_binding(checkout)
    assert binding.session_id == "thr_exact"
    assert binding.transcript_path == transcript.resolve()
    assert binding.checkout_root == checkout.resolve()
    assert not (checkout / ".rsi-proposal-session").exists()


def test_other_session_cannot_replace_binding(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    first = write_transcript(tmp_path / "first.jsonl")
    second = write_transcript(tmp_path / "second.jsonl")
    module.activate(checkout)

    assert module.capture_hook(checkout, "claude-code", hook("one", first, checkout))
    assert not module.capture_hook(checkout, "claude-code", hook("two", second, checkout))

    binding = module.load_binding(checkout)
    assert binding.session_id == "one"
    assert binding.transcript_path == first.resolve()


def test_exact_session_refreshes_its_transcript_path(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    first = write_transcript(tmp_path / "first.jsonl")
    refreshed = write_transcript(tmp_path / "refreshed.jsonl")
    module.activate(checkout)
    assert module.capture_hook(checkout, "codex", hook("thr_same", first, checkout))

    assert module.capture_hook(checkout, "codex", hook("thr_same", refreshed, checkout))
    assert module.load_binding(checkout).transcript_path == refreshed.resolve()


def test_capture_rejects_bad_platform_cwd_and_transcript(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "transcript.jsonl")
    outside = tmp_path / "outside"
    outside.mkdir()
    module.activate(checkout)

    with pytest.raises(module.ProposalSessionError, match="unsupported platform"):
        module.capture_hook(checkout, "other", hook("one", transcript, checkout))
    with pytest.raises(module.ProposalSessionError, match="outside"):
        module.capture_hook(checkout, "codex", hook("one", transcript, outside))
    with pytest.raises(module.ProposalSessionError, match="transcript"):
        module.capture_hook(
            checkout,
            "codex",
            hook("one", tmp_path / "missing.jsonl", checkout),
        )


def test_activate_rejects_a_non_git_directory(tmp_path: Path) -> None:
    with pytest.raises(module.ProposalSessionError, match="Git"):
        module.activate(tmp_path)


def test_hook_cli_ignores_missing_activation_and_other_session(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "transcript.jsonl")
    payload = json.dumps(hook("one", transcript, checkout))

    no_activation = invoke(
        "hook", "--platform", "codex", "--checkout", str(checkout), input=payload
    )
    assert no_activation.returncode == 0
    assert no_activation.stdout == no_activation.stderr == ""

    module.activate(checkout)
    assert module.capture_hook(checkout, "codex", hook("one", transcript, checkout))
    other_session = invoke(
        "hook",
        "--platform",
        "codex",
        "--checkout",
        str(checkout),
        input=json.dumps(hook("two", transcript, checkout)),
    )
    assert other_session.returncode == 0
    assert other_session.stdout == other_session.stderr == ""
    assert module.load_binding(checkout).session_id == "one"


def test_malformed_active_hook_is_diagnostic_and_status_stays_unbound(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    module.activate(checkout)

    malformed = invoke(
        "hook", "--platform", "claude-code", "--checkout", str(checkout), input="{not-json"
    )
    assert malformed.returncode == 0
    assert malformed.stdout == ""
    assert "proposal-session hook:" in malformed.stderr

    status = invoke("status", "--checkout", str(checkout))
    assert status.returncode == 0
    assert status.stdout.strip() == "unbound"
    with pytest.raises(module.ProposalSessionError, match="binding"):
        module.load_binding(checkout)
