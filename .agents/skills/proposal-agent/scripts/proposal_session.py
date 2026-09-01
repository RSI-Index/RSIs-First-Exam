from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn


_SCHEMA = 1
_SUPPORTED_PLATFORMS = frozenset({"codex", "claude-code"})


class ProposalSessionError(RuntimeError):
    """A local problem the contributor can resolve before continuing."""


@dataclass(frozen=True)
class DiscussionRef:
    node_id: str
    number: int
    url: str


@dataclass(frozen=True)
class SessionBinding:
    platform: str
    session_id: str
    transcript_path: Path
    checkout_root: Path
    discussion: DiscussionRef | None = None


def _fail(message: str) -> NoReturn:
    raise ProposalSessionError(message)


def _state_dir(checkout_root: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout_root), "rev-parse", "--absolute-git-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(f"checkout is not a Git repository: {checkout_root}")
    return Path(completed.stdout.strip()) / "rsi-proposal-session"


def _checkout_root(checkout_root: Path) -> Path:
    try:
        root = checkout_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail(f"checkout does not exist: {checkout_root}")
    if not root.is_dir():
        _fail(f"checkout is not a directory: {root}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(f"checkout is not a Git repository: {root}")
    actual_root = Path(completed.stdout.strip()).resolve()
    if actual_root != root:
        _fail(f"checkout must be the Git worktree root: {root}")
    return root


def _json_object_with_unique_keys(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_json_object_with_unique_keys
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _fail(f"{label} is malformed: {error}")
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        _fail(f"{label} must be a JSON object")
    return parsed


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _load_activation(state_dir: Path, checkout_root: Path) -> bool:
    activation_path = state_dir / "activation.json"
    if not activation_path.is_file():
        return False
    activation = _read_json(activation_path, "activation state")
    if set(activation) != {"schema", "checkout_root"}:
        _fail("activation state has unexpected fields")
    if activation["schema"] != _SCHEMA or not isinstance(activation["checkout_root"], str):
        _fail("activation state has invalid values")
    try:
        activated_root = Path(activation["checkout_root"]).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail("activation state has an invalid checkout root")
    if activated_root != checkout_root:
        _fail("activation state belongs to another checkout")
    return True


def _valid_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"hook payload has invalid {field}")
    return value


def _resolve_hook_cwd(value: object, checkout_root: Path) -> Path:
    cwd = _valid_string(value, "cwd")
    try:
        resolved = Path(cwd).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail("hook payload cwd does not exist")
    if not resolved.is_dir():
        _fail("hook payload cwd is not a directory")
    try:
        resolved.relative_to(checkout_root)
    except ValueError:
        _fail("hook payload cwd is outside the activated checkout")
    return resolved


def _resolve_transcript(value: object) -> Path:
    transcript = _valid_string(value, "transcript_path")
    try:
        resolved = Path(transcript).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail("hook payload transcript_path does not exist")
    if not resolved.is_file():
        _fail("hook payload transcript_path is not a file")
    return resolved


def _read_discussion(value: object) -> DiscussionRef | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"node_id", "number", "url"}:
        _fail("binding discussion is malformed")
    node_id = value["node_id"]
    number = value["number"]
    url = value["url"]
    if (
        not isinstance(node_id, str)
        or not node_id
        or type(number) is not int
        or number < 0
        or not isinstance(url, str)
        or not url
    ):
        _fail("binding discussion has invalid values")
    return DiscussionRef(node_id=node_id, number=number, url=url)


def _binding_from_json(payload: dict[str, object], checkout_root: Path) -> SessionBinding:
    required = {
        "schema",
        "platform",
        "session_id",
        "transcript_path",
        "checkout_root",
        "discussion",
    }
    if set(payload) != required:
        _fail("binding state has unexpected fields")
    if payload["schema"] != _SCHEMA:
        _fail("binding state has an unsupported schema")
    platform = payload["platform"]
    if not isinstance(platform, str) or platform not in _SUPPORTED_PLATFORMS:
        _fail("binding state has an unsupported platform")
    session_id = payload["session_id"]
    if not isinstance(session_id, str) or not session_id.strip():
        _fail("binding state has an invalid session ID")
    recorded_root = payload["checkout_root"]
    if not isinstance(recorded_root, str):
        _fail("binding state has an invalid checkout root")
    try:
        binding_root = Path(recorded_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail("binding state has an invalid checkout root")
    if binding_root != checkout_root:
        _fail("binding state belongs to another checkout")
    transcript_value = payload["transcript_path"]
    if not isinstance(transcript_value, str):
        _fail("binding state has an invalid transcript path")
    try:
        transcript_path = Path(transcript_value).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail("bound transcript no longer exists")
    if not transcript_path.is_file() or str(transcript_path) != transcript_value:
        _fail("binding state has an invalid transcript path")
    return SessionBinding(
        platform=platform,
        session_id=session_id,
        transcript_path=transcript_path,
        checkout_root=checkout_root,
        discussion=_read_discussion(payload["discussion"]),
    )


def _binding_payload(binding: SessionBinding) -> dict[str, object]:
    discussion: dict[str, object] | None = None
    if binding.discussion is not None:
        discussion = {
            "node_id": binding.discussion.node_id,
            "number": binding.discussion.number,
            "url": binding.discussion.url,
        }
    return {
        "schema": _SCHEMA,
        "platform": binding.platform,
        "session_id": binding.session_id,
        "transcript_path": str(binding.transcript_path),
        "checkout_root": str(binding.checkout_root),
        "discussion": discussion,
    }


def _load_existing_binding(state_dir: Path, checkout_root: Path) -> SessionBinding | None:
    binding_path = state_dir / "binding.json"
    if not binding_path.is_file():
        return None
    return _binding_from_json(_read_json(binding_path, "binding state"), checkout_root)


@contextmanager
def _binding_lock(state_dir: Path):
    descriptor = os.open(state_dir / "binding.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        _fail(f"cannot lock proposal session state: {error}")


def activate(checkout_root: Path) -> Path:
    root = _checkout_root(checkout_root)
    state_dir = _state_dir(root)
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_json(state_dir / "activation.json", {"schema": _SCHEMA, "checkout_root": str(root)})
    return state_dir


def capture_hook(
    checkout_root: Path,
    platform: str,
    payload: Mapping[str, object],
) -> bool:
    if platform not in _SUPPORTED_PLATFORMS:
        _fail(f"unsupported platform: {platform}")
    root = _checkout_root(checkout_root)
    state_dir = _state_dir(root)
    if not _load_activation(state_dir, root):
        return False
    if not isinstance(payload, Mapping):
        _fail("hook payload must be a JSON object")
    if payload.get("hook_event_name") != "Stop":
        return False
    session_id = _valid_string(payload.get("session_id"), "session_id")

    with _binding_lock(state_dir):
        existing = _load_existing_binding(state_dir, root)
        if existing is not None and (existing.platform, existing.session_id) != (
            platform,
            session_id,
        ):
            return False

        _resolve_hook_cwd(payload.get("cwd"), root)
        transcript_path = _resolve_transcript(payload.get("transcript_path"))
        binding = SessionBinding(
            platform=platform,
            session_id=session_id,
            transcript_path=transcript_path,
            checkout_root=root,
            discussion=None if existing is None else existing.discussion,
        )
        _atomic_json(state_dir / "binding.json", _binding_payload(binding))
        return True


def load_binding(checkout_root: Path) -> SessionBinding:
    root = _checkout_root(checkout_root)
    state_dir = _state_dir(root)
    if not _load_activation(state_dir, root):
        _fail("proposal session capture is not activated")
    binding = _load_existing_binding(state_dir, root)
    if binding is None:
        _fail("proposal session binding is not available")
    return binding


def _hook_cli(arguments: argparse.Namespace) -> int:
    checkout = Path(arguments.checkout)
    try:
        root = _checkout_root(checkout)
        state_dir = _state_dir(root)
        if not _load_activation(state_dir, root):
            return 0
    except ProposalSessionError:
        return 0
    try:
        payload = json.loads(sys.stdin.read(), object_pairs_hook=_json_object_with_unique_keys)
        if not isinstance(payload, dict):
            _fail("hook payload must be a JSON object")
        capture_hook(root, arguments.platform, payload)
    except (ProposalSessionError, ValueError, json.JSONDecodeError) as error:
        print(f"proposal-session hook: {error}", file=sys.stderr)
    return 0


def _status_cli(arguments: argparse.Namespace) -> int:
    try:
        root = _checkout_root(Path(arguments.checkout))
        state_dir = _state_dir(root)
        if not _load_activation(state_dir, root):
            print("inactive")
            return 0
        binding = _load_existing_binding(state_dir, root)
        if binding is None:
            print("unbound")
            return 0
        print(f"bound {binding.platform} {binding.session_id}")
        return 0
    except ProposalSessionError as error:
        print(f"proposal-session: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage local proposal session capture")
    commands = parser.add_subparsers(dest="command", required=True)

    activate_parser = commands.add_parser("activate")
    activate_parser.add_argument("--checkout", required=True)

    hook_parser = commands.add_parser("hook")
    hook_parser.add_argument("--platform", required=True, choices=sorted(_SUPPORTED_PLATFORMS))
    hook_parser.add_argument("--checkout", required=True)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("--checkout", required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "activate":
        try:
            print(activate(Path(arguments.checkout)))
        except ProposalSessionError as error:
            print(f"proposal-session: {error}", file=sys.stderr)
            return 1
        return 0
    if arguments.command == "hook":
        return _hook_cli(arguments)
    return _status_cli(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
