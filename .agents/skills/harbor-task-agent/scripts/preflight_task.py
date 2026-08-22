#!/usr/bin/env python3
"""Plan or explicitly execute a final authoring-time Environment preflight.

The default mode is read-only. Execution builds or obtains the task image and
starts one untouched, no-egress, no-GPU container on a private internal bridge
for Agent-led inspection. It never runs the task evaluator, training, Solution,
or reward path.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import tomllib


GIB = 1024**3
TEXT_NAMES = {"Dockerfile"}
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".ini",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class PreflightError(RuntimeError):
    """A safe planning or setup failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan an Environment preflight, or execute it only after explicit "
            "contributor authorization."
        )
    )
    parser.add_argument("task_dir", type=Path)
    parser.add_argument(
        "--required-free-gb",
        type=float,
        required=True,
        help="conservative Docker-filesystem headroom approved for this build, in GiB",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the stateful build/pull and create the inspection container",
    )
    parser.add_argument(
        "--acknowledge-authorized",
        action="store_true",
        help="assert that the contributor explicitly authorized the printed plan",
    )
    parser.add_argument("--image-tag", help="non-existing image tag to create")
    parser.add_argument("--container-name", help="non-existing container name to create")
    parser.add_argument(
        "--network-name", help="non-existing internal bridge name to create"
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        timeout=timeout,
    )


def load_task(task_dir: Path) -> dict:
    try:
        with (task_dir / "task.toml").open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PreflightError(f"cannot read task.toml: {error}") from error
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise PreflightError("task.toml has no [environment] table")
    workdir = environment.get("workdir")
    if not isinstance(workdir, str) or not workdir.startswith("/"):
        raise PreflightError("[environment].workdir must be an absolute path")
    return config


def text_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES:
            result.append(path)
    return result


def detected_build_network(environment_dir: Path) -> tuple[list[str], list[str]]:
    hosts: set[str] = set()
    notes: set[str] = set()
    combined: list[str] = []
    for path in text_files(environment_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        combined.append(text)
        for match in re.finditer(r"(?:https?|git)://[^\s'\"\\]+", text):
            host = urlsplit(match.group(0)).hostname
            if host:
                hosts.add(host.lower())
        for host in re.findall(r"\bgit@([A-Za-z0-9.-]+):", text):
            hosts.add(host.lower())

    body = "\n".join(combined)
    for image in re.findall(r"^\s*FROM\s+(?:--\S+\s+)?([^\s]+)", body, re.MULTILINE | re.IGNORECASE):
        image = image.split("@", 1)[0]
        first = image.split("/", 1)[0]
        hosts.add(first.lower() if "." in first or ":" in first else "registry-1.docker.io")
    if re.search(r"\b(?:pip|uv\s+pip)\s+install\b", body):
        hosts.update(("pypi.org", "files.pythonhosted.org"))
    if re.search(r"\b(?:snapshot_download|hf_hub_download|HfApi)\b", body):
        hosts.add("huggingface.co")
    if re.search(r"\bapt(?:-get)?\s+(?:update|install)\b", body):
        notes.add("base-image configured APT repositories (resolved only during build)")
    if re.search(r"\bgit\s+(?:clone|fetch)\b", body):
        notes.add("Git hosts and any redirects/submodule/LFS endpoints used by the pinned source")
    return sorted(hosts), sorted(notes)


def compose_scalar(compose: Path, key: str) -> str | None:
    if not compose.is_file():
        return None
    try:
        text = compose.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(rf"^\s{{4}}{re.escape(key)}:\s*([^#\n]+)", text, re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip().strip("'\"")


def environment_fingerprint(environment_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in environment_dir.rglob("*") if path.is_file()):
        relative = path.relative_to(environment_dir).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def docker_root_and_free() -> tuple[Path, int]:
    try:
        completed = run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"], capture=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise PreflightError("Docker is unavailable or the daemon is not reachable") from error
    root = Path(completed.stdout.strip())
    if not root.is_absolute():
        raise PreflightError("Docker returned an invalid data-root path")
    try:
        free = shutil.disk_usage(root).free
    except OSError as error:
        raise PreflightError(f"cannot inspect Docker filesystem capacity: {error}") from error
    return root, free


def image_exists(image: str) -> bool:
    return run(
        ["docker", "image", "inspect", image], capture=True, check=False
    ).returncode == 0


def container_exists(name: str) -> bool:
    return run(
        ["docker", "container", "inspect", name], capture=True, check=False
    ).returncode == 0


def network_exists(name: str) -> bool:
    return run(
        ["docker", "network", "inspect", name], capture=True, check=False
    ).returncode == 0


def validate_task(task_dir: Path) -> None:
    validator = Path(__file__).with_name("validate_task.py")
    completed = run(
        [sys.executable, str(validator), str(task_dir)], capture=True, check=False
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise PreflightError("static task validation must pass before preflight")


def print_plan(
    *,
    task_dir: Path,
    source: str,
    image: str,
    container: str,
    network: str,
    docker_root: Path,
    free_bytes: int,
    required_bytes: int,
    hosts: list[str],
    notes: list[str],
    workdir: str,
    build_timeout: float,
) -> None:
    print("Environment preflight plan")
    print(f"Task: {task_dir}")
    print(f"Environment source: {source}")
    print(f"Docker data root: {docker_root}")
    print(f"Free space: {free_bytes / GIB:.1f} GiB")
    print(f"Required headroom: {required_bytes / GIB:.1f} GiB (Agent estimate)")
    print(f"Disk status: {'sufficient' if free_bytes >= required_bytes else 'INSUFFICIENT'}")
    print(f"Build/pull network hosts observed: {', '.join(hosts) if hosts else 'none detected'}")
    for note in notes:
        print(f"Network review note: {note}")
    print(f"Build timeout: {build_timeout:.0f} seconds")
    print("Expected duration: task-dependent; the declared build timeout is the upper bound")
    print(f"Will create/use image: {image}")
    print(f"Will create container: {container}")
    print(f"Will create internal bridge: {network}")
    print(f"Inspection WORKDIR: {workdir}")
    print(
        "Inspection container: private internal bridge (no external route), "
        "GPUs=none, task tests mounted read-only"
    )
    print("Will not run tests/test.sh, Solution, training, evaluation, submission, or reward writing")
    print(
        "The created image, container, and internal bridge will be retained "
        "for inspection"
    )
    print("Removing any of them is a separate state-changing action requiring authorization")
    print("Execution requires a later explicit contributor authorization")


def inspect_image(image: str) -> None:
    completed = run(
        ["docker", "image", "inspect", image, "--format", "{{json .Config.Volumes}}"],
        capture=True,
    )
    if completed.stdout.strip() not in ("null", "{}"):
        raise PreflightError("image declares Docker volumes and is incompatible with snapshot ownership")


def create_inspection_network(name: str) -> None:
    run(["docker", "network", "create", "--driver", "bridge", "--internal", name])


def create_inspection_container(
    *,
    image: str,
    name: str,
    network: str,
    task_dir: Path,
    workdir: str,
    environment: dict,
    compose: Path,
) -> None:
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        network,
        "--entrypoint",
        "/bin/sh",
        "--mount",
        f"type=bind,src={task_dir / 'tests'},dst=/tests,readonly",
    ]
    shm_size = compose_scalar(compose, "shm_size")
    user = compose_scalar(compose, "user")
    if shm_size:
        command.extend(("--shm-size", shm_size))
    if user:
        command.extend(("--user", user))
    shared_env = environment.get("env", {})
    if isinstance(shared_env, dict):
        for key, value in sorted(shared_env.items()):
            if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                command.extend(("--env", f"{key}={value}"))
    command.extend(
        (
            image,
            "-c",
            "trap 'exit 0' TERM INT; while :; do sleep 3600; done",
        )
    )
    run(command)
    run(["docker", "start", name])
    completed = run(
        ["docker", "exec", name, "/bin/sh", "-c", 'test -d "$1" && test -r "$1"', "sh", workdir],
        check=False,
    )
    if completed.returncode != 0:
        raise PreflightError(f"configured WORKDIR is absent or unreadable: {workdir}")


def execute(
    *,
    task_dir: Path,
    config: dict,
    source: str,
    image: str,
    container: str,
    network: str,
) -> None:
    environment_dir = task_dir / "environment"
    dockerfile = environment_dir / "Dockerfile"
    environment = config["environment"]
    timeout = float(environment.get("build_timeout_sec", 600))

    if container_exists(container):
        raise PreflightError(f"refusing to overwrite existing container: {container}")
    if network_exists(network):
        raise PreflightError(f"refusing to reuse existing Docker network: {network}")
    if source == "Dockerfile":
        if image_exists(image):
            raise PreflightError(f"refusing to overwrite existing image tag: {image}")
        run(
            [
                "docker",
                "build",
                "--file",
                str(dockerfile),
                "--tag",
                image,
                str(environment_dir),
            ],
            timeout=timeout,
        )
    else:
        if not image_exists(image):
            run(["docker", "pull", image], timeout=timeout)

    inspect_image(image)
    create_inspection_network(network)
    create_inspection_container(
        image=image,
        name=container,
        network=network,
        task_dir=task_dir,
        workdir=environment["workdir"],
        environment=environment,
        compose=environment_dir / "docker-compose.yaml",
    )
    print("Environment preflight container is ready for Agent-led read-only inspection")
    print(f"Inspect: docker exec -it {container} /bin/bash")
    print(f"Root inspect when needed: docker exec -u 0 -it {container} /bin/bash")
    print(f"Internal bridge retained: {network}")
    print("Do not run /tests/test.sh, training, evaluation, or rsi-submit in this preflight")
    print(
        "Container, image, and internal bridge are intentionally retained; "
        "removal requires separate authorization"
    )


def main() -> int:
    args = parse_args()
    if args.required_free_gb <= 0:
        raise PreflightError("--required-free-gb must be positive")
    if args.acknowledge_authorized and not args.execute:
        raise PreflightError("--acknowledge-authorized is valid only with --execute")
    if args.execute and not args.acknowledge_authorized:
        raise PreflightError("--execute requires --acknowledge-authorized")

    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        raise PreflightError("task directory does not exist")
    validate_task(task_dir)
    config = load_task(task_dir)
    environment = config["environment"]
    environment_dir = task_dir / "environment"
    dockerfile = environment_dir / "Dockerfile"
    compose = environment_dir / "docker-compose.yaml"

    source = "Dockerfile"
    image = args.image_tag
    if dockerfile.is_file():
        if image is None:
            image = f"rsi-preflight-{task_dir.name}:{environment_fingerprint(environment_dir)}"
    else:
        source = "prebuilt image"
        configured = environment.get("docker_image")
        if not isinstance(configured, str) or not configured:
            configured = compose_scalar(compose, "image")
        if not isinstance(configured, str) or not configured:
            raise PreflightError("task has neither a Dockerfile nor a prebuilt image")
        if image is not None and image != configured:
            raise PreflightError("--image-tag cannot replace the task's approved prebuilt image")
        image = configured

    fingerprint = environment_fingerprint(environment_dir)
    container = args.container_name or f"rsi-preflight-{task_dir.name}-{fingerprint}"
    network = args.network_name or f"{container}-net"
    docker_root, free_bytes = docker_root_and_free()
    required_bytes = int(args.required_free_gb * GIB)
    hosts, notes = detected_build_network(environment_dir)
    if source == "prebuilt image":
        notes.append("approved image registry access is required when the image is not local")
    timeout = float(environment.get("build_timeout_sec", 600))
    print_plan(
        task_dir=task_dir,
        source=source,
        image=image,
        container=container,
        network=network,
        docker_root=docker_root,
        free_bytes=free_bytes,
        required_bytes=required_bytes,
        hosts=hosts,
        notes=notes,
        workdir=environment["workdir"],
        build_timeout=timeout,
    )

    if not args.execute:
        return 0
    if free_bytes < required_bytes:
        raise PreflightError("Docker filesystem does not meet the approved free-space requirement")
    execute(
        task_dir=task_dir,
        config=config,
        source=source,
        image=image,
        container=container,
        network=network,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(f"preflight error: {error}", file=sys.stderr)
        raise SystemExit(2)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"preflight execution failed: {type(error).__name__}", file=sys.stderr)
        print(
            "An image, container, or internal bridge may have been retained; "
            "inspect them before separately authorized cleanup.",
            file=sys.stderr,
        )
        raise SystemExit(2)
