"""Private, deployment-specific credentials for the legacy host-side Judge API.

Credential resolution is explicit and lazy: importing this module never reads
or creates a credential. The generated default is shared by local processes
running as the same user, independently of their current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import secrets
import stat

_MAX_SECRET_LENGTH = 4096


def _validate(value: str) -> str:
    if not 32 <= len(value) <= _MAX_SECRET_LENGTH or any(
        not 33 <= ord(character) <= 126 for character in value
    ):
        raise ValueError(
            "legacy Judge admin secret must contain 32 to 4096 printable "
            "ASCII characters without whitespace"
        )
    return value


def _private_parent(path: Path, *, create: bool) -> int:
    """Open the parent through non-symlink components, returning a pinned fd."""
    if not path.is_absolute() or path.name in {"", ".", ".."} or ".." in path.parts:
        raise ValueError("RSI_ADMIN_SECRET_FILE must be an absolute file path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        parts = path.parent.parts[1:]
        for index, component in enumerate(parts):
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create or index != len(parts) - 1:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass  # Another local process created the private directory.
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError(
                "legacy Judge credential directory must be user-owned and private"
            )
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read(parent: int, name: str) -> str:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    descriptor = os.open(name, flags, dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > _MAX_SECRET_LENGTH + 1
        ):
            raise ValueError(
                "legacy Judge credential must be a user-owned regular file with mode 0600"
            )
        chunks = []
        remaining = _MAX_SECRET_LENGTH + 2
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_uid, before.st_mode) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_uid,
            after.st_mode,
        ) or len(content) != after.st_size:
            raise ValueError("legacy Judge credential changed while reading")
        try:
            return _validate(content.removesuffix(b"\n").decode("ascii"))
        except UnicodeDecodeError:
            raise ValueError(
                "legacy Judge credential must contain ASCII text"
            ) from None
    finally:
        os.close(descriptor)


def _create_default(parent: int, name: str) -> None:
    """Publish complete bytes atomically without replacing a concurrent winner."""
    temporary = f".admin-secret-{secrets.token_hex(16)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        content = (secrets.token_urlsafe(48) + "\n").encode("ascii")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(descriptor)
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            pass
        os.fsync(parent)
    finally:
        os.close(descriptor)
        os.unlink(temporary, dir_fd=parent)


def get_admin_secret() -> str:
    """Read configured credentials or create the private local-user default.

    Explicit files must already exist. Missing, blank, conflicting or insecure
    configuration raises without exposing the supplied value or falling back.
    """
    inline = os.environ.get("RSI_ADMIN_SECRET")
    filename = os.environ.get("RSI_ADMIN_SECRET_FILE")
    if inline is not None and filename is not None:
        raise ValueError(
            "configure only one of RSI_ADMIN_SECRET and RSI_ADMIN_SECRET_FILE"
        )
    if inline is not None:
        return _validate(inline)
    if filename is not None and not filename.strip():
        raise ValueError("RSI_ADMIN_SECRET_FILE must not be blank")
    path = (
        Path(filename).expanduser()
        if filename is not None
        else Path.home() / ".rsi-loop" / "admin-secret"
    )
    parent = None
    try:
        parent = _private_parent(path, create=filename is None)
        try:
            return _read(parent, path.name)
        except FileNotFoundError:
            if filename is not None:
                raise
            _create_default(parent, path.name)
            return _read(parent, path.name)
    except OSError:
        raise ValueError(
            "cannot securely access legacy Judge credential; check the configured "
            "file, directory permissions, ownership and symlinks"
        ) from None
    finally:
        if parent is not None:
            os.close(parent)
