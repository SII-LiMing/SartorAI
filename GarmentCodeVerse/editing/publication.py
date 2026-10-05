"""Safe target identifiers and no-overwrite directory publication.

The semantic-edit store and simulation workspace share these two filesystem
boundaries.  Candidate construction, serialization, and pipeline state live in
their owning modules rather than in a legacy pilot-specific dataset writer.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile


_TARGET_ID_PATTERN = re.compile(r"rand_[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class UnsafeTargetIdError(ValueError):
    """A target ID is not a portable, safe basename."""


class OutputExistsError(FileExistsError):
    """Publication would overwrite an existing path."""


class DatasetWriteError(RuntimeError):
    """A safe atomic directory publication cannot be completed."""


def require_disjoint_roots(**roots: Path | str) -> dict[str, Path]:
    """Resolve roots and reject equal or nested active datasets."""

    resolved = {
        name: Path(value).expanduser().resolve()
        for name, value in roots.items()
    }
    items = tuple(resolved.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"{left_name} and {right_name} must be disjoint"
                )
    return resolved


def validate_target_id(target_id: str) -> str:
    """Validate and return a portable simulator-compatible target ID."""

    if not isinstance(target_id, str):
        raise UnsafeTargetIdError("target_id must be a string")
    if _TARGET_ID_PATTERN.fullmatch(target_id) is None:
        raise UnsafeTargetIdError(
            "target_id must start with 'rand_' and contain only ASCII "
            "letters, digits, underscores, or hyphens"
        )
    if (
        Path(target_id).name != target_id
        or "/" in target_id
        or "\\" in target_id
    ):
        raise UnsafeTargetIdError("target_id must be a safe basename")
    if "_specification" in target_id:
        raise UnsafeTargetIdError(
            "target_id must not contain the reserved '_specification' token"
        )
    return target_id


def preflight_directory_publish(staging: Path, final: Path) -> None:
    """Validate a no-overwrite directory move without changing the filesystem."""

    if staging.is_symlink() or not staging.is_dir():
        raise DatasetWriteError(
            f"staging directory does not exist or is unsafe: {staging}"
        )
    if final.parent.is_symlink() or not final.parent.is_dir():
        raise DatasetWriteError(
            f"publication parent is not a safe directory: {final.parent}"
        )
    if os.path.lexists(final):
        raise OutputExistsError(f"refusing to overwrite existing path: {final}")
    if staging.stat().st_dev != final.parent.stat().st_dev:
        raise DatasetWriteError(
            "staging and final paths are on different filesystems: "
            f"{staging} vs {final}"
        )


def publish_directory(
    staging: Path,
    final: Path,
) -> None:
    """Rename ``staging`` to a previously unused ``final`` path."""

    preflight_directory_publish(staging, final)
    try:
        os.rename(staging, final)
    except (FileExistsError, IsADirectoryError, NotADirectoryError) as exc:
        raise OutputExistsError(
            f"refusing to overwrite existing path: {final}"
        ) from exc


def write_new_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            stream.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def atomic_replace_bytes(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes:
        raise TypeError("payload must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    temporary = Path(raw_temporary)
    try:
        try:
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
