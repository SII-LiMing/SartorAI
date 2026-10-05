"""Render validated GarmentCode specifications into transient 2D previews.

The edit runner publishes only immutable SVG and PNG bytes returned by this
module.  Rendering itself happens in a uniquely-created directory below the
dataset-owned ``.state`` tree, so neither source samples nor final target
directories are ever passed to :class:`pygarment.pattern.wrappers.VisPattern`.

The heavy visualization dependency is imported lazily.  Unit tests and other
dependency-light pipeline stages can therefore import this module without
requiring Cairo, matplotlib, or scipy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .canonical_io import canonical_json_bytes
from .publication import validate_target_id


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PatternRenderError(RuntimeError):
    """Raised when rendering does not produce valid 2D artifacts."""


class InvalidRenderWorkRoot(ValueError):
    """Raised when rendering is requested outside a dataset ``.state`` tree."""


@dataclass(frozen=True)
class RenderedPattern:
    """Immutable 2D visualization payloads ready for dataset publication."""

    svg_bytes: bytes
    png_bytes: bytes


def _validated_work_root(work_root: os.PathLike[str] | str) -> Path:
    if not isinstance(work_root, (str, os.PathLike)):
        raise TypeError("work_root must be a string or path-like value")
    try:
        resolved = Path(work_root).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise InvalidRenderWorkRoot(
            f"render work_root must already exist: {work_root}"
        ) from error
    if not resolved.is_dir():
        raise InvalidRenderWorkRoot(
            f"render work_root must be a directory: {resolved}"
        )
    if ".state" not in resolved.parts:
        raise InvalidRenderWorkRoot(
            f"render work_root must be inside a dataset .state tree: {resolved}"
        )
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise InvalidRenderWorkRoot(
            f"render work_root must be writable and searchable: {resolved}"
        )
    return resolved


def _read_rendered_bytes(svg_path: Path, png_path: Path) -> RenderedPattern:
    try:
        svg_payload = svg_path.read_bytes()
    except (FileNotFoundError, IsADirectoryError, OSError) as error:
        raise PatternRenderError(
            f"renderer did not produce a readable SVG: {svg_path.name}"
        ) from error
    if not svg_payload:
        raise PatternRenderError(f"renderer produced an empty SVG: {svg_path.name}")

    try:
        png_payload = png_path.read_bytes()
    except (FileNotFoundError, IsADirectoryError, OSError) as error:
        raise PatternRenderError(
            f"renderer did not produce a readable PNG: {png_path.name}"
        ) from error
    if not png_payload.startswith(PNG_SIGNATURE):
        raise PatternRenderError(
            f"renderer produced an invalid PNG signature: {png_path.name}"
        )

    # ``read_bytes`` already returns immutable ``bytes`` values.  Constructing
    # the frozen result here makes it explicit that no temporary path escapes.
    return RenderedPattern(svg_bytes=svg_payload, png_bytes=png_payload)


def render_pattern(
    specification: Mapping[str, Any],
    target_id: str,
    work_root: os.PathLike[str] | str,
) -> RenderedPattern:
    """Render one specification to SVG/PNG and return only immutable bytes.

    ``work_root`` must be an existing writable directory at or below a
    dataset-owned ``.state`` directory.  A unique child directory is removed
    after both successful and failed renders.  The renderer is deliberately
    given no source path and no final publication path.
    """

    target_id = validate_target_id(target_id)
    if not isinstance(specification, Mapping):
        raise TypeError("specification must be a mapping")
    specification_payload = canonical_json_bytes(specification)
    resolved_work_root = _validated_work_root(work_root)

    temporary = Path(
        tempfile.mkdtemp(prefix=f"{target_id}.render.", dir=resolved_work_root)
    )
    try:
        specification_path = temporary / f"{target_id}_specification.json"
        with specification_path.open("xb") as stream:
            stream.write(specification_payload)

        from pygarment.pattern.wrappers import VisPattern

        VisPattern(os.fspath(specification_path)).serialize(
            os.fspath(temporary),
            to_subfolder=False,
            with_3d=False,
            with_text=False,
            view_ids=False,
        )

        return _read_rendered_bytes(
            temporary / f"{target_id}_pattern.svg",
            temporary / f"{target_id}_pattern.png",
        )
    finally:
        # The exact unique directory is the only recursive cleanup target.
        # ``work_root`` and every sibling are left untouched.
        shutil.rmtree(temporary)
