"""One single-use GarmentCode simulation workspace for semantic edits.

Only the candidates awaiting physical simulation are copied into the hidden
workspace.  GarmentCode writes its native products there.  The semantic-edit
store later moves accepted products into the final flat batch and rejected
products into ``.state``.  No per-round dataset directories are created.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from editing.canonical_io import (
    canonical_hash,
    canonical_json_bytes,
    dump_yaml_document,
    load_json_bytes,
    load_yaml_bytes,
    sha256_bytes,
)
from editing.publication import (
    OutputExistsError,
    publish_directory,
    validate_target_id,
    write_new_bytes,
)


WORKSPACE_SCHEMA_VERSION = "gcd-semantic-edit-simulation-workspace-v1"
FILTER_RESULT_SCHEMA_VERSION = "gcd-semantic-edit-filter-result-v1"
_BATCH_ID_RE = re.compile(r"batch_[0-9]+\Z")
BODY_SAMPLES_ROOT = Path(
    "/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/"
    "DataSet/5000_body_shapes_and_measures"
)


class SimulationWorkspaceError(RuntimeError):
    """The simulation workspace is missing or inconsistent."""


@dataclass(frozen=True)
class SimulationWorkspace:
    root: Path
    simulation_id: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class FilterProjection:
    simulation_id: str
    results: tuple[Mapping[str, Any], ...]


def _strict_file(path: Path, description: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SimulationWorkspaceError(
            f"{description} is missing or unsafe: {path}"
        )
    payload = path.read_bytes()
    if not payload:
        raise SimulationWorkspaceError(f"{description} is empty: {path}")
    return payload


def _candidate_ids(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise SimulationWorkspaceError(
            f"{field_name} must be a nonempty list"
        )
    result: list[str] = []
    for item in value:
        try:
            result.append(validate_target_id(item))
        except Exception as exc:
            raise SimulationWorkspaceError(
                f"{field_name} contains an invalid candidate ID"
            ) from exc
    if len(result) != len(set(result)):
        raise SimulationWorkspaceError(f"{field_name} contains duplicates")
    return tuple(result)


def _panel_count(path: Path) -> int:
    specification = load_json_bytes(_strict_file(path, "specification"))
    pattern = specification.get("pattern")
    panels = pattern.get("panels") if isinstance(pattern, Mapping) else None
    if not isinstance(panels, Mapping) or not panels:
        raise SimulationWorkspaceError(
            f"specification has no nonempty pattern.panels: {path}"
        )
    return len(panels)


def _manifest_identity(
    *,
    batch_id: str,
    global_seed: str,
    candidate_ids: tuple[str, ...],
) -> dict[str, Any]:
    identity = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "batch_id": batch_id,
        "global_seed": global_seed,
        "candidate_ids": list(candidate_ids),
    }
    return {
        **identity,
        "simulation_id": canonical_hash(identity),
    }


def load_simulation_workspace(
    output_root: str | Path,
) -> SimulationWorkspace:
    root = Path(output_root).expanduser().resolve()
    workspace = root / ".state" / "simulation_workspace"
    if workspace.is_symlink() or not workspace.is_dir():
        raise SimulationWorkspaceError(
            f"simulation workspace is missing or unsafe: {workspace}"
        )
    payload = _strict_file(
        workspace / "simulation_manifest.json",
        "simulation workspace manifest",
    )
    manifest = load_json_bytes(payload)
    candidate_ids = _candidate_ids(
        manifest["candidate_ids"],
        "simulation workspace candidate_ids",
    )
    data_root = workspace / "random_body" / "data"
    if data_root.is_symlink() or not data_root.is_dir():
        raise SimulationWorkspaceError(
            "simulation workspace data directory is missing"
        )
    return SimulationWorkspace(
        root=workspace,
        simulation_id=manifest["simulation_id"],
        candidate_ids=candidate_ids,
    )


def remove_simulation_workspace(output_root: str | Path) -> None:
    root = Path(output_root).expanduser().resolve()
    workspace = root / ".state" / "simulation_workspace"
    if not os.path.lexists(workspace):
        return
    load_simulation_workspace(root)
    shutil.rmtree(workspace)


def _require_new_workspace(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise OutputExistsError(f"simulation workspace already exists: {path}")


def prepare_simulation_workspace(
    output_root: str | Path,
    pending_pairs: Sequence[Mapping[str, Any]],
    *,
    global_seed: str | int,
    batch_id: str,
) -> SimulationWorkspace:
    """Publish one new target-only simulation workspace."""

    if _BATCH_ID_RE.fullmatch(batch_id) is None:
        raise ValueError("batch_id must match batch_<nonnegative integer>")
    if isinstance(global_seed, bool) or not isinstance(global_seed, (str, int)):
        raise TypeError("global_seed must be a string or integer")
    root = Path(output_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise SimulationWorkspaceError("output root is missing or unsafe")
    if not pending_pairs:
        raise ValueError("simulation workspace requires pending candidates")

    rows: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in pending_pairs:
        if not isinstance(row, Mapping):
            raise TypeError("pending pair rows must be mappings")
        if row.get("target_simulation_status") != "pending":
            raise ValueError("pending pair has a non-pending status")
        candidate_id = validate_target_id(row.get("candidate_id"))
        if row.get("target_id") != candidate_id or candidate_id in seen:
            raise ValueError("pending pair identity is invalid or duplicated")
        seen.add(candidate_id)
        relative = Path(batch_id) / "random_body" / "data" / candidate_id
        if row.get("target_path") != relative.as_posix():
            raise ValueError("pending pair target_path is invalid")
        canonical = root / relative
        if canonical.is_symlink() or not canonical.is_dir():
            raise SimulationWorkspaceError(
                f"pending candidate directory is missing: {candidate_id}"
            )
        rows.append((candidate_id, canonical))
    rows.sort(key=lambda item: item[0].encode("ascii"))
    ids = tuple(candidate_id for candidate_id, _path in rows)
    manifest = _manifest_identity(
        batch_id=batch_id,
        global_seed=str(global_seed),
        candidate_ids=ids,
    )

    workspace = root / ".state" / "simulation_workspace"
    _require_new_workspace(workspace)

    work_root = root / ".state" / "work"
    if work_root.is_symlink() or not work_root.is_dir():
        raise SimulationWorkspaceError("store work directory is unsafe")
    staging = Path(
        tempfile.mkdtemp(prefix=".simulation-workspace.", dir=work_root)
    )
    try:
        data_root = staging / "random_body" / "data"
        data_root.mkdir(parents=True)
        panel_counts: dict[str, int] = {}
        for candidate_id, canonical in rows:
            destination = data_root / candidate_id
            shutil.copytree(canonical, destination)
            panel_counts[candidate_id] = _panel_count(
                destination / f"{candidate_id}_specification.json"
            )
        seed = (
            f"{global_seed}::simulation_workspace::"
            f"{manifest['simulation_id'].split(':', 1)[-1][:16]}"
        )
        properties = {
            "design_file": "./assets/design_params/design_beautiful.yaml",
            "body_default": "mean_all",
            "body_samples": BODY_SAMPLES_ROOT.as_posix(),
            "name": f"{root.name}/simulation_workspace",
            "to_subfolders": True,
            "frozen": False,
            "size": len(rows),
            "target_count": len(rows),
            "generator": {
                "config": {"random_seed": seed},
                "stats": {"panel_count": panel_counts},
            },
        }
        write_new_bytes(
            staging / "dataset_properties.yaml",
            dump_yaml_document(properties),
            mode=0o666,
        )
        write_new_bytes(
            staging / "simulation_manifest.json",
            canonical_json_bytes(manifest),
            mode=0o666,
        )
        publish_directory(staging, workspace)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_simulation_workspace(root)


def load_filter_projection(
    workspace: SimulationWorkspace,
    filter_report_path: str | Path,
) -> FilterProjection:
    """Read only the original GarmentCode filter decisions for this workspace."""

    report_path = Path(filter_report_path).expanduser().resolve()
    expected_report = (
        workspace.root / "random_body" / "filter_fail_report_full.yaml"
    )
    if report_path != expected_report:
        raise SimulationWorkspaceError(
            "filter report path does not match simulation workspace"
        )
    payload = _strict_file(report_path, "GarmentCode filter report")
    report = load_yaml_bytes(payload)
    passed = frozenset(report.get("passed_cases", ()))
    failed = frozenset(report.get("final_failed_cases", ()))
    outcomes = {candidate_id: "pass" for candidate_id in passed}
    outcomes.update({candidate_id: "fail" for candidate_id in failed})
    raw_reasons = report.get("final_fail_reasons", {})
    report_hash = sha256_bytes(payload)
    results: list[Mapping[str, Any]] = []
    for candidate_id in workspace.candidate_ids:
        try:
            status = outcomes[candidate_id]
        except KeyError as exc:
            raise SimulationWorkspaceError(
                f"filter result is missing for {candidate_id}"
            ) from exc
        reasons = raw_reasons.get(candidate_id, ()) if status == "fail" else ()
        results.append(
            {
                "schema_version": FILTER_RESULT_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "target_id": candidate_id,
                "simulation_id": workspace.simulation_id,
                "filter_report_sha256": report_hash,
                "target_simulation_status": status,
                "final_acceptance_status": status,
                "failure_reasons": tuple(reasons),
            }
        )
    return FilterProjection(
        simulation_id=workspace.simulation_id,
        results=tuple(results),
    )
