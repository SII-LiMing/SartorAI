"""Small write-once store for the semantic-edit pipeline.

The store deliberately separates three states:

* source snapshots are captured once under ``.state/source_snapshots``;
* five-file candidates temporarily occupy the simulator-compatible data path;
* a physical pass adds the two final files and moves its native GarmentCode
  product beside ``random_body/data``.

No method reads or hashes the original dataset after the source snapshot is
created.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from editing.canonical_io import (
    canonical_json_bytes,
    load_json_bytes,
    plain_copy,
    sha256_bytes,
)
from editing.publication import (
    OutputExistsError,
    atomic_replace_bytes,
    preflight_directory_publish,
    publish_directory,
    validate_target_id,
    write_new_bytes,
)
from editing.simulation_workspace import load_simulation_workspace

from .instructions import validate_instruction_metadata
from .prompt import APPROVED_PROMPT_BUILDER_SHA256

PLAN_FORMAT = "semantic-edit-plan"
RUN_FORMAT = "semantic-edit-run"
PAIR_FORMAT = "semantic-edit-pair"
MAX_PHYSICAL_SIMULATION_ATTEMPTS_PER_SOURCE = 3
_PHYSICAL_SIMULATION_STATUSES = frozenset(
    {
        "pending_simulation",
        "in_simulation",
        "simulation_failed",
        "prompt_failed",
        "accepted",
    }
)
_RUN_CONTRACT = {
    "format": RUN_FORMAT,
    "panel_geometry_policy": "preserve_source_unaffected_panels",
    "max_physical_attempts_per_source": (
        MAX_PHYSICAL_SIMULATION_ATTEMPTS_PER_SOURCE
    ),
    "prompt_builder_sha256": APPROVED_PROMPT_BUILDER_SHA256,
}
_BATCH_ID_RE = re.compile(r"batch_[0-9]+\Z")
_SPLIT_SAMPLE_RE = re.compile(r"rand_[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_PAIR_CANDIDATE_INSTRUCTION_KEYS = (
    "slot_id",
    "source_key",
    "source_sample_id",
    "operation_signature",
    "operation",
    "specific_instruction",
    "canonical_instruction",
    "instruction_provenance",
)


class SimpleEditStoreError(RuntimeError):
    """The store cannot complete the requested transition."""


class SemanticEditError(RuntimeError):
    """The simplified edit run cannot advance safely."""


class SlotPhase(str, Enum):
    ACCEPTED = "accepted"
    IN_SIMULATION = "in_simulation"
    PENDING_SIMULATION = "pending_simulation"
    PHYSICAL_LIMIT_REACHED = "physical_limit_reached"
    RETRYABLE_OR_EXHAUSTED = "retryable_or_exhausted"
    NOT_GENERATED = "not_generated"


@dataclass(frozen=True, slots=True)
class SplitSelection:
    path: Path
    name: str
    file_sha256: str
    source_paths: frozenset[str]

    def payload(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "name": self.name,
            "file_sha256": self.file_sha256,
            "listed_source_count": len(self.source_paths),
        }


@dataclass(frozen=True)
class AcceptedProjection:
    prompt_bytes: bytes
    pair_record: Mapping[str, Any]
    candidate_metadata: Mapping[str, Any]
    png_bytes: bytes
    svg_bytes: bytes


@dataclass(slots=True)
class ProjectionTransition:
    """A fully validated filesystem transition ready to commit once."""

    candidate_id: str
    workspace_root: Path
    outcome: str
    slot_id: str
    state: dict[str, Any]
    state_payload: bytes
    prompt_bytes: bytes | None = None
    pair_bytes: bytes | None = None
    png_bytes: bytes | None = None
    svg_bytes: bytes | None = None


def _regular_file(path: Path, field_name: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise SemanticEditError(f"{field_name} is missing or unsafe: {candidate}")
    return candidate.resolve()


def _slot_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    slots = manifest.get("slots")
    if not isinstance(slots, list) or not slots:
        raise SemanticEditError("run manifest has no slots")
    result: dict[str, Mapping[str, Any]] = {}
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise SemanticEditError("run manifest slot is not a mapping")
        slot_id = slot.get("slot_id")
        if not isinstance(slot_id, str) or slot_id in result:
            raise SemanticEditError("run manifest slot identity is invalid")
        result[slot_id] = slot
    return result


def _split_source_path(value: Any, input_root: Path, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    path = Path(value)
    if not path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"{field_name} must be a normalized absolute path")
    try:
        relative = path.relative_to(input_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} is outside input_root") from exc
    parts = relative.parts
    valid = (
        len(parts) == 3
        and _BATCH_ID_RE.fullmatch(parts[0]) is not None
        and parts[1] == "random_body"
        and _SPLIT_SAMPLE_RE.fullmatch(parts[2]) is not None
    ) or (
        len(parts) == 4
        and _BATCH_ID_RE.fullmatch(parts[0]) is not None
        and parts[1:3] == ("random_body", "data")
        and _SPLIT_SAMPLE_RE.fullmatch(parts[3]) is not None
    )
    if not valid:
        raise ValueError(f"{field_name} is not a supported sample directory")
    return value


def load_split_selection(
    *,
    split_path: Path | str | None,
    split_name: str | None,
    input_root: Path | str,
) -> SplitSelection | None:
    """Load one subset, or return ``None`` for full processing."""

    if split_path is None and split_name is None:
        return None
    if split_path is None or split_name is None:
        raise ValueError("split_path and split_name must be provided together")
    if (
        not isinstance(split_name, str)
        or not split_name
        or split_name != split_name.strip()
    ):
        raise ValueError("split_name must be non-empty trimmed text")

    root = Path(input_root).expanduser().resolve(strict=True)
    raw_path = Path(split_path).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError(f"split_path is missing or unsafe: {raw_path}")
    path = raw_path.resolve()
    payload_bytes = path.read_bytes()
    try:
        document = json.loads(payload_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("split_path is not valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping) or len(document) != 2:
        raise ValueError("split JSON must contain exactly two dataset mappings")
    if split_name not in document:
        raise ValueError(f"split_name is absent from split JSON: {split_name}")

    datasets: dict[str, frozenset[str]] = {}
    for name, values in document.items():
        if not isinstance(name, str) or not name:
            raise ValueError("split JSON keys must be non-empty strings")
        if not isinstance(values, list):
            raise ValueError(f"split dataset must be a list: {name}")
        normalized = tuple(
            _split_source_path(value, root, f"{name}[{index}]")
            for index, value in enumerate(values)
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"split dataset contains duplicate paths: {name}")
        datasets[name] = frozenset(normalized)
    first, second = datasets.values()
    if first & second:
        raise ValueError("split datasets must be disjoint")

    return SplitSelection(
        path=path,
        name=split_name,
        file_sha256=sha256_bytes(payload_bytes),
        source_paths=datasets[split_name],
    )


def selected_slot_ids(
    slots: Mapping[str, Mapping[str, Any]],
    *,
    input_root: Path | str,
    selection: SplitSelection | None,
) -> frozenset[str]:
    """Select manifest slots by their canonical source directory."""

    if selection is None:
        return frozenset(slots)
    root = Path(input_root).expanduser().resolve(strict=True)
    selected: set[str] = set()
    for slot_id, slot in slots.items():
        source_key = slot.get("source_key")
        if not isinstance(source_key, str) or not source_key:
            raise ValueError(f"manifest slot has an invalid source_key: {slot_id}")
        source_path = root.joinpath(*source_key.split("/")).as_posix()
        if source_path in selection.source_paths:
            selected.add(slot_id)
    return frozenset(selected)


def _assert_current_run_contract(manifest: Mapping[str, Any]) -> None:
    expected = {
        **_RUN_CONTRACT,
        "candidate_mode": "registry_planned",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        fields = ", ".join(sorted(mismatches, key=str.encode))
        raise SemanticEditError(
            f"run is incompatible with the current pipeline: {fields}"
        )


def _slot_attempts(
    state: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise SemanticEditError("slot state has no attempts list")
    if not all(isinstance(attempt, Mapping) for attempt in attempts):
        raise SemanticEditError("slot state attempt rows must be mappings")
    return attempts


def _physical_simulation_attempt_count(state: Mapping[str, Any]) -> int:
    attempts = _slot_attempts(state)
    return sum(
        attempt.get("status") in _PHYSICAL_SIMULATION_STATUSES
        for attempt in attempts
    )


def _classify_slot(state: Mapping[str, Any]) -> SlotPhase:
    attempts = _slot_attempts(state)
    statuses = {attempt.get("status") for attempt in attempts}
    accepted_attempts = [
        attempt for attempt in attempts if attempt.get("status") == "accepted"
    ]
    if state.get("accepted_target_id") is not None:
        if len(accepted_attempts) != 1:
            raise SemanticEditError(
                f"accepted slot has invalid attempts: {state.get('slot_id')}"
            )
        if "in_simulation" in statuses:
            raise SemanticEditError(
                "accepted slot has inconsistent in-simulation attempt: "
                f"{state.get('slot_id')}"
            )
        return SlotPhase.ACCEPTED
    if accepted_attempts:
        raise SemanticEditError(
            "slot has an accepted attempt without an accepted target: "
            f"{state.get('slot_id')}"
        )
    if "in_simulation" in statuses:
        return SlotPhase.IN_SIMULATION
    if "pending_simulation" in statuses:
        return SlotPhase.PENDING_SIMULATION
    if (
        _physical_simulation_attempt_count(state)
        >= MAX_PHYSICAL_SIMULATION_ATTEMPTS_PER_SOURCE
    ):
        return SlotPhase.PHYSICAL_LIMIT_REACHED
    if attempts:
        return SlotPhase.RETRYABLE_OR_EXHAUSTED
    return SlotPhase.NOT_GENERATED


def _accepted_rule_id(state: Mapping[str, Any]) -> str:
    """Return the accepted rule for status reporting after phase validation."""

    attempts = _slot_attempts(state)
    accepted_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("status") == "accepted"
    ]
    if len(accepted_attempts) != 1:
        raise SemanticEditError(
            f"accepted slot has invalid attempts: {state.get('slot_id')}"
        )
    rule_id = accepted_attempts[0].get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        raise SemanticEditError(
            f"accepted slot has invalid rule id: {state.get('slot_id')}"
        )
    return rule_id


def _validate_instruction_payload(
    document: Mapping[str, Any],
    field_name: str,
) -> None:
    """Validate the persisted instruction envelope at each store boundary."""

    try:
        validate_instruction_metadata(document)
    except (TypeError, ValueError) as exc:
        raise SimpleEditStoreError(
            f"{field_name} instruction payload is invalid"
        ) from exc


def _validate_pair_matches_candidate(
    pair: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_id: str,
) -> None:
    mismatched = [
        key
        for key in _PAIR_CANDIDATE_INSTRUCTION_KEYS
        if pair.get(key) != candidate.get(key)
    ]
    if mismatched:
        raise SimpleEditStoreError(
            "final pair instruction contract differs from candidate metadata: "
            f"{candidate_id}; fields={mismatched}"
        )


def _read_nonempty_file(path: Path, field_name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SimpleEditStoreError(f"{field_name} is missing or unsafe: {path}")
    payload = path.read_bytes()
    if not payload:
        raise SimpleEditStoreError(f"{field_name} is empty: {path}")
    return payload


def _preflight_new_file(path: Path) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise SimpleEditStoreError(
            f"output parent is missing or unsafe: {path.parent}"
        )
    if os.path.lexists(path):
        raise OutputExistsError(f"refusing to overwrite existing path: {path}")


def _require_matching_file(
    path: Path,
    expected: bytes,
    field_name: str,
) -> None:
    observed = _read_nonempty_file(path, field_name)
    if observed != expected:
        raise SimpleEditStoreError(
            f"{field_name} differs from pending candidate: {path}"
        )


def _load_mapping(path: Path, field_name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SimpleEditStoreError(f"{field_name} is missing or unsafe: {path}")
    try:
        value = load_json_bytes(path.read_bytes())
    except Exception as exc:
        raise SimpleEditStoreError(f"{field_name} is not strict JSON") from exc
    return plain_copy(value, tuples_as_lists=True)


def _slot_state_payload(state: Mapping[str, Any]) -> bytes:
    slot_id = state.get("slot_id")
    if (
        type(slot_id) is not str
        or not slot_id
        or slot_id in {".", ".."}
        or "/" in slot_id
        or "\\" in slot_id
        or Path(slot_id).name != slot_id
        or state.get("format") != RUN_FORMAT
        or not isinstance(state.get("attempts"), list)
    ):
        raise SimpleEditStoreError("slot state is invalid")
    try:
        return canonical_json_bytes(state)
    except (TypeError, ValueError) as exc:
        raise SimpleEditStoreError("slot state is invalid") from exc


def candidate_artifact_names(candidate_id: str) -> frozenset[str]:
    validate_target_id(candidate_id)
    return frozenset(
        {
            "design_params.yaml",
            "body_measurements.yaml",
            f"{candidate_id}_specification.json",
            f"{candidate_id}_pattern.png",
            f"{candidate_id}_pattern.svg",
        }
    )


def _manifest_batch_id(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("batch_id")
    if type(value) is not str or _BATCH_ID_RE.fullmatch(value) is None:
        raise SimpleEditStoreError("run manifest batch_id is invalid")
    slots = manifest.get("slots")
    if not isinstance(slots, (list, tuple)):
        raise SimpleEditStoreError("run manifest slots must be a JSON list")
    source_key_pattern = re.compile(
        rf"{re.escape(value)}/random_body/(?:data/)?"
        r"rand_[A-Za-z0-9][A-Za-z0-9_-]*\Z",
        re.ASCII,
    )
    for slot in slots:
        if not isinstance(slot, Mapping):
            raise SimpleEditStoreError(
                "run manifest slots must contain mappings"
            )
        source_key = slot.get("source_key")
        if (
            type(source_key) is not str
            or source_key_pattern.fullmatch(source_key) is None
        ):
            raise SimpleEditStoreError(
                "run manifest batch_id does not match its source keys"
            )
    return value


class SimpleEditStore:
    """Filesystem state machine for one semantic-edit run."""

    def __init__(
        self,
        root: Path | str,
        *,
        batch_id: str,
        run_manifest: dict[str, Any],
    ) -> None:
        if type(batch_id) is not str or _BATCH_ID_RE.fullmatch(batch_id) is None:
            raise SimpleEditStoreError("batch_id is invalid")
        self._run_manifest = deepcopy(run_manifest)
        self.root = Path(root).expanduser().resolve()
        self.batch_id = batch_id
        self.batch_root = self.root / batch_id
        self.body_root = self.batch_root / "random_body"
        self.state_root = self.root / ".state"
        self.data_root = self.body_root / "data"
        self.patterns_vis_root = self.body_root / "patterns_vis"
        self.renders_root = self.body_root / "renders"
        self.snapshots_root = self.state_root / "source_snapshots"
        self.slot_states_root = self.state_root / "slot_states"
        self.candidate_metadata_root = self.state_root / "candidate_metadata"
        self.rejected_root = self.state_root / "rejected_candidates"
        self.rejected_simulations_root = (
            self.state_root / "rejected_simulations"
        )
        self.filter_reports_root = self.state_root / "filter_reports"
        self.work_root = self.state_root / "work"
        self.simulation_workspace_root = (
            self.state_root / "simulation_workspace"
        )
        self.run_manifest_path = self.root / "run_manifest.json"

    @classmethod
    def create(
        cls,
        root: Path | str,
        run_manifest: Mapping[str, Any],
    ) -> "SimpleEditStore":
        target = Path(root).expanduser().absolute()
        if target.exists() or target.is_symlink():
            raise OutputExistsError(f"output root already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(run_manifest, Mapping):
            raise SimpleEditStoreError("run manifest must be a JSON object")
        document = plain_copy(run_manifest, tuples_as_lists=True)
        batch_id = _manifest_batch_id(document)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.",
                suffix=".partial",
                dir=target.parent,
            )
        )
        try:
            store = cls(
                staging,
                batch_id=batch_id,
                run_manifest=document,
            )
            for directory in (
                store.state_root,
                store.data_root,
                store.patterns_vis_root,
                store.renders_root,
                store.snapshots_root,
                store.slot_states_root,
                store.candidate_metadata_root,
                store.rejected_root,
                store.rejected_simulations_root,
                store.filter_reports_root,
                store.work_root,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            if document.get("format") != RUN_FORMAT:
                raise SimpleEditStoreError("run manifest format is unsupported")
            write_new_bytes(store.run_manifest_path, canonical_json_bytes(document))
            publish_directory(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return cls(
            target,
            batch_id=batch_id,
            run_manifest=document,
        )

    @classmethod
    def open(cls, root: Path | str) -> "SimpleEditStore":
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir() or root_path.is_symlink():
            raise SimpleEditStoreError(
                f"output root is missing or unsafe: {root_path}"
            )
        manifest = _load_mapping(
            root_path / "run_manifest.json",
            "run manifest",
        )
        if manifest.get("format") != RUN_FORMAT:
            raise SimpleEditStoreError("run manifest format is unsupported")
        store = cls(
            root_path,
            batch_id=_manifest_batch_id(manifest),
            run_manifest=manifest,
        )
        for directory in (
            store.state_root,
            store.data_root,
            store.snapshots_root,
            store.slot_states_root,
            store.candidate_metadata_root,
            store.rejected_root,
            store.work_root,
        ):
            if directory.is_symlink() or not directory.is_dir():
                raise SimpleEditStoreError(
                    f"required store directory is missing or unsafe: {directory}"
                )
        return store

    @property
    def run_manifest(self) -> dict[str, Any]:
        return deepcopy(self._run_manifest)

    def snapshot_source(
        self,
        sample_id: str,
        *,
        design_bytes: bytes,
        body_bytes: bytes,
        specification_bytes: bytes,
    ) -> None:
        if not isinstance(sample_id, str) or not sample_id.startswith("rand_"):
            raise ValueError("source sample ID is invalid")
        if not design_bytes or not body_bytes:
            raise ValueError("source snapshot bytes must be non-empty")
        if type(specification_bytes) is not bytes or not specification_bytes:
            raise ValueError("source specification bytes must be non-empty")
        destination = self.snapshots_root / sample_id
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{sample_id}.", dir=self.work_root)
        )
        try:
            write_new_bytes(temporary / "design_params.yaml", design_bytes)
            write_new_bytes(temporary / "body_measurements.yaml", body_bytes)
            write_new_bytes(
                temporary / "source_specification.json",
                specification_bytes,
            )
            publish_directory(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def source_snapshot(self, sample_id: str) -> tuple[bytes, bytes]:
        directory = self.snapshots_root / sample_id
        try:
            design_bytes = (directory / "design_params.yaml").read_bytes()
            body_bytes = (directory / "body_measurements.yaml").read_bytes()
        except OSError as exc:
            raise SimpleEditStoreError(
                f"source snapshot is missing: {sample_id}"
            ) from exc
        if not design_bytes or not body_bytes:
            raise SimpleEditStoreError(
                f"source snapshot is empty: {sample_id}"
            )
        return design_bytes, body_bytes

    def source_specification(self, sample_id: str) -> bytes:
        path = (
            self.snapshots_root
            / sample_id
            / "source_specification.json"
        )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise SimpleEditStoreError(
                f"source specification snapshot is missing: {sample_id}"
            ) from exc
        if not payload:
            raise SimpleEditStoreError(
                f"source specification snapshot is empty: {sample_id}"
            )
        return payload

    def initialize_slot_state(
        self,
        *,
        slot_id: str,
        source_key: str,
        source_sample_id: str,
    ) -> None:
        state = {
            "format": RUN_FORMAT,
            "slot_id": slot_id,
            "source_key": source_key,
            "source_sample_id": source_sample_id,
            "attempts": [],
            "accepted_target_id": None,
        }
        path = self.slot_states_root / f"{slot_id}.json"
        if path.exists():
            existing = _load_mapping(path, f"slot state {slot_id}")
            if (
                existing.get("format") != RUN_FORMAT
                or existing.get("slot_id") != slot_id
                or existing.get("source_key") != source_key
                or existing.get("source_sample_id") != source_sample_id
                or not isinstance(existing.get("attempts"), list)
            ):
                raise SimpleEditStoreError(
                    f"conflicting slot state: {slot_id}"
                )
            return
        write_new_bytes(path, canonical_json_bytes(state))

    def load_slot_state(self, slot_id: str) -> dict[str, Any]:
        state = _load_mapping(
            self.slot_states_root / f"{slot_id}.json",
            f"slot state {slot_id}",
        )
        if (
            state.get("format") != RUN_FORMAT
            or state.get("slot_id") != slot_id
            or not isinstance(state.get("attempts"), list)
        ):
            raise SimpleEditStoreError(f"slot state is invalid: {slot_id}")
        return state

    def save_slot_state(self, state: Mapping[str, Any]) -> None:
        payload = _slot_state_payload(state)
        slot_id = state["slot_id"]
        assert isinstance(slot_id, str)
        self._save_slot_state_payload(slot_id, payload)

    def _save_slot_state_payload(self, slot_id: str, payload: bytes) -> None:
        atomic_replace_bytes(
            self.slot_states_root / f"{slot_id}.json",
            payload,
        )

    def candidate_metadata(self, candidate_id: str) -> dict[str, Any]:
        validate_target_id(candidate_id)
        metadata = _load_mapping(
            self.candidate_metadata_root / f"{candidate_id}.json",
            f"candidate metadata {candidate_id}",
        )
        if (
            metadata.get("format") != RUN_FORMAT
            or metadata.get("candidate_id") != candidate_id
        ):
            raise SimpleEditStoreError(
                f"candidate metadata identity is invalid: {candidate_id}"
            )
        _validate_instruction_payload(metadata, f"candidate metadata {candidate_id}")
        return metadata

    def publish_candidate(
        self,
        candidate_id: str,
        *,
        artifacts: Mapping[str, bytes],
        metadata: Mapping[str, Any],
    ) -> Path:
        validate_target_id(candidate_id)
        expected = candidate_artifact_names(candidate_id)
        if frozenset(artifacts) != expected:
            raise SimpleEditStoreError(
                f"candidate artifacts must be exact: {sorted(expected)}"
            )
        if any(type(payload) is not bytes or not payload for payload in artifacts.values()):
            raise SimpleEditStoreError("candidate artifacts must be non-empty bytes")
        document = dict(metadata)
        if (
            document.get("format") != RUN_FORMAT
            or document.get("candidate_id") != candidate_id
        ):
            raise SimpleEditStoreError("candidate metadata identity is invalid")
        _validate_instruction_payload(document, f"candidate metadata {candidate_id}")
        metadata_path = self.candidate_metadata_root / f"{candidate_id}.json"
        write_new_bytes(metadata_path, canonical_json_bytes(document))

        destination = self.data_root / candidate_id
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{candidate_id}.", dir=self.work_root)
        )
        try:
            for name in sorted(artifacts):
                write_new_bytes(temporary / name, artifacts[name])
            publish_directory(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def candidate_projection_bytes(
        self,
        candidate_id: str,
    ) -> tuple[bytes, bytes, bytes, bytes]:
        """Read every candidate input needed to finalize a physical pass."""

        source = self.data_root / candidate_id
        if source.is_symlink() or not source.is_dir():
            raise SimpleEditStoreError(
                f"pending candidate is missing or unsafe: {candidate_id}"
            )
        files = {
            name: _read_nonempty_file(source / name, f"candidate {candidate_id}")
            for name in candidate_artifact_names(candidate_id)
        }
        return (
            files["design_params.yaml"],
            files["body_measurements.yaml"],
            files[f"{candidate_id}_pattern.png"],
            files[f"{candidate_id}_pattern.svg"],
        )

    def prepare_projection_transition(
        self,
        *,
        candidate_id: str,
        workspace_root: Path | str,
        state: Mapping[str, Any],
        attempt_index: int,
        outcome: str,
        failure_message: str | None,
        accepted: AcceptedProjection | None,
    ) -> ProjectionTransition:
        """Validate and serialize one projection result without mutation."""

        validate_target_id(candidate_id)
        if not isinstance(state, Mapping):
            raise SimpleEditStoreError("slot state must be a mapping")
        updated = deepcopy(dict(state))
        attempts = updated.get("attempts")
        if not isinstance(attempts, list):
            raise SimpleEditStoreError("slot state has no attempts list")
        if updated.get("accepted_target_id") is not None or any(
            isinstance(current, Mapping)
            and current.get("status") == "accepted"
            for current in attempts
        ):
            raise SimpleEditStoreError("slot already has an accepted result")
        if type(attempt_index) is not int:
            raise SimpleEditStoreError("attempt index must be an integer")
        if attempt_index < 0 or attempt_index >= len(attempts):
            raise SimpleEditStoreError("attempt index is out of range")
        attempt = attempts[attempt_index]
        if type(attempt) is not dict:
            raise SimpleEditStoreError("indexed attempt must be a mapping")
        if attempt.get("candidate_id") != candidate_id:
            raise SimpleEditStoreError("indexed attempt does not match candidate")
        if attempt.get("status") != "in_simulation":
            raise SimpleEditStoreError("candidate is not awaiting projection")

        prompt_bytes: bytes | None = None
        pair_bytes: bytes | None = None
        png_bytes: bytes | None = None
        svg_bytes: bytes | None = None
        if outcome == "accepted":
            if not isinstance(accepted, AcceptedProjection):
                raise SimpleEditStoreError("accepted projection payload is required")
            if any(
                type(payload) is not bytes or not payload
                for payload in (
                    accepted.prompt_bytes,
                    accepted.png_bytes,
                    accepted.svg_bytes,
                )
            ):
                raise SimpleEditStoreError(
                    "accepted projection artifacts must be non-empty bytes"
                )
            if not isinstance(accepted.pair_record, Mapping):
                raise SimpleEditStoreError("final pair record must be a mapping")
            if not isinstance(accepted.candidate_metadata, Mapping):
                raise SimpleEditStoreError("candidate metadata must be a mapping")
            pair = deepcopy(dict(accepted.pair_record))
            metadata = deepcopy(dict(accepted.candidate_metadata))
            if (
                pair.get("format") != PAIR_FORMAT
                or pair.get("target_id") != candidate_id
                or pair.get("target_simulation_status") != "pass"
            ):
                raise SimpleEditStoreError("final pair record identity is invalid")
            _validate_instruction_payload(pair, f"final pair {candidate_id}")
            _validate_pair_matches_candidate(pair, metadata, candidate_id)
            pair_bytes = canonical_json_bytes(pair)
            prompt_bytes = accepted.prompt_bytes
            png_bytes = accepted.png_bytes
            svg_bytes = accepted.svg_bytes
            failure_code = None
        elif outcome in {"simulation_failed", "prompt_failed"}:
            if accepted is not None:
                raise SimpleEditStoreError(
                    "rejected projection cannot include accepted payload"
                )
            failure_code = (
                "physical_simulation_failed"
                if outcome == "simulation_failed"
                else "target_prompt_failed"
            )
        else:
            raise SimpleEditStoreError("projection outcome is invalid")

        if outcome == "accepted":
            updated["accepted_target_id"] = candidate_id
        attempt["status"] = outcome
        attempt["failure_code"] = failure_code
        attempt["failure_message"] = failure_message
        state_payload = _slot_state_payload(updated)
        slot_id = updated["slot_id"]
        assert isinstance(slot_id, str)

        workspace = Path(workspace_root).expanduser().resolve()
        random_body = workspace / "random_body"
        candidate = self.data_root / candidate_id
        product = random_body / candidate_id
        if (
            workspace.is_symlink()
            or workspace != self.simulation_workspace_root
            or random_body.is_symlink()
            or not random_body.is_dir()
        ):
            raise SimpleEditStoreError("simulation workspace is missing or unsafe")
        moves = (
            ((product, self.body_root / candidate_id),)
            if outcome == "accepted"
            else (
                (candidate, self.rejected_root / candidate_id),
                (product, self.rejected_simulations_root / candidate_id),
            )
        )
        for source, destination in moves:
            preflight_directory_publish(source, destination)
        if outcome == "accepted":
            if candidate.is_symlink() or not candidate.is_dir():
                raise SimpleEditStoreError(
                    f"pending candidate is missing or unsafe: {candidate_id}"
                )
            assert isinstance(png_bytes, bytes)
            assert isinstance(svg_bytes, bytes)
            for path, expected, field_name in (
                (
                    product / f"{candidate_id}_pattern.png",
                    png_bytes,
                    f"simulation product pattern PNG for {candidate_id}",
                ),
                (
                    product / f"{candidate_id}_pattern.svg",
                    svg_bytes,
                    f"simulation product pattern SVG for {candidate_id}",
                ),
            ):
                _require_matching_file(path, expected, field_name)
            for path in (
                candidate / f"{candidate_id}_prompts.json",
                candidate / f"{candidate_id}_edit_metadata.json",
                product / f"{candidate_id}_prompts.json",
                self.patterns_vis_root / f"{candidate_id}_pattern.png",
            ):
                _preflight_new_file(path)
        return ProjectionTransition(
            candidate_id=candidate_id,
            workspace_root=workspace,
            outcome=outcome,
            slot_id=slot_id,
            state=updated,
            state_payload=state_payload,
            prompt_bytes=prompt_bytes,
            pair_bytes=pair_bytes,
            png_bytes=png_bytes,
            svg_bytes=svg_bytes,
        )

    def commit_projection_transition(
        self,
        transition: ProjectionTransition,
    ) -> dict[str, Any]:
        """Commit one transition previously returned by ``prepare``."""

        candidate_id = transition.candidate_id
        workspace_product = (
            transition.workspace_root / "random_body" / candidate_id
        )
        candidate = self.data_root / candidate_id
        if transition.outcome == "accepted":
            assert all(
                payload is not None
                for payload in (
                    transition.prompt_bytes,
                    transition.pair_bytes,
                    transition.png_bytes,
                    transition.svg_bytes,
                )
            )
            prompt_bytes = transition.prompt_bytes
            pair_bytes = transition.pair_bytes
            png_bytes = transition.png_bytes
            svg_bytes = transition.svg_bytes
            assert isinstance(prompt_bytes, bytes)
            assert isinstance(pair_bytes, bytes)
            assert isinstance(png_bytes, bytes)
            assert isinstance(svg_bytes, bytes)
            product = self.body_root / candidate_id
            publish_directory(workspace_product, product)
            for path, payload in (
                (candidate / f"{candidate_id}_prompts.json", prompt_bytes),
                (candidate / f"{candidate_id}_edit_metadata.json", pair_bytes),
                (product / f"{candidate_id}_prompts.json", prompt_bytes),
                (
                    self.patterns_vis_root / f"{candidate_id}_pattern.png",
                    png_bytes,
                ),
            ):
                write_new_bytes(path, payload)
        else:
            publish_directory(candidate, self.rejected_root / candidate_id)
            publish_directory(
                workspace_product,
                self.rejected_simulations_root / candidate_id,
            )
        self._save_slot_state_payload(
            transition.slot_id,
            transition.state_payload,
        )
        return deepcopy(transition.state)

    def archive_filter_report(
        self,
        simulation_id: str,
        report_bytes: bytes,
    ) -> Path:
        path = self.preflight_filter_report(simulation_id, report_bytes)
        if not os.path.lexists(path):
            write_new_bytes(path, report_bytes)
        return path

    def preflight_filter_report(
        self,
        simulation_id: str,
        report_bytes: bytes,
    ) -> Path:
        """Validate archival, allowing only byte-identical reentry."""

        if (
            not isinstance(simulation_id, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", simulation_id) is None
            or not isinstance(report_bytes, bytes)
            or not report_bytes
        ):
            raise SimpleEditStoreError("filter report identity is invalid")
        digest = simulation_id.split(":", 1)[1]
        path = self.filter_reports_root / f"{digest}.yaml"
        if os.path.lexists(path):
            if _read_nonempty_file(path, "filter report") != report_bytes:
                raise SimpleEditStoreError(
                    "filter report archive differs from this simulation"
                )
            return path
        _preflight_new_file(path)
        return path

    def write_final_dataset_properties(
        self,
        batch_payload: bytes,
        body_payload: bytes,
    ) -> tuple[Path, Path]:
        if any(
            not isinstance(payload, bytes) or not payload
            for payload in (batch_payload, body_payload)
        ):
            raise SimpleEditStoreError(
                "final dataset properties must be nonempty bytes"
            )
        batch_path = self.batch_root / "dataset_properties.yaml"
        body_path = self.body_root / "dataset_properties_random_body.yaml"
        atomic_replace_bytes(batch_path, batch_payload)
        atomic_replace_bytes(body_path, body_payload)
        return batch_path, body_path

    def accepted_pairs_payload(
        self,
        states_by_slot: Mapping[str, Mapping[str, Any]],
        staged_pairs: Mapping[str, Mapping[str, Any]],
        staged_metadata: Mapping[str, Mapping[str, Any]],
    ) -> bytes:
        """Build the future accepted-pair inventory without writing it."""

        rows: list[Mapping[str, Any]] = []
        for slot_id in sorted(states_by_slot, key=str.encode):
            state = states_by_slot[slot_id]
            pair = staged_pairs.get(slot_id)
            candidate_id = pair.get("target_id") if pair else state.get(
                "accepted_target_id"
            )
            if candidate_id is None:
                continue
            validate_target_id(candidate_id)
            if pair is None:
                pair = _load_mapping(
                    self.data_root
                    / candidate_id
                    / f"{candidate_id}_edit_metadata.json",
                    f"final pair {candidate_id}",
                )
            if (
                pair.get("format") != PAIR_FORMAT
                or pair.get("target_id") != candidate_id
                or pair.get("slot_id") != slot_id
            ):
                raise SimpleEditStoreError(
                    f"accepted pair does not match slot state: {candidate_id}"
                )
            _validate_instruction_payload(pair, f"final pair {candidate_id}")
            candidate = staged_metadata.get(candidate_id)
            if candidate is None:
                candidate = self.candidate_metadata(candidate_id)
            _validate_pair_matches_candidate(
                pair,
                candidate,
                candidate_id,
            )
            rows.append(pair)
        return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)

    def write_accepted_pairs(self, payload: bytes) -> Path:
        path = self.root / self.batch_id / "accepted_pairs.jsonl"
        atomic_replace_bytes(path, payload)
        return path

    def rebuild_accepted_pairs(self) -> Path:
        states = {
            path.stem: _load_mapping(path, "slot state")
            for path in self.slot_states_root.glob("slot_*.json")
        }
        return self.write_accepted_pairs(
            self.accepted_pairs_payload(states, {}, {})
        )


def status(
    output_root: Path | str,
    *,
    split_path: Path | str | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    store = SimpleEditStore.open(output_root)
    manifest = store.run_manifest
    _assert_current_run_contract(manifest)
    slots = _slot_map(manifest)
    selection = load_split_selection(
        split_path=split_path,
        split_name=split_name,
        input_root=manifest["input_root"],
    )
    selected_ids = selected_slot_ids(
        slots,
        input_root=manifest["input_root"],
        selection=selection,
    )
    counts: Counter[str] = Counter()
    accepted_rules: Counter[str] = Counter()
    selected_slots = tuple(
        (slot_id, slot)
        for slot_id, slot in slots.items()
        if slot_id in selected_ids
    )
    print(
        f"[semantic-edit-status] {store.batch_id}: "
        f"selected={len(selected_slots)}",
        flush=True,
    )
    for progress_index, (slot_id, slot) in enumerate(
        selected_slots,
        start=1,
    ):
        state = store.load_slot_state(slot_id)
        phase = _classify_slot(state)
        counts[phase.value] += 1
        if phase is SlotPhase.ACCEPTED:
            accepted_rules[_accepted_rule_id(state)] += 1
        print(
            f"[semantic-edit-status] {store.batch_id}: "
            f"checked={progress_index}/{len(selected_slots)} "
            f"source={slot['source_sample_id']} state={phase.value}",
            flush=True,
        )
    workspace_info: dict[str, Any] = {
        "present": store.simulation_workspace_root.exists(),
        "active": counts["in_simulation"] + counts["pending_simulation"] > 0,
        "simulation_id": None,
        "candidate_count": 0,
        "candidate_ids": [],
    }
    if workspace_info["present"]:
        workspace = load_simulation_workspace(store.root)
        workspace_info.update(
            {
                "simulation_id": workspace.simulation_id,
                "candidate_count": len(workspace.candidate_ids),
                "candidate_ids": list(workspace.candidate_ids),
            }
        )
    result = {
        "format": RUN_FORMAT,
        "output_root": store.root.as_posix(),
        "batch_id": store.batch_id,
        "source_count": len(selected_ids),
        "state_counts": dict(sorted(counts.items())),
        "accepted_rule_distribution": dict(sorted(accepted_rules.items())),
        "max_physical_simulation_attempts_per_source": (
            MAX_PHYSICAL_SIMULATION_ATTEMPTS_PER_SOURCE
        ),
        "simulation_workspace": workspace_info,
    }
    if selection is not None:
        result["manifest_source_count"] = len(slots)
        result["split_selection"] = selection.payload()
    return result
