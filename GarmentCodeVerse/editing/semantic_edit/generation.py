"""Candidate generation for the single-operation semantic editing pipeline.

Generation consumes the cached capability report and source snapshots. A
no-edit reconstruction identifies panels unaffected by each edit; those panels
are restored from the source specification before validation and simulation.
Physical projection adds the final prompt after simulation passes.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from editing.canonical_io import (
    canonical_hash,
    design_dotted_get,
    load_body_yaml_bytes,
    load_design_yaml_bytes,
    load_json_bytes,
    plain_copy,
    sha256_bytes,
    serialize_and_reload_design,
)
from editing.config import EditConfig, config_digest
from editing.models import planned_operation_from_payload
from editing.pattern_compare import semantic_pattern_hash
from editing.pattern_renderer import PatternRenderError, render_pattern
from editing.publication import require_disjoint_roots
from editing.semantic_edit.instructions import (
    render_deterministic_specific_instruction,
)
from editing.semantic_edit.prompt import APPROVED_PROMPT_BUILDER_SHA256
from editing.simulation_workspace import prepare_simulation_workspace

from .planning import Capability, build_run_manifest, select_candidate
from .store import (
    MAX_PHYSICAL_SIMULATION_ATTEMPTS_PER_SOURCE,
    PLAN_FORMAT,
    RUN_FORMAT,
    SemanticEditError,
    SimpleEditStore,
    SplitSelection,
    SlotPhase,
    _RUN_CONTRACT,
    _assert_current_run_contract,
    _classify_slot,
    _physical_simulation_attempt_count,
    _regular_file,
    _slot_map,
    load_split_selection,
    selected_slot_ids,
)


_LOCALLY_RECHECKABLE_EFFECT_CODES = frozenset(
    {"effect/topology_mismatch", "undeclared_topology_change"}
)
_ACTIVE_STATUSES = frozenset(
    {"pending_simulation", "in_simulation", "accepted"}
)
class CandidateRejected(SemanticEditError):
    """One candidate failed a recoverable generation gate."""

    def __init__(self, code: str, message: str) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("candidate rejection code must be non-empty")
        if not isinstance(message, str) or not message:
            raise ValueError("candidate rejection message must be non-empty")
        self.code = code
        super().__init__(message)


class InstructionContractError(SemanticEditError):
    """Persisted instruction text disagrees with its semantic operation."""


def _effect_result_payload(result: Any | None) -> dict[str, Any] | None:
    """Return the complete canonical evidence emitted by one effect policy."""

    if result is None:
        return None
    evidence = plain_copy(result.evidence, tuples_as_lists=True)
    return {
        "passed": result.passed,
        "stage": result.stage,
        "code": result.code,
        "evidence": evidence,
    }


def _generation_failure_evidence(
    exc: Exception,
    *,
    strict_effect: Any | None,
    local_effect: Any | None,
) -> dict[str, Any]:
    """Preserve diagnostic detail without changing compact failure summaries."""

    message = str(exc) or type(exc).__name__
    return {
        "exception": {
            "class": type(exc).__name__,
            "message": message,
        },
        "effect": {
            "strict": _effect_result_payload(strict_effect),
            "local": _effect_result_payload(local_effect),
        },
    }


def _directory(path: Path, field_name: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise SemanticEditError(f"{field_name} is missing or unsafe: {candidate}")
    return candidate.resolve()


def _source_yaml_file(
    source_root: Path,
    sample_id: str,
    filename: str,
) -> Path:
    """Resolve canonical or sample-prefixed source YAML for snapshotting."""

    names = (filename, f"{sample_id}_{filename}")
    matches = tuple(
        source_root / name
        for name in names
        if os.path.lexists(source_root / name)
    )
    if len(matches) != 1:
        raise SemanticEditError(
            f"{sample_id} must have exactly one source YAML named "
            f"{names[0]!r} or {names[1]!r}"
        )
    return _regular_file(matches[0], f"{sample_id} source {filename}")


def _fresh_manifest(
    *,
    input_root: Path,
    analysis_root: Path,
    seed: str,
) -> dict[str, Any]:
    report_path = _regular_file(
        analysis_root / "pilot_eligibility_report.json",
        "cached eligibility report",
    )
    report = load_json_bytes(report_path.read_bytes())
    manifest = build_run_manifest(
        analysis_report_path=report_path,
        seed=seed,
    )
    if not isinstance(manifest, Mapping):
        raise SemanticEditError("manifest builder returned a non-mapping")
    result = dict(manifest)
    batch_ids = {
        slot["source_key"].split("/", 1)[0]
        for slot in _slot_map(result).values()
    }
    if (
        len(batch_ids) != 1
        or re.fullmatch(r"batch_[0-9]+", next(iter(batch_ids))) is None
    ):
        raise SemanticEditError("source plan must contain exactly one batch")
    batch_id = next(iter(batch_ids))
    result["batch_id"] = batch_id
    if report.get("format") != PLAN_FORMAT:
        raise SemanticEditError("source plan format is unsupported")
    if report.get("batch_id") != batch_id:
        raise SemanticEditError(
            "source-plan batch_id does not match its source keys"
        )
    config_path_value = report.get("config_path")
    if not isinstance(config_path_value, str):
        raise SemanticEditError("source plan has no config_path")
    config_path = _regular_file(
        Path(config_path_value),
        "source-plan config",
    )
    config_bytes = config_path.read_bytes()
    if report.get("config_file_sha256") != sha256_bytes(config_bytes):
        raise SemanticEditError("source-plan config identity changed")
    if (
        report.get("prompt_builder_sha256")
        != APPROVED_PROMPT_BUILDER_SHA256
    ):
        raise SemanticEditError(
            "source-plan prompt-builder identity changed"
        )
    from .registry import load_semantic_config

    config = load_semantic_config(config_path)
    if report.get("effective_config_hash") != config_digest(config):
        raise SemanticEditError(
            "source-plan effective config identity changed"
        )
    result.update(
        {
            "candidate_mode": "registry_planned",
            "effect_config_path": config_path.as_posix(),
            "effect_config_sha256": sha256_bytes(config_bytes),
            "effect_config_hash": config_digest(config),
        }
    )
    result.update(
        {
            **_RUN_CONTRACT,
            "input_root": input_root.as_posix(),
            "analysis_root": analysis_root.as_posix(),
            "analysis_report_path": report_path.as_posix(),
            "seed": seed,
        }
    )
    return result


def _manifest_identity_matches(
    manifest: Mapping[str, Any],
    *,
    input_root: Path,
    analysis_root: Path,
    seed: str,
) -> bool:
    config_path = _regular_file(
        Path(manifest["effect_config_path"]),
        "effect config",
    )
    config_bytes = config_path.read_bytes()
    from .registry import load_semantic_config

    config = load_semantic_config(config_path)
    expected = {
        **_RUN_CONTRACT,
        "candidate_mode": "registry_planned",
        "input_root": input_root.as_posix(),
        "analysis_root": analysis_root.as_posix(),
        "effect_config_path": config_path.as_posix(),
        "effect_config_sha256": sha256_bytes(config_bytes),
        "effect_config_hash": config_digest(config),
        "seed": seed,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def _open_or_initialize_run_and_snapshots(
    *,
    input_root: Path,
    analysis_root: Path,
    output_root: Path,
    seed: str,
    selection: SplitSelection | None,
) -> tuple[SimpleEditStore, dict[str, Any], frozenset[str]]:
    if output_root.exists() or output_root.is_symlink():
        store = SimpleEditStore.open(output_root)
        manifest = store.run_manifest
        if not _manifest_identity_matches(
            manifest,
            input_root=input_root,
            analysis_root=analysis_root,
            seed=seed,
        ):
            raise SemanticEditError(
                "existing output root belongs to a different run"
            )
    else:
        manifest = _fresh_manifest(
            input_root=input_root,
            analysis_root=analysis_root,
            seed=seed,
        )
        store = SimpleEditStore.create(output_root, manifest)

    _assert_current_run_contract(manifest)
    slots = _slot_map(manifest)
    selected_ids = selected_slot_ids(
        slots,
        input_root=input_root,
        selection=selection,
    )
    for slot_id, slot in slots.items():
        sample_id = slot["source_sample_id"]
        store.initialize_slot_state(
            slot_id=slot_id,
            source_key=slot["source_key"],
            source_sample_id=sample_id,
        )
        if slot_id not in selected_ids:
            continue
        snapshot_root = store.snapshots_root / sample_id
        if not snapshot_root.exists():
            source_root = input_root.joinpath(*slot["source_key"].split("/"))
            if source_root.is_symlink() or not source_root.is_dir():
                raise SemanticEditError(
                    f"{sample_id} source directory is missing or unsafe: "
                    f"{source_root}"
                )
            design_path = _source_yaml_file(
                source_root,
                sample_id,
                "design_params.yaml",
            )
            body_path = _source_yaml_file(
                source_root,
                sample_id,
                "body_measurements.yaml",
            )
            specification_paths = tuple(
                sorted(source_root.glob("*_specification.json"))
            )
            if len(specification_paths) != 1:
                raise SemanticEditError(
                    f"{sample_id} must have exactly one source specification"
                )
            specification_path = _regular_file(
                specification_paths[0],
                f"{sample_id} source specification",
            )
            store.snapshot_source(
                sample_id,
                design_bytes=design_path.read_bytes(),
                body_bytes=body_path.read_bytes(),
                specification_bytes=specification_path.read_bytes(),
            )
    return store, manifest, selected_ids


def _accepted_and_reserved_distribution(
    states: Iterable[Mapping[str, Any]],
) -> tuple[Counter[str], Counter[str]]:
    """Count final acceptances plus reservations in the active round.

    Failed attempts never occupy distribution quota.
    """
    rule_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter()
    for state in states:
        for attempt in state["attempts"]:
            if attempt.get("status") not in _ACTIVE_STATUSES:
                continue
            rule_counts[attempt["rule_id"]] += 1
            capability_counts[attempt["capability_key"]] += 1
    return rule_counts, capability_counts


def _apply_single_cached_operation(
    design: Mapping[str, Any],
    planned_payload: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    planned = planned_operation_from_payload(planned_payload)
    operation = planned.operation
    writes = tuple(operation.writes)
    if (
        not writes
        or set(operation.before) != set(writes)
        or set(operation.after) != set(writes)
    ):
        raise SemanticEditError("cached operation has an invalid write contract")

    try:
        selected_rule = registry[operation.rule_id]
    except KeyError as exc:
        raise SemanticEditError(
            f"cached operation uses an unknown rule: {operation.rule_id}"
        ) from exc
    if (
        getattr(selected_rule, "id", None) != operation.rule_id
        or not callable(getattr(selected_rule, "apply", None))
    ):
        raise SemanticEditError(
            f"cached operation rule is invalid: {operation.rule_id}"
        )

    source = plain_copy(design, tuples_as_lists=True)
    for path in writes:
        current = design_dotted_get(source, path)
        before = operation.before[path]
        after = operation.after[path]
        before_hash = canonical_hash({"value": before})
        after_hash = canonical_hash({"value": after})
        if canonical_hash({"value": current}) != before_hash:
            raise SemanticEditError(
                f"cached operation source value drifted at {path}"
            )
        if before_hash == after_hash:
            raise SemanticEditError(
                f"cached operation does not change its parameter at {path}"
            )

    try:
        applied = selected_rule.apply(
            plain_copy(source, tuples_as_lists=True),
            planned,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateRejected(
            "operation_apply_failed",
            f"{operation.rule_id} apply failed: {type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(applied, Mapping):
        raise CandidateRejected(
            "operation_apply_failed",
            f"{operation.rule_id} apply returned a non-mapping result",
        )
    edited = plain_copy(applied, tuples_as_lists=True)

    for path in writes:
        after = operation.after[path]
        if canonical_hash(
            {"value": design_dotted_get(edited, path)}
        ) != canonical_hash({"value": after}):
            raise CandidateRejected(
                "operation_apply_mismatch",
                f"edited parameter was not applied at {path}"
            )
    return edited, planned, selected_rule


def _attempt_candidate(
    *,
    store: SimpleEditStore,
    slot: Mapping[str, Any],
    state: dict[str, Any],
    candidate: Mapping[str, Any],
    effect_config: EditConfig,
    registry: Mapping[str, Any],
    source_rebuild_cache: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    attempt_index = len(state["attempts"])
    candidate_id = (
        f"{slot['source_sample_id']}__edit_{attempt_index:03d}"
    )
    capability = candidate["capability"]
    capability_value = Capability.from_payload(capability)
    base_attempt = {
        "attempt_index": attempt_index,
        "candidate_id": candidate_id,
        "operation_signature": candidate["operation_signature"],
        "rule_id": capability_value.rule_id,
        "capability_key": capability_value.count_key,
        "capability": dict(capability),
        "simulation_id": None,
    }
    from .construction import (
        PANEL_COMPARISON_ATOL, InvalidTargetSpecification,
        build_target_pattern, preserve_unchanged_panels,
    )

    strict_effect: Any | None = None
    local_effect: Any | None = None
    try:
        source_design_bytes, source_body_bytes = store.source_snapshot(
            slot["source_sample_id"]
        )
        source_design = load_design_yaml_bytes(source_design_bytes)
        edited, planned, selected_rule = _apply_single_cached_operation(
            source_design,
            candidate["planned_operation_payload"],
            registry,
        )
        if planned.operation.rule_id != capability_value.rule_id:
            raise SemanticEditError(
                "cached operation rule does not match its capability"
            )
        try:
            instruction = render_deterministic_specific_instruction(
                planned,
                source_key=slot["source_key"],
                expected_operation_signature=(
                    candidate["operation_signature"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise InstructionContractError(
                "instruction renderer rejected a cached operation"
            ) from exc
        serialized = serialize_and_reload_design(edited)
        built = build_target_pattern(
            serialized.design,
            source_body_bytes,
            candidate_id,
        )
        if not built.success or built.specification is None:
            failure_code = built.failure_code or "construction/invalid_pattern"
            failure_message = (
                built.failure_message or "target pattern construction failed"
            )
            raise CandidateRejected(failure_code, failure_message)
        effect_gate = "pass"
        ignored_effect_code: str | None = None
        source_specification = load_json_bytes(
            store.source_specification(slot["source_sample_id"])
        )
        if source_rebuild_cache is None:
            source_rebuild_cache = {}
        source_id = slot["source_sample_id"]
        if source_id not in source_rebuild_cache:
            source_rebuild_cache[source_id] = build_target_pattern(
                source_design, source_body_bytes, source_id,
            )
        baseline = source_rebuild_cache[source_id]
        if not baseline.success:
            raise CandidateRejected(
                "source_reconstruction_failed",
                baseline.failure_message or "source reconstruction failed",
            )
        try:
            built, preserved_panels = preserve_unchanged_panels(
                source_specification, baseline.specification, built.specification,
            )
        except InvalidTargetSpecification as exc:
            raise CandidateRejected(
                "panel_preservation_failed", str(exc)
            ) from exc
        source_body = load_body_yaml_bytes(source_body_bytes)
        strict_effect = selected_rule.validate(
            source_design=source_design,
            target_design=serialized.design,
            body=source_body,
            source_specification=source_specification,
            target_specification=built.specification,
            planned_operation=planned,
            config=effect_config,
        )
        if not strict_effect.passed:
            if strict_effect.code not in _LOCALLY_RECHECKABLE_EFFECT_CODES:
                raise CandidateRejected(
                    strict_effect.code,
                    f"{strict_effect.stage}: {strict_effect.code}",
                )
            local_effect = selected_rule.validate(
                source_design=source_design,
                target_design=serialized.design,
                body=source_body,
                source_specification=source_specification,
                target_specification=built.specification,
                planned_operation=planned,
                config=effect_config,
                topology_policy="local",
            )
            if not local_effect.passed:
                raise CandidateRejected(
                    local_effect.code,
                    (
                        f"{local_effect.stage}: {local_effect.code}; "
                        f"strict_effect_code={strict_effect.code}"
                    ),
                )
            effect_gate = "pass_local_topology"
            ignored_effect_code = strict_effect.code
        try:
            rendered = render_pattern(
                built.specification,
                candidate_id,
                store.work_root,
            )
        except PatternRenderError as exc:
            raise CandidateRejected(
                "PatternRenderError",
                str(exc) or type(exc).__name__,
            ) from exc
        specification_hash = semantic_pattern_hash(built.specification)
        metadata = {
            "format": RUN_FORMAT,
            "candidate_id": candidate_id,
            "slot_id": slot["slot_id"],
            "source_key": slot["source_key"],
            "source_sample_id": slot["source_sample_id"],
            "attempt_index": attempt_index,
            "operation_count": 1,
            "operation_signature": candidate["operation_signature"],
            "operation": planned.to_payload(),
            "capability": dict(capability),
            "specific_instruction": instruction.text,
            "canonical_instruction": instruction.canonical_text,
            "instruction_provenance": instruction.provenance_payload(),
            "target_design_hash": serialized.canonical_hash,
            "target_pattern_hash": specification_hash,
            "target_specification_sha256": (
                built.specification_sha256
            ),
            "construction_gates": {
                "target_construction": "pass",
                "source_reconstruction": "pass",
                "panel_preservation": {
                    "status": "pass",
                    "preserved_panels": list(preserved_panels),
                    "comparison_atol": PANEL_COMPARISON_ATOL,
                },
                "panel_self_intersection": "pass",
                "pattern_data_complete_and_finite": "pass",
                "edited_parameter_changed": "pass",
                "visible_effect": effect_gate,
                "ignored_effect_code": ignored_effect_code,
                "target_pattern_render": "pass",
            },
        }
        store.publish_candidate(
            candidate_id,
            artifacts={
                "design_params.yaml": serialized.yaml_bytes,
                "body_measurements.yaml": source_body_bytes,
                f"{candidate_id}_specification.json": (
                    built.specification_json_bytes
                ),
                f"{candidate_id}_pattern.png": rendered.png_bytes,
                f"{candidate_id}_pattern.svg": rendered.svg_bytes,
            },
            metadata=metadata,
        )
    except (
        CandidateRejected,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as exc:
        if isinstance(exc, CandidateRejected):
            failure_code = exc.code
            failure_message = str(exc)
        else:
            failure_code = "generation_exception"
            failure_message = f"{type(exc).__name__}: {exc}"
        failure = {
            **base_attempt,
            "status": "generation_failed",
            "failure_code": failure_code,
            "failure_message": failure_message,
            "failure_evidence": _generation_failure_evidence(
                exc,
                strict_effect=strict_effect,
                local_effect=local_effect,
            ),
        }
        state["attempts"].append(failure)
        store.save_slot_state(state)
        return False, failure
    passed = {
        **base_attempt,
        "physical_attempt_number": (
            _physical_simulation_attempt_count(state) + 1
        ),
        "status": "pending_simulation",
        "failure_code": None,
        "failure_message": None,
    }
    state["attempts"].append(passed)
    store.save_slot_state(state)
    return True, passed


def generate(
    *,
    input_root: Path | str,
    analysis_root: Path | str,
    output_root: Path | str,
    seed: str = "42",
    split_path: Path | str | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    """Generate one pending singleton candidate per unresolved source."""

    input_path = _directory(Path(input_root), "input root")
    analysis_path = _directory(Path(analysis_root), "analysis root")
    roots = require_disjoint_roots(
        input_root=input_path,
        analysis_root=analysis_path,
        output_root=output_root,
    )
    input_path = roots["input_root"]
    analysis_path = roots["analysis_root"]
    output_path = roots["output_root"]
    selection = load_split_selection(
        split_path=split_path,
        split_name=split_name,
        input_root=input_path,
    )
    workspace_path = output_path / ".state" / "simulation_workspace"
    if os.path.lexists(workspace_path):
        raise SemanticEditError(
            "project the existing simulation workspace before generating"
        )
    store, manifest, selected_ids = _open_or_initialize_run_and_snapshots(
        input_root=input_path,
        analysis_root=analysis_path,
        output_root=output_path,
        seed=str(seed),
        selection=selection,
    )
    slots = _slot_map(manifest)
    states_by_slot = {
        slot_id: store.load_slot_state(slot_id)
        for slot_id in slots
    }
    phases_by_slot = {
        slot_id: _classify_slot(state)
        for slot_id, state in states_by_slot.items()
    }
    unfinished_projection_slots = [
        slot_id
        for slot_id, phase in phases_by_slot.items()
        if phase is SlotPhase.IN_SIMULATION
    ]
    if unfinished_projection_slots:
        raise SemanticEditError(
            "finish project/finalization before generating more candidates; "
            f"unfinished_slots={len(unfinished_projection_slots)}"
        )
    from .registry import build_semantic_registry, load_semantic_config

    effect_config = load_semantic_config(
        Path(manifest["effect_config_path"])
    )
    registry = build_semantic_registry(effect_config)
    generated: list[str] = []
    exhausted: list[str] = []
    physical_limit_reached: list[str] = []
    generation_failures = 0
    rule_counts, capability_counts = (
        _accepted_and_reserved_distribution(states_by_slot.values())
    )
    selected_slots = tuple(
        (slot_id, slot)
        for slot_id, slot in slots.items()
        if slot_id in selected_ids
    )
    print(
        f"[semantic-edit-generate] {store.batch_id}: "
        f"selected={len(selected_slots)}",
        flush=True,
    )
    for progress_index, (slot_id, slot) in enumerate(
        selected_slots,
        start=1,
    ):
        state = states_by_slot[slot_id]
        phase = phases_by_slot[slot_id]
        progress = (
            f"[semantic-edit-generate] {store.batch_id}: "
            f"sample={progress_index}/{len(selected_slots)} "
            f"source={slot['source_sample_id']}"
        )
        print(f"{progress} phase={phase.value}", flush=True)
        if phase is SlotPhase.ACCEPTED:
            print(f"{progress} result=already_accepted", flush=True)
            continue
        if phase in {
            SlotPhase.IN_SIMULATION,
            SlotPhase.PENDING_SIMULATION,
        }:
            print(f"{progress} result=already_pending", flush=True)
            continue
        if phase is SlotPhase.PHYSICAL_LIMIT_REACHED:
            physical_limit_reached.append(slot_id)
            print(f"{progress} result=physical_limit_reached", flush=True)
            continue
        source_rebuild_cache: dict[str, Any] = {}
        while True:
            attempted = {
                attempt["operation_signature"]
                for attempt in state["attempts"]
            }
            candidate = select_candidate(
                slot,
                attempted_signatures=attempted,
                rule_counts=rule_counts,
                capability_counts=capability_counts,
            )
            if candidate is None:
                exhausted.append(slot_id)
                print(f"{progress} result=candidates_exhausted", flush=True)
                break
            passed, attempt = _attempt_candidate(
                store=store,
                slot=slot,
                state=state,
                candidate=candidate,
                effect_config=effect_config,
                registry=registry,
                source_rebuild_cache=source_rebuild_cache,
            )
            if passed:
                generated.append(attempt["candidate_id"])
                rule_counts[attempt["rule_id"]] += 1
                capability_counts[attempt["capability_key"]] += 1
                print(
                    f"{progress} result=generated "
                    f"candidate={attempt['candidate_id']}",
                    flush=True,
                )
                break
            generation_failures += 1
            print(
                f"{progress} result=candidate_failed "
                f"candidate={attempt['candidate_id']} "
                f"code={attempt['failure_code']}",
                flush=True,
            )

    pending: list[dict[str, Any]] = []
    pending_slot_ids: list[str] = []
    for slot_id in slots:
        if slot_id not in selected_ids:
            continue
        state = states_by_slot[slot_id]
        for attempt in state["attempts"]:
            if attempt["status"] != "pending_simulation":
                continue
            candidate_id = attempt["candidate_id"]
            pending.append(
                {
                    "candidate_id": candidate_id,
                    "target_id": candidate_id,
                    "target_path": f"{store.batch_id}/random_body/data/{candidate_id}",
                    "target_simulation_status": "pending",
                }
            )
            pending_slot_ids.append(slot_id)

    workspace = None
    if pending:
        workspace = prepare_simulation_workspace(
            store.root,
            pending,
            global_seed=str(seed),
            batch_id=store.batch_id,
        )
        for slot_id in pending_slot_ids:
            state = states_by_slot[slot_id]
            for attempt in state["attempts"]:
                if attempt["status"] == "pending_simulation":
                    attempt["status"] = "in_simulation"
                    attempt["simulation_id"] = workspace.simulation_id
            store.save_slot_state(state)

    accepted_pairs_path = store.batch_root / "accepted_pairs.jsonl"
    if os.path.lexists(accepted_pairs_path):
        if accepted_pairs_path.is_symlink() or not accepted_pairs_path.is_file():
            raise SemanticEditError("accepted_pairs.jsonl is unsafe")
    else:
        store.rebuild_accepted_pairs()
    accepted_count = sum(
        phase is SlotPhase.ACCEPTED
        for phase in phases_by_slot.values()
    )
    result = {
        "format": RUN_FORMAT,
        "output_root": store.root.as_posix(),
        "batch_id": store.batch_id,
        "source_count": len(slots),
        "new_candidate_count": len(generated),
        "new_candidate_ids": generated,
        "generation_failure_count": generation_failures,
        "exhausted_slot_count": len(exhausted),
        "exhausted_slot_ids": exhausted,
        "max_physical_simulation_attempts_per_source": (
            MAX_PHYSICAL_SIMULATION_ATTEMPTS_PER_SOURCE
        ),
        "physical_limit_reached_slot_count": len(physical_limit_reached),
        "physical_limit_reached_slot_ids": physical_limit_reached,
        "accepted_pair_count": accepted_count,
        "simulation_workspace": (
            workspace.root.as_posix() if workspace is not None else None
        ),
        "simulation_id": (
            workspace.simulation_id if workspace is not None else None
        ),
        "simulation_candidate_count": len(pending),
        "simulation_candidate_ids": [row["candidate_id"] for row in pending],
        "rule_distribution": dict(sorted(rule_counts.items())),
        "capability_distribution": dict(sorted(capability_counts.items())),
    }
    if selection is not None:
        result["selected_source_count"] = len(selected_ids)
        result["selected_accepted_pair_count"] = sum(
            phases_by_slot[slot_id] is SlotPhase.ACCEPTED
            for slot_id in selected_ids
        )
        result["split_selection"] = selection.payload()
    return result
