"""Physical projection and final publication for semantic edits."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from editing.canonical_io import (
    dump_yaml_document,
    load_json_bytes,
    load_yaml_bytes,
    plain_copy,
)
from editing.semantic_edit.prompt import PromptBuilder, PromptError
from editing.simulation_workspace import (
    load_filter_projection,
    load_simulation_workspace,
    remove_simulation_workspace,
)

from .store import (
    PAIR_FORMAT,
    RUN_FORMAT,
    AcceptedProjection,
    ProjectionTransition,
    SemanticEditError,
    SimpleEditStore,
    SlotPhase,
    _assert_current_run_contract,
    _classify_slot,
    _regular_file,
    _slot_map,
)


@dataclass(frozen=True)
class ValidatedProjection:
    result: Mapping[str, Any]
    metadata: Mapping[str, Any]
    state: dict[str, Any]
    attempt_index: int
    decision: Literal["pass", "fail"]

    @property
    def candidate_id(self) -> str:
        return str(self.result["candidate_id"])


@dataclass(frozen=True)
class PreparedProjection:
    validated: ValidatedProjection
    accepted: AcceptedProjection | None
    transition: ProjectionTransition


def _find_attempt(
    state: dict[str, Any],
    candidate_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if metadata.get("candidate_id") != candidate_id:
        raise SemanticEditError(
            f"candidate metadata identity drifted: {candidate_id}"
        )
    if metadata.get("slot_id") != state.get("slot_id"):
        raise SemanticEditError(
            f"candidate slot identity drifted: {candidate_id}"
        )
    if (
        metadata.get("source_key") != state.get("source_key")
        or metadata.get("source_sample_id")
        != state.get("source_sample_id")
    ):
        raise SemanticEditError(
            f"candidate source identity drifted: {candidate_id}"
        )
    matches = [
        attempt
        for attempt in state["attempts"]
        if attempt.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise SemanticEditError(
            f"candidate does not have exactly one slot attempt: {candidate_id}"
        )
    attempt = matches[0]
    if attempt.get("attempt_index") != metadata.get("attempt_index"):
        raise SemanticEditError(
            f"candidate attempt identity drifted: {candidate_id}"
        )
    if attempt.get("operation_signature") != metadata.get(
        "operation_signature"
    ):
        raise SemanticEditError(
            f"candidate operation identity drifted: {candidate_id}"
        )
    return attempt


def _final_pair_record(
    *,
    candidate_id: str,
    metadata: Mapping[str, Any],
    prompt: Any,
    simulation_result: Mapping[str, Any],
    batch_id: str,
) -> dict[str, Any]:
    return {
        "format": PAIR_FORMAT,
        "slot_id": metadata["slot_id"],
        "source_key": metadata["source_key"],
        "source_sample_id": metadata["source_sample_id"],
        "source_path": metadata["source_key"],
        "target_id": candidate_id,
        "target_path": f"{batch_id}/random_body/data/{candidate_id}",
        "specific_instruction": metadata["specific_instruction"],
        "canonical_instruction": metadata["canonical_instruction"],
        "instruction_provenance": plain_copy(
            metadata["instruction_provenance"], tuples_as_lists=True
        ),
        "operation_count": 1,
        "operation_signature": metadata["operation_signature"],
        "operation": metadata["operation"],
        "capability": metadata["capability"],
        "target_design_hash": metadata["target_design_hash"],
        "target_pattern_hash": metadata["target_pattern_hash"],
        "target_specification_sha256": metadata[
            "target_specification_sha256"
        ],
        "construction_gates": metadata["construction_gates"],
        "target_prompt": {
            "path": f"{candidate_id}_prompts.json",
            "prompt_builder_sha256": prompt.builder_sha256,
        },
        "target_simulation_status": "pass",
        "simulation_provenance": plain_copy(
            simulation_result, tuples_as_lists=True
        ),
    }


def _merge_accepted_statistics(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    accepted_ids: frozenset[str],
) -> dict[str, Any]:
    """Merge only raw per-case GarmentCode statistics for accepted targets."""

    merged: dict[str, Any] = {}
    keys = set(previous) | set(current)
    for key in sorted(keys):
        if key == "fails":
            categories: set[str] = set()
            for source in (previous.get(key), current.get(key)):
                if isinstance(source, Mapping):
                    categories.update(str(item) for item in source)
            merged[key] = {
                category: [] for category in sorted(categories)
            }
            continue
        if key == "fails_count":
            merged[key] = 0
            continue
        old_value = previous.get(key)
        new_value = current.get(key)
        if not isinstance(old_value, Mapping) and not isinstance(
            new_value, Mapping
        ):
            continue
        values: dict[str, Any] = {}
        if isinstance(old_value, Mapping):
            values.update(plain_copy(old_value, tuples_as_lists=True))
        if isinstance(new_value, Mapping):
            values.update(plain_copy(new_value, tuples_as_lists=True))
        filtered = {
            candidate_id: values[candidate_id]
            for candidate_id in sorted(accepted_ids, key=str.encode)
            if candidate_id in values
        }
        if filtered:
            merged[key] = filtered
    return merged


def _final_dataset_properties_payloads(
    store: SimpleEditStore,
    accepted_ids: tuple[str, ...],
    *,
    seed: str,
    simulation_properties: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    panel_counts: dict[str, int] = {}
    for candidate_id in accepted_ids:
        specification_path = store.data_root / candidate_id / (
            f"{candidate_id}_specification.json"
        )
        specification = load_json_bytes(
            _regular_file(
                specification_path,
                f"accepted specification {candidate_id}",
            ).read_bytes()
        )
        pattern = specification.get("pattern")
        panels = pattern.get("panels") if isinstance(pattern, Mapping) else None
        if not isinstance(panels, Mapping) or not panels:
            raise SemanticEditError(
                f"accepted specification has no panels: {candidate_id}"
            )
        panel_counts[candidate_id] = len(panels)
    accepted_set = frozenset(accepted_ids)
    batch_properties = {
        "design_file": "./assets/design_params/design_beautiful.yaml",
        "body_default": "mean_all",
        "body_samples": ".",
        "name": f"{store.root.name}/{store.batch_id}",
        "to_subfolders": True,
        "size": len(accepted_ids),
        "target_count": len(accepted_ids),
        "generator": {
            "config": {"random_seed": seed},
            "stats": {"panel_count": panel_counts},
        },
    }
    previous_body: Mapping[str, Any] = {}
    body_properties_path = (
        store.body_root / "dataset_properties_random_body.yaml"
    )
    if body_properties_path.is_file() and not body_properties_path.is_symlink():
        previous_body = load_yaml_bytes(body_properties_path.read_bytes())
    body_properties = {
        **plain_copy(previous_body, tuples_as_lists=True),
        **plain_copy(simulation_properties, tuples_as_lists=True),
        **batch_properties,
        "data_folder": store.root.as_posix(),
        "frozen": True,
    }
    for section_name in ("generator", "sim", "render"):
        previous_section = previous_body.get(section_name)
        current_section = simulation_properties.get(section_name)
        previous_section = (
            previous_section if isinstance(previous_section, Mapping) else {}
        )
        current_section = (
            current_section if isinstance(current_section, Mapping) else {}
        )
        section = {
            **plain_copy(previous_section, tuples_as_lists=True),
            **plain_copy(current_section, tuples_as_lists=True),
        }
        previous_stats = previous_section.get("stats")
        current_stats = current_section.get("stats")
        section["stats"] = _merge_accepted_statistics(
            previous_stats if isinstance(previous_stats, Mapping) else {},
            current_stats if isinstance(current_stats, Mapping) else {},
            accepted_set,
        )
        body_properties[section_name] = section
    body_properties["generator"]["stats"]["panel_count"] = panel_counts
    return (
        dump_yaml_document(batch_properties),
        dump_yaml_document(body_properties),
    )


def _validate_projection_records(
    *,
    results: tuple[Mapping[str, Any], ...],
    store: SimpleEditStore,
    states_by_slot: Mapping[str, dict[str, Any]],
    simulation_id: str,
) -> tuple[ValidatedProjection, ...]:
    """Validate identities and return records without persistent mutation."""

    validated: list[ValidatedProjection] = []
    seen_slots: set[str] = set()
    for result in results:
        candidate_id = result["candidate_id"]
        metadata = store.candidate_metadata(candidate_id)
        slot_id = metadata.get("slot_id")
        if not isinstance(slot_id, str) or slot_id not in states_by_slot:
            raise SemanticEditError(
                f"candidate slot is not in the run manifest: {candidate_id}"
            )
        if slot_id in seen_slots:
            raise SemanticEditError(
                f"simulation has multiple candidates for one slot: {slot_id}"
            )
        seen_slots.add(slot_id)
        state = states_by_slot[slot_id]
        if _classify_slot(state) is not SlotPhase.IN_SIMULATION:
            raise SemanticEditError("candidate is not awaiting projection")
        attempt = _find_attempt(state, candidate_id, metadata)
        if attempt["status"] != "in_simulation":
            raise SemanticEditError("candidate is not awaiting projection")
        if attempt.get("simulation_id") != simulation_id:
            raise SemanticEditError("candidate belongs to another simulation")
        decision_value = result.get("target_simulation_status")
        if decision_value == "pass":
            decision: Literal["pass", "fail"] = "pass"
        elif decision_value == "fail":
            decision = "fail"
        else:
            raise SemanticEditError(
                f"candidate has an invalid simulation decision: {candidate_id}"
            )
        attempt_index = next(
            index
            for index, current in enumerate(state["attempts"])
            if current is attempt
        )
        validated.append(
            ValidatedProjection(
                result=result,
                metadata=metadata,
                state=state,
                attempt_index=attempt_index,
                decision=decision,
            )
        )
    return tuple(validated)


def _load_projection_builder(
    *,
    records: tuple[ValidatedProjection, ...],
    manifest: Mapping[str, Any],
    prompt_builder: Path | str,
) -> PromptBuilder | None:
    """Load the prompt builder only when at least one pass needs it."""

    if not any(record.decision == "pass" for record in records):
        return None
    builder_path = _regular_file(
        Path(prompt_builder),
        "prompt builder",
    )
    return PromptBuilder.load(
        builder_path,
        expected_sha256=manifest["prompt_builder_sha256"],
    )


def _prepare_projection_records(
    *,
    records: tuple[ValidatedProjection, ...],
    store: SimpleEditStore,
    workspace_root: Path,
    builder: PromptBuilder | None,
) -> tuple[PreparedProjection, ...]:
    """Render pass prompts and preflight every candidate without mutation."""

    prepared: list[PreparedProjection] = []
    for progress_index, record in enumerate(records, start=1):
        candidate_id = record.candidate_id
        progress = (
            "[semantic-edit-project] "
            f"candidate={progress_index}/{len(records)} id={candidate_id}"
        )
        print(f"{progress} decision={record.decision}", flush=True)
        accepted: AcceptedProjection | None = None
        if record.decision == "fail":
            outcome: Literal[
                "accepted", "simulation_failed", "prompt_failed"
            ] = "simulation_failed"
            failure_message = "; ".join(
                record.result.get("failure_reasons", ())
            ) or "physical simulation failed"
        else:
            if builder is None:
                raise SemanticEditError("prompt builder was not initialized")
            design, body, png_bytes, svg_bytes = (
                store.candidate_projection_bytes(candidate_id)
            )
            try:
                prompt = builder.render(
                    design,
                    body,
                    output_root=store.root,
                )
            except PromptError as exc:
                outcome = "prompt_failed"
                failure_message = str(exc) or type(exc).__name__
            else:
                outcome = "accepted"
                failure_message = None
                accepted = AcceptedProjection(
                    prompt_bytes=prompt.json_bytes,
                    pair_record=_final_pair_record(
                        candidate_id=candidate_id,
                        metadata=record.metadata,
                        prompt=prompt,
                        simulation_result=record.result,
                        batch_id=store.batch_id,
                    ),
                    candidate_metadata=record.metadata,
                    png_bytes=png_bytes,
                    svg_bytes=svg_bytes,
                )
        transition = store.prepare_projection_transition(
            candidate_id=candidate_id,
            workspace_root=workspace_root,
            state=record.state,
            attempt_index=record.attempt_index,
            outcome=outcome,
            failure_message=failure_message,
            accepted=accepted,
        )
        prepared.append(
            PreparedProjection(
                validated=record,
                accepted=accepted,
                transition=transition,
            )
        )
        print(f"{progress} outcome={outcome}", flush=True)
    return tuple(prepared)


def _prepare_projection_summary(
    *,
    store: SimpleEditStore,
    manifest: Mapping[str, Any],
    states_by_slot: Mapping[str, dict[str, Any]],
    records: tuple[PreparedProjection, ...],
    simulation_properties: Mapping[str, Any],
) -> tuple[tuple[str, ...], bytes, bytes, bytes]:
    """Build accepted ids, accepted-pairs bytes, and property bytes."""

    newly_accepted = [
        record.validated.candidate_id
        for record in records
        if record.accepted is not None
    ]
    accepted_ids = tuple(
        sorted(
            [
                state["accepted_target_id"]
                for state in states_by_slot.values()
                if state["accepted_target_id"] is not None
            ]
            + newly_accepted,
            key=str.encode,
        )
    )
    staged_pairs = {
        record.accepted.pair_record["slot_id"]: record.accepted.pair_record
        for record in records
        if record.accepted is not None
    }
    for path in (
        store.batch_root / "accepted_pairs.jsonl",
        store.batch_root / "dataset_properties.yaml",
        store.body_root / "dataset_properties_random_body.yaml",
    ):
        if path.parent.is_symlink() or not path.parent.is_dir() or (
            os.path.lexists(path) and (path.is_symlink() or not path.is_file())
        ):
            raise SemanticEditError(f"final output is unsafe: {path}")
    accepted_pairs = store.accepted_pairs_payload(
        states_by_slot,
        staged_pairs,
        {
            record.validated.candidate_id: record.validated.metadata
            for record in records
        },
    )
    batch_properties, body_properties = _final_dataset_properties_payloads(
        store,
        accepted_ids,
        seed=str(manifest["seed"]),
        simulation_properties=simulation_properties,
    )
    return accepted_ids, accepted_pairs, batch_properties, body_properties


def _commit_projection_records(
    *,
    store: SimpleEditStore,
    records: tuple[PreparedProjection, ...],
) -> dict[str, list[str]]:
    """Apply each prepared candidate transition and collect outcome ids."""

    newly: dict[str, list[str]] = {
        status: []
        for status in ("accepted", "simulation_failed", "prompt_failed")
    }
    for record in records:
        validated = record.validated
        store.commit_projection_transition(record.transition)
        newly[record.transition.outcome].append(validated.candidate_id)
    return newly


def project(
    *,
    output_root: Path | str,
    prompt_builder: Path | str,
) -> dict[str, Any]:
    """Finalize the active workspace using GarmentCode's filter decisions."""

    store = SimpleEditStore.open(output_root)
    manifest = store.run_manifest
    _assert_current_run_contract(manifest)
    slots = _slot_map(manifest)
    states_by_slot = {
        slot_id: store.load_slot_state(slot_id)
        for slot_id in slots
    }
    workspace = load_simulation_workspace(store.root)
    report_path = workspace.root / "random_body" / "filter_fail_report_full.yaml"
    projection = load_filter_projection(workspace, report_path)
    print(
        f"[semantic-edit-project] {store.batch_id}: "
        f"candidates={len(projection.results)}",
        flush=True,
    )
    report_bytes = report_path.read_bytes()
    properties_path = (
        workspace.root
        / "random_body"
        / "dataset_properties_random_body.yaml"
    )
    if properties_path.is_symlink() or not properties_path.is_file():
        raise SemanticEditError("GarmentCode simulation properties are missing")
    properties = load_yaml_bytes(properties_path.read_bytes())
    if properties.get("frozen") is not True:
        raise SemanticEditError(
            "GarmentCode simulation has not reached its frozen terminal state"
        )

    validated = _validate_projection_records(
        results=projection.results,
        store=store,
        states_by_slot=states_by_slot,
        simulation_id=projection.simulation_id,
    )
    builder = _load_projection_builder(
        records=validated,
        manifest=manifest,
        prompt_builder=prompt_builder,
    )
    prepared = _prepare_projection_records(
        records=validated,
        store=store,
        workspace_root=workspace.root,
        builder=builder,
    )
    accepted_ids, pairs_bytes, batch_bytes, body_bytes = (
        _prepare_projection_summary(
            store=store,
            manifest=manifest,
            states_by_slot=states_by_slot,
            records=prepared,
            simulation_properties=properties,
        )
    )

    # First persistent operation starts here.
    store.archive_filter_report(projection.simulation_id, report_bytes)
    newly = _commit_projection_records(
        store=store,
        records=prepared,
    )
    accepted_path = store.write_accepted_pairs(pairs_bytes)
    store.write_final_dataset_properties(batch_bytes, body_bytes)
    remove_simulation_workspace(store.root)
    return {
        "format": RUN_FORMAT,
        "output_root": store.root.as_posix(),
        "batch_id": store.batch_id,
        "simulation_id": projection.simulation_id,
        "simulation_candidate_count": len(projection.results),
        "newly_accepted_count": len(newly["accepted"]),
        "newly_accepted_ids": newly["accepted"],
        "newly_failed_count": len(newly["simulation_failed"]),
        "newly_failed_ids": newly["simulation_failed"],
        "newly_prompt_failed_count": len(newly["prompt_failed"]),
        "newly_prompt_failed_ids": newly["prompt_failed"],
        "accepted_pair_count": len(accepted_ids),
        "remaining_source_count": len(slots) - len(accepted_ids),
        "accepted_pairs_path": accepted_path.as_posix(),
        "final_batch_root": store.batch_root.as_posix(),
    }
