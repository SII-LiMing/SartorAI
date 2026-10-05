"""Plan, generate, project, and status lifecycle behavior."""

from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from editing.canonical_io import (
    canonical_json_bytes,
    dump_yaml_document,
    load_json_bytes,
    sha256_bytes,
)
from editing.config import config_digest
from editing.models import Operation
import editing.semantic_edit.generation as generation
import editing.semantic_edit.construction as construction
import editing.semantic_edit.pipeline as pipeline
import editing.semantic_edit.projection as projection
import editing.semantic_edit.registry as registry_module
import editing.semantic_edit.store as store_module
from editing.semantic_edit.instructions import (
    render_deterministic_specific_instruction,
    validate_instruction_metadata,
)
from editing.semantic_edit.registry import load_semantic_config
from editing.models import (
    PlannedOperation,
    SemanticState,
    SemanticTransition,
    TopologyKind,
    canonical_operation_signature,
)
from editing.semantic_edit.prompt import PromptError
from editing.semantic_edit.planning import Capability
from editing.semantic_edit.store import (
    PAIR_FORMAT,
    PLAN_FORMAT,
    RUN_FORMAT,
    SimpleEditStore,
    candidate_artifact_names,
    load_split_selection,
    selected_slot_ids,
)
from editing.simulation_workspace import prepare_simulation_workspace


SOURCE_ID = "rand_TEST123"
SOURCE_KEY = f"batch_0/random_body/data/{SOURCE_ID}"
SLOT_ID = "slot_000"


def test_pipeline_exports_only_lifecycle_entry_points() -> None:
    assert pipeline.__all__ == ("plan", "generate", "project", "status")


def _primary_planned_operation() -> PlannedOperation:
    rule_id = "shirt_length"
    path = "design.fixture.value"
    transition = SemanticTransition(
        rule_id=rule_id,
        scope="garment",
        channels=("garment",),
        feature_phrase="the shirt",
        source_state=SemanticState(
            rule_id=rule_id,
            state_id="short",
            phrase="short",
            attributes={},
        ),
        target_state=SemanticState(
            rule_id=rule_id,
            state_id="long",
            phrase="long",
            attributes={},
        ),
        direction="increase",
    )
    operation = Operation(
        rule_id=rule_id,
        semantic_group=rule_id,
        scope="garment",
        direction="increase",
        magnitude="specific",
        structural_level=0,
        reads=(path,),
        writes=(path,),
        conflicts=(),
        prerequisites=(),
        edit_mode="replace",
        effect_metric_id="semantic-state",
        effect_contract_id="state-change",
        before={path: 1.0},
        after={path: 2.0},
        requested_magnitude="specific",
        realized_magnitude="specific",
        instruction_tokens={},
    )
    return PlannedOperation(
        operation=operation,
        transitions=(transition,),
        topology_kind=TopologyKind.NONE,
        applicability_evidence={"applicable": True},
        conflict_evidence={"conflict_free": True},
    )


PRIMARY_PLANNED = _primary_planned_operation()
PRIMARY_SIGNATURE = canonical_operation_signature(PRIMARY_PLANNED)
PRIMARY_INSTRUCTION = render_deterministic_specific_instruction(
    PRIMARY_PLANNED,
    source_key=SOURCE_KEY,
    expected_operation_signature=PRIMARY_SIGNATURE,
)


def _alternate_planned_operation(
    rule_id: str,
    direction: str,
    source_state_id: str,
    target_state_id: str,
    after: float,
) -> PlannedOperation:
    operation = replace(
        PRIMARY_PLANNED.operation,
        rule_id=rule_id,
        semantic_group=rule_id,
        direction=direction,
        after={"design.fixture.value": after},
    )
    transition = replace(
        PRIMARY_PLANNED.transitions[0],
        rule_id=rule_id,
        direction=direction,
        feature_phrase={
            "sleeve_length": "the sleeves",
            "pants_length": "the trousers",
        }[rule_id],
        source_state=replace(
            PRIMARY_PLANNED.transitions[0].source_state,
            rule_id=rule_id,
            state_id=source_state_id,
            phrase=source_state_id,
        ),
        target_state=replace(
            PRIMARY_PLANNED.transitions[0].target_state,
            rule_id=rule_id,
            state_id=target_state_id,
            phrase=target_state_id,
        ),
    )
    return replace(
        PRIMARY_PLANNED,
        operation=operation,
        transitions=(transition,),
    )


FALLBACK_PLANNED = _alternate_planned_operation(
    "sleeve_length", "decrease", "long", "short", 0.5
)
THIRD_PLANNED = _alternate_planned_operation(
    "pants_length", "increase", "short", "long", 3.0
)
FALLBACK_SIGNATURE = canonical_operation_signature(FALLBACK_PLANNED)
THIRD_SIGNATURE = canonical_operation_signature(THIRD_PLANNED)
CANDIDATE_ID = f"{SOURCE_ID}__edit_000"
SOURCE_DESIGN = b"design: {fixture: {value: 1.0}}\n"
SOURCE_BODY = b"body: {height: 1.7}\n"
def _fixture_specification(height: float) -> dict[str, Any]:
    panel = {
        "vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "edges": [{"endpoints": [0, 1]}, {"endpoints": [1, 2]}, {"endpoints": [2, 0]}],
        "translation": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
    }
    panels = {"front": deepcopy(panel), "back": deepcopy(panel)}
    panels["front"]["vertices"][2][1] = height
    return {
        "pattern": {
            "panels": panels, "panel_order": ["front", "back"],
            "stitches": [[{"panel": "front", "edge": 0}, {"panel": "back", "edge": 0}]],
        },
        "properties": {"units_in_meter": 100}, "parameters": {}, "parameter_order": [],
    }


SOURCE_SPECIFICATION = canonical_json_bytes(_fixture_specification(1.0))
TARGET_DESIGN = b"design: {fixture: {value: 2.0}}\n"
TARGET_SPECIFICATION = canonical_json_bytes(_fixture_specification(2.0))
PROMPT_BYTES = b'{"sample_prompt":"target"}\n'
PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture"
SVG_BYTES = b"<svg></svg>\n"
EFFECT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "semantic_edit.yaml"
)
EFFECT_CONFIG_BYTES = EFFECT_CONFIG_PATH.read_bytes()
EFFECT_CONFIG_HASH = config_digest(load_semantic_config(EFFECT_CONFIG_PATH))


def _capability(
    rule_id: str,
    direction: str,
    source_state_id: str,
    target_state_id: str,
) -> dict[str, str]:
    return {
        "rule_id": rule_id,
        "direction": direction,
        "source_state_id": source_state_id,
        "target_state_id": target_state_id,
    }


def _candidate(
    *,
    signature: str,
    capability: dict[str, str],
) -> dict[str, Any]:
    return {
        "operation_signature": signature,
        "planned_operation_payload": {
            PRIMARY_SIGNATURE: PRIMARY_PLANNED,
            FALLBACK_SIGNATURE: FALLBACK_PLANNED,
            THIRD_SIGNATURE: THIRD_PLANNED,
        }[signature].to_payload(),
        "capability": capability,
    }


def _manifest(_builder_path: Path) -> dict[str, Any]:
    candidates = [
        _candidate(
            signature=PRIMARY_SIGNATURE,
            capability=_capability("shirt_length", "increase", "short", "long"),
        ),
        _candidate(
            signature=FALLBACK_SIGNATURE,
            capability=_capability("sleeve_length", "decrease", "long", "short"),
        ),
        _candidate(
            signature=THIRD_SIGNATURE,
            capability=_capability("pants_length", "increase", "short", "long"),
        ),
    ]
    return {
        "format": RUN_FORMAT,
        "panel_geometry_policy": "preserve_source_unaffected_panels",
        "max_physical_attempts_per_source": (
            store_module.MAX_PHYSICAL_SIMULATION_ATTEMPTS_PER_SOURCE
        ),
        "candidate_mode": "registry_planned",
        "batch_id": "batch_0",
        "input_root": "/fixture/input",
        "analysis_root": "/fixture/analysis",
        "prompt_builder_sha256": (generation.APPROVED_PROMPT_BUILDER_SHA256),
        "effect_config_path": EFFECT_CONFIG_PATH.as_posix(),
        "effect_config_sha256": sha256_bytes(EFFECT_CONFIG_BYTES),
        "effect_config_hash": EFFECT_CONFIG_HASH,
        "seed": "42",
        "slots": [
            {
                "slot_id": SLOT_ID,
                "source_key": SOURCE_KEY,
                "source_sample_id": SOURCE_ID,
                "primary_operation_signature": PRIMARY_SIGNATURE,
                "candidates": candidates,
            }
        ],
    }


def _store(tmp_path: Path) -> tuple[SimpleEditStore, Path, dict[str, Any]]:
    builder_path = tmp_path / "build_prompts.py"
    builder_path.write_bytes(b"# protected fixture\n")
    manifest = _manifest(builder_path)
    store = SimpleEditStore.create(tmp_path / "output", manifest)
    store.snapshot_source(
        SOURCE_ID,
        design_bytes=SOURCE_DESIGN,
        body_bytes=SOURCE_BODY,
        specification_bytes=SOURCE_SPECIFICATION,
    )
    store.initialize_slot_state(
        slot_id=SLOT_ID,
        source_key=SOURCE_KEY,
        source_sample_id=SOURCE_ID,
    )
    return store, builder_path, manifest


def _metadata(
    *,
    candidate_id: str = CANDIDATE_ID,
    signature: str = PRIMARY_SIGNATURE,
    attempt_index: int = 0,
) -> dict[str, Any]:
    capability = _capability("shirt_length", "increase", "short", "long")
    return {
        "format": RUN_FORMAT,
        "candidate_id": candidate_id,
        "slot_id": SLOT_ID,
        "source_key": SOURCE_KEY,
        "source_sample_id": SOURCE_ID,
        "attempt_index": attempt_index,
        "operation_count": 1,
        "operation_signature": signature,
        "operation": PRIMARY_PLANNED.to_payload(),
        "capability": capability,
        "specific_instruction": PRIMARY_INSTRUCTION.text,
        "canonical_instruction": PRIMARY_INSTRUCTION.canonical_text,
        "instruction_provenance": PRIMARY_INSTRUCTION.provenance_payload(),
        "target_design_hash": "sha256:" + "c" * 64,
        "target_pattern_hash": "sha256:" + "d" * 64,
        "target_specification_sha256": "sha256:" + "e" * 64,
        "construction_gates": {
            "target_construction": "pass",
            "panel_self_intersection": "pass",
            "pattern_data_complete_and_finite": "pass",
            "edited_parameter_changed": "pass",
            "target_pattern_render": "pass",
        },
    }


def _artifacts(
    *,
    candidate_id: str = CANDIDATE_ID,
) -> dict[str, bytes]:
    return {
        "design_params.yaml": TARGET_DESIGN,
        "body_measurements.yaml": SOURCE_BODY,
        f"{candidate_id}_specification.json": TARGET_SPECIFICATION,
        f"{candidate_id}_pattern.png": PNG_BYTES,
        f"{candidate_id}_pattern.svg": SVG_BYTES,
    }


def _patch_candidate_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        generation,
        "load_design_yaml_bytes",
        lambda payload: {"fixture": {"value": 1.0}},
    )
    monkeypatch.setattr(
        generation,
        "load_body_yaml_bytes",
        lambda payload: {"bust": 92.0},
    )
    monkeypatch.setattr(
        generation,
        "serialize_and_reload_design",
        lambda design: SimpleNamespace(
            design=design,
            yaml_bytes=TARGET_DESIGN,
            canonical_hash="sha256:" + "c" * 64,
        ),
    )
    monkeypatch.setattr(
        construction,
        "build_target_pattern",
        lambda design, body, name: SimpleNamespace(
            success=True,
            specification=_fixture_specification(design["fixture"]["value"]),
            specification_json_bytes=TARGET_SPECIFICATION,
            specification_sha256="sha256:" + "e" * 64,
            failure_code=None,
            failure_message=None,
        ),
    )
    monkeypatch.setattr(
        generation,
        "render_pattern",
        lambda specification, name, output_root: SimpleNamespace(
            png_bytes=PNG_BYTES,
            svg_bytes=SVG_BYTES,
        ),
    )
    monkeypatch.setattr(
        generation,
        "semantic_pattern_hash",
        lambda specification: "sha256:" + "d" * 64,
    )


def _effect_registry(validate: Any = None) -> dict[str, Any]:
    if validate is None:
        validate = lambda **_kwargs: SimpleNamespace(
            passed=True,
            code="effect/passed",
            stage="effect",
            evidence={},
        )
    def apply(design: dict[str, Any], planned: PlannedOperation) -> dict[str, Any]:
        return {
            "fixture": {
                "value": planned.operation.after["design.fixture.value"]
            }
        }

    return {
        rule_id: SimpleNamespace(id=rule_id, apply=apply, validate=validate)
        for rule_id in ("shirt_length", "sleeve_length", "pants_length")
    }


def test_cached_operation_uses_rule_apply_outside_source_enum_range() -> None:
    path = "design.collar.f_collar.v"
    operation = replace(
        PRIMARY_PLANNED.operation,
        rule_id="neckline_type",
        semantic_group="collar",
        reads=(path,),
        writes=(path,),
        before={path: "Bezier2NeckHalf"},
        after={path: "CircleNeckHalf"},
    )
    transition = replace(
        PRIMARY_PLANNED.transitions[0],
        rule_id="neckline_type",
        source_state=replace(
            PRIMARY_PLANNED.transitions[0].source_state,
            rule_id="neckline_type",
            state_id="Bezier2NeckHalf",
        ),
        target_state=replace(
            PRIMARY_PLANNED.transitions[0].target_state,
            rule_id="neckline_type",
            state_id="CircleNeckHalf",
        ),
    )
    planned = replace(
        PRIMARY_PLANNED,
        operation=operation,
        transitions=(transition,),
    )
    source = {
        "collar": {
            "f_collar": {
                "v": "Bezier2NeckHalf",
                "range": ["Bezier2NeckHalf"],
                "type": "select",
            }
        }
    }

    def apply(design: dict[str, Any], _planned: PlannedOperation) -> dict[str, Any]:
        design["collar"]["f_collar"]["v"] = "CircleNeckHalf"
        return design

    edited, decoded, selected = generation._apply_single_cached_operation(
        source,
        planned.to_payload(),
        {
            "neckline_type": SimpleNamespace(
                id="neckline_type",
                apply=apply,
            )
        },
    )

    assert decoded.operation.rule_id == "neckline_type"
    assert selected.id == "neckline_type"
    assert edited["collar"]["f_collar"]["v"] == "CircleNeckHalf"
    assert edited["collar"]["f_collar"]["range"] == ["Bezier2NeckHalf"]
    assert source["collar"]["f_collar"]["v"] == "Bezier2NeckHalf"


def test_apply_value_error_is_recorded_and_next_candidate_is_selectable() -> None:
    state = {"attempts": [], "accepted_target_id": None}
    saved: list[dict[str, Any]] = []
    store = SimpleNamespace(
        source_snapshot=lambda _sample_id: (SOURCE_DESIGN, SOURCE_BODY),
        save_slot_state=lambda value: saved.append(value),
    )
    slot = _manifest(Path("unused"))["slots"][0]
    registry = _effect_registry()

    def fail_apply(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("invalid candidate value")

    registry["shirt_length"].apply = fail_apply
    passed, failure = generation._attempt_candidate(
        store=store,
        slot=slot,
        state=state,
        candidate=slot["candidates"][0],
        effect_config=load_semantic_config(EFFECT_CONFIG_PATH),
        registry=registry,
    )

    assert passed is False
    assert failure["status"] == "generation_failed"
    assert failure["failure_code"] == "operation_apply_failed"
    assert "ValueError: invalid candidate value" in failure["failure_message"]
    assert saved and saved[-1]["attempts"] == [failure]
    next_candidate = generation.select_candidate(
        slot,
        attempted_signatures={failure["operation_signature"]},
        rule_counts={},
        capability_counts={},
    )
    assert next_candidate is not None
    assert next_candidate["operation_signature"] != failure["operation_signature"]


def test_unexpected_candidate_assertion_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"attempts": [], "accepted_target_id": None}
    saved: list[dict[str, Any]] = []
    store = SimpleNamespace(
        source_snapshot=lambda _sample_id: (SOURCE_DESIGN, SOURCE_BODY),
        save_slot_state=lambda value: saved.append(value),
    )
    slot = _manifest(Path("unused"))["slots"][0]

    def fail_construction(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("degenerate edge")

    monkeypatch.setattr(
        construction,
        "build_target_pattern",
        fail_construction,
    )
    passed, failure = generation._attempt_candidate(
        store=store,
        slot=slot,
        state=state,
        candidate=slot["candidates"][0],
        effect_config=load_semantic_config(EFFECT_CONFIG_PATH),
        registry=_effect_registry(),
    )

    assert passed is False
    assert failure["status"] == "generation_failed"
    assert failure["failure_code"] == "generation_exception"
    assert failure["failure_message"] == "AssertionError: degenerate edge"
    assert saved and saved[-1]["attempts"] == [failure]


def test_candidate_zero_division_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"attempts": [], "accepted_target_id": None}
    saved: list[dict[str, Any]] = []
    store = SimpleNamespace(
        source_snapshot=lambda _sample_id: (SOURCE_DESIGN, SOURCE_BODY),
        save_slot_state=lambda value: saved.append(value),
    )
    slot = _manifest(Path("unused"))["slots"][0]

    def fail_construction(*_args: Any, **_kwargs: Any) -> None:
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(
        construction,
        "build_target_pattern",
        fail_construction,
    )
    passed, failure = generation._attempt_candidate(
        store=store,
        slot=slot,
        state=state,
        candidate=slot["candidates"][0],
        effect_config=load_semantic_config(EFFECT_CONFIG_PATH),
        registry=_effect_registry(),
    )

    assert passed is False
    assert failure["status"] == "generation_failed"
    assert failure["failure_code"] == "generation_exception"
    assert failure["failure_message"] == (
        "ZeroDivisionError: float division by zero"
    )
    assert saved and saved[-1]["attempts"] == [failure]


def _write_filter_report(
    workspace_root: Path,
    *,
    status: str,
    candidate_id: str = CANDIDATE_ID,
) -> Path:
    failed = [candidate_id] if status == "fail" else []
    passed = [candidate_id] if status == "pass" else []
    report = {
        "schema_version": 1,
        "report_type": "filter_fail_report_full",
        "gcd_dir": workspace_root.as_posix(),
        "body_type": "random_body",
        "body_root": (workspace_root / "random_body").as_posix(),
        "dataset_properties": (
            workspace_root / "random_body" / "dataset_properties_random_body.yaml"
        ).as_posix(),
        "parameters": {"no_delete": True},
        "summary": {},
        "case_names": [candidate_id],
        "passed_cases": passed,
        "final_failed_cases": failed,
        "final_fail_reasons": ({candidate_id: ["self_intersection"]} if failed else {}),
    }
    path = workspace_root / "random_body" / "filter_fail_report_full.yaml"
    path.write_bytes(dump_yaml_document(report))
    return path


def _stage_candidate(
    store: SimpleEditStore,
    *,
    status: str,
    frozen: bool = True,
) -> dict[str, Any]:
    metadata = _metadata()
    store.publish_candidate(
        CANDIDATE_ID,
        artifacts=_artifacts(),
        metadata=metadata,
    )
    workspace = prepare_simulation_workspace(
        store.root,
        [
            {
                "candidate_id": CANDIDATE_ID,
                "target_id": CANDIDATE_ID,
                "target_path": (f"{store.batch_id}/random_body/data/{CANDIDATE_ID}"),
                "target_simulation_status": "pending",
            }
        ],
        global_seed="42",
        batch_id=store.batch_id,
    )
    product = workspace.root / "random_body" / CANDIDATE_ID
    product.mkdir()
    (product / f"{CANDIDATE_ID}_sim.ply").write_bytes(b"ply\n")
    (product / f"{CANDIDATE_ID}_pattern.png").write_bytes(PNG_BYTES)
    (product / f"{CANDIDATE_ID}_pattern.svg").write_bytes(SVG_BYTES)
    (
        workspace.root / "random_body" / "dataset_properties_random_body.yaml"
    ).write_bytes(
        dump_yaml_document(
            {
                "frozen": frozen,
                "generator": {"stats": {"panel_count": {CANDIDATE_ID: 1}}},
            }
        )
    )
    _write_filter_report(workspace.root, status=status)
    state = store.load_slot_state(SLOT_ID)
    capability = metadata["capability"]
    state["attempts"].append(
        {
            "attempt_index": 0,
            "candidate_id": CANDIDATE_ID,
            "operation_signature": PRIMARY_SIGNATURE,
            "rule_id": capability["rule_id"],
            "capability_key": Capability.from_payload(capability).count_key,
            "capability": capability,
            "simulation_id": workspace.simulation_id,
            "physical_attempt_number": 1,
            "status": "in_simulation",
            "failure_code": None,
            "failure_message": None,
        }
    )
    store.save_slot_state(state)
    return state


def test_public_generation_snapshots_flattened_prefixed_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    analysis_root = tmp_path / "analysis"
    source_key = f"batch_0/random_body/{SOURCE_ID}"
    source_root = input_root.joinpath(*source_key.split("/"))
    source_root.mkdir(parents=True)
    analysis_root.mkdir()
    (source_root / f"{SOURCE_ID}_design_params.yaml").write_bytes(
        SOURCE_DESIGN
    )
    (source_root / f"{SOURCE_ID}_body_measurements.yaml").write_bytes(
        SOURCE_BODY
    )
    (source_root / f"{SOURCE_ID}_specification.json").write_bytes(
        SOURCE_SPECIFICATION
    )
    builder_path = tmp_path / "build_prompts.py"
    builder_path.write_bytes(b"# fixture\n")
    manifest = _manifest(builder_path)
    manifest["slots"][0]["source_key"] = source_key
    (analysis_root / "pilot_eligibility_report.json").write_bytes(
        canonical_json_bytes(
            {
                "format": PLAN_FORMAT,
                "batch_id": "batch_0",
                "config_path": EFFECT_CONFIG_PATH.as_posix(),
                "config_file_sha256": sha256_bytes(EFFECT_CONFIG_BYTES),
                "effective_config_hash": EFFECT_CONFIG_HASH,
                "prompt_builder_sha256": (
                    generation.APPROVED_PROMPT_BUILDER_SHA256
                ),
            }
        )
    )
    monkeypatch.setattr(
        generation,
        "build_run_manifest",
        lambda **_kwargs: {"slots": manifest["slots"]},
    )
    monkeypatch.setattr(
        registry_module,
        "build_semantic_registry",
        lambda _config: _effect_registry(),
    )
    _patch_candidate_build(monkeypatch)

    output_root = tmp_path / "output"
    result = pipeline.generate(
        input_root=input_root,
        analysis_root=analysis_root,
        output_root=output_root,
        seed="42",
    )

    store = SimpleEditStore.open(output_root)
    assert result["new_candidate_count"] == 1
    assert store.source_snapshot(SOURCE_ID) == (SOURCE_DESIGN, SOURCE_BODY)
    assert store.source_specification(SOURCE_ID) == SOURCE_SPECIFICATION
    assert store.candidate_metadata(CANDIDATE_ID)["source_key"] == source_key


def test_public_generation_retries_three_different_physical_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    analysis_root = tmp_path / "analysis"
    source_root = input_root.joinpath(*SOURCE_KEY.split("/"))
    source_root.mkdir(parents=True)
    analysis_root.mkdir()
    (source_root / "design_params.yaml").write_bytes(SOURCE_DESIGN)
    (source_root / "body_measurements.yaml").write_bytes(SOURCE_BODY)
    (source_root / f"{SOURCE_ID}_specification.json").write_bytes(SOURCE_SPECIFICATION)
    builder_path = tmp_path / "build_prompts.py"
    builder_path.write_bytes(b"# fixture\n")
    manifest = _manifest(builder_path)
    (analysis_root / "pilot_eligibility_report.json").write_bytes(
        canonical_json_bytes(
            {
                "format": PLAN_FORMAT,
                "batch_id": "batch_0",
                "config_path": EFFECT_CONFIG_PATH.as_posix(),
                "config_file_sha256": sha256_bytes(EFFECT_CONFIG_BYTES),
                "effective_config_hash": EFFECT_CONFIG_HASH,
                "prompt_builder_sha256": (generation.APPROVED_PROMPT_BUILDER_SHA256),
            }
        )
    )
    monkeypatch.setattr(
        generation,
        "build_run_manifest",
        lambda **_kwargs: {"slots": manifest["slots"]},
    )
    monkeypatch.setattr(
        registry_module,
        "build_semantic_registry",
        lambda _config: _effect_registry(),
    )
    monkeypatch.setattr(
        projection.PromptBuilder,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed simulations must not load prompts")
        ),
    )
    _patch_candidate_build(monkeypatch)

    output_root = tmp_path / "output"

    def generate() -> dict[str, Any]:
        return pipeline.generate(
            input_root=input_root,
            analysis_root=analysis_root,
            output_root=output_root,
            seed="42",
        )

    def fail_active_candidate(candidate_id: str) -> None:
        store = SimpleEditStore.open(output_root)
        target = store.data_root / candidate_id
        assert frozenset(path.name for path in target.iterdir()) == (
            candidate_artifact_names(candidate_id)
        )
        assert not any("prompt" in path.name for path in target.iterdir())

        workspace_root = store.simulation_workspace_root
        product = workspace_root / "random_body" / candidate_id
        product.mkdir()
        (product / f"{candidate_id}_sim.ply").write_bytes(b"ply\n")
        (
            workspace_root / "random_body" / "dataset_properties_random_body.yaml"
        ).write_bytes(
            dump_yaml_document(
                {
                    "frozen": True,
                    "generator": {"stats": {"panel_count": {candidate_id: 1}}},
                }
            )
        )
        _write_filter_report(
            workspace_root,
            status="fail",
            candidate_id=candidate_id,
        )
        result = pipeline.project(
            output_root=store.root,
            prompt_builder=builder_path,
        )
        assert result["newly_failed_ids"] == [candidate_id]

    expected_signatures = (
        PRIMARY_SIGNATURE,
        FALLBACK_SIGNATURE,
        THIRD_SIGNATURE,
    )
    candidate_ids: list[str] = []
    for attempt_index, signature in enumerate(expected_signatures):
        result = generate()
        candidate_id = f"{SOURCE_ID}__edit_{attempt_index:03d}"
        candidate_ids.append(candidate_id)
        assert result["new_candidate_ids"] == [candidate_id]
        metadata = SimpleEditStore.open(output_root).candidate_metadata(candidate_id)
        assert metadata["operation_signature"] == signature
        fail_active_candidate(candidate_id)

    final = generate()
    state = SimpleEditStore.open(output_root).load_slot_state(SLOT_ID)
    assert final["new_candidate_count"] == 0
    assert final["physical_limit_reached_slot_ids"] == [SLOT_ID]
    assert [row["candidate_id"] for row in state["attempts"]] == candidate_ids
    assert {row["status"] for row in state["attempts"]} == {"simulation_failed"}


def test_project_rejects_a_nonterminal_garmentcode_simulation(
    tmp_path: Path,
) -> None:
    store, builder_path, manifest = _store(tmp_path)
    _stage_candidate(store, status="pass", frozen=False)

    with pytest.raises(
        store_module.SemanticEditError,
        match="frozen terminal state",
    ):
        pipeline.project(
            output_root=store.root,
            prompt_builder=builder_path,
        )

    assert store.simulation_workspace_root.is_dir()


def test_prompt_render_failure_marks_only_candidate_prompt_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, builder_path, _ = _store(tmp_path)
    _stage_candidate(store, status="pass")

    def fail_render(*_args: Any, **_kwargs: Any) -> None:
        raise PromptError("parser failed")

    monkeypatch.setattr(
        projection.PromptBuilder,
        "load",
        lambda *args, **kwargs: SimpleNamespace(render=fail_render),
    )
    result = pipeline.project(
        output_root=store.root,
        prompt_builder=builder_path,
    )

    state = store.load_slot_state(SLOT_ID)
    assert result["newly_prompt_failed_ids"] == [CANDIDATE_ID]
    assert result["newly_accepted_ids"] == []
    assert state["attempts"][0]["status"] == "prompt_failed"
    assert state["attempts"][0]["failure_code"] == "target_prompt_failed"
    assert (store.rejected_root / CANDIDATE_ID).is_dir()


def test_physical_pass_finalizes_flat_garmentcode_layout_once(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    store, builder_path, manifest = _store(tmp_path)
    _stage_candidate(store, status="pass")
    load_calls: list[tuple[Path, dict[str, Any]]] = []
    render_calls: list[tuple[bytes, bytes]] = []

    class FakeBuilder:
        def render(
            self,
            design_bytes: bytes,
            body_bytes: bytes,
            *,
            output_root: Path,
        ) -> Any:
            assert output_root == store.root
            render_calls.append((design_bytes, body_bytes))
            return SimpleNamespace(
                json_bytes=PROMPT_BYTES,
                builder_sha256=generation.APPROVED_PROMPT_BUILDER_SHA256,
            )

    def fake_load(path: Path, **kwargs: Any) -> FakeBuilder:
        load_calls.append((path, kwargs))
        return FakeBuilder()

    monkeypatch.setattr(projection.PromptBuilder, "load", fake_load)
    first = pipeline.project(
        output_root=store.root,
        prompt_builder=builder_path,
    )

    assert first["format"] == RUN_FORMAT

    target = store.data_root / CANDIDATE_ID
    product = store.body_root / CANDIDATE_ID
    assert first["newly_accepted_ids"] == [CANDIDATE_ID]
    assert frozenset(path.name for path in target.iterdir()) == frozenset(
        {
            "design_params.yaml",
            "body_measurements.yaml",
            f"{CANDIDATE_ID}_specification.json",
            f"{CANDIDATE_ID}_pattern.png",
            f"{CANDIDATE_ID}_pattern.svg",
            f"{CANDIDATE_ID}_prompts.json",
            f"{CANDIDATE_ID}_edit_metadata.json",
        }
    )
    assert product.is_dir()
    assert (product / f"{CANDIDATE_ID}_sim.ply").is_file()
    assert (product / f"{CANDIDATE_ID}_prompts.json").read_bytes() == (PROMPT_BYTES)
    assert (product / f"{CANDIDATE_ID}_pattern.png").read_bytes() == PNG_BYTES
    assert (product / f"{CANDIDATE_ID}_pattern.svg").read_bytes() == SVG_BYTES
    assert (
        store.body_root / "patterns_vis" / f"{CANDIDATE_ID}_pattern.png"
    ).read_bytes() == PNG_BYTES
    assert (store.batch_root / "dataset_properties.yaml").is_file()
    assert (store.body_root / "dataset_properties_random_body.yaml").is_file()
    assert not (store.root / "simulation_rounds").exists()
    assert not store.simulation_workspace_root.exists()
    state = store.load_slot_state(SLOT_ID)
    assert state["format"] == RUN_FORMAT
    assert state["accepted_target_id"] == CANDIDATE_ID
    assert state["attempts"][0]["status"] == "accepted"
    accepted = (store.batch_root / "accepted_pairs.jsonl").read_text()
    assert load_json_bytes(accepted.encode())["format"] == PAIR_FORMAT
    pair = load_json_bytes((target / f"{CANDIDATE_ID}_edit_metadata.json").read_bytes())
    replayed = validate_instruction_metadata(pair)
    assert pair["specific_instruction"] == replayed.text
    assert pair["canonical_instruction"] == replayed.canonical_text
    assert pair["instruction_provenance"] == replayed.provenance_payload()

    assert load_calls == [
        (
            builder_path.resolve(),
            {"expected_sha256": manifest["prompt_builder_sha256"]},
        )
    ]
    assert len(render_calls) == 1


def test_public_results_and_status_use_current_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "input"
    analysis_root = tmp_path / "analysis"
    source_root = input_root.joinpath(*SOURCE_KEY.split("/"))
    source_root.mkdir(parents=True)
    analysis_root.mkdir()
    (source_root / "design_params.yaml").write_bytes(SOURCE_DESIGN)
    (source_root / "body_measurements.yaml").write_bytes(SOURCE_BODY)
    (source_root / f"{SOURCE_ID}_specification.json").write_bytes(SOURCE_SPECIFICATION)
    builder_path = tmp_path / "build_prompts.py"
    builder_path.write_bytes(b"# protected fixture\n")
    fixture_manifest = _manifest(builder_path)
    (analysis_root / "pilot_eligibility_report.json").write_bytes(
        canonical_json_bytes(
            {
                "format": PLAN_FORMAT,
                "batch_id": "batch_0",
                "config_path": EFFECT_CONFIG_PATH.as_posix(),
                "config_file_sha256": sha256_bytes(EFFECT_CONFIG_BYTES),
                "effective_config_hash": EFFECT_CONFIG_HASH,
                "prompt_builder_sha256": (generation.APPROVED_PROMPT_BUILDER_SHA256),
            }
        )
    )
    output_root = tmp_path / "output"
    monkeypatch.setattr(
        generation,
        "build_run_manifest",
        lambda **kwargs: {"slots": fixture_manifest["slots"]},
    )
    monkeypatch.setattr(
        generation,
        "select_candidate",
        lambda *args, **kwargs: None,
    )
    generated = pipeline.generate(
        input_root=input_root,
        analysis_root=analysis_root,
        output_root=output_root,
        seed="42",
    )
    store = SimpleEditStore.open(output_root)

    assert generated["format"] == RUN_FORMAT
    manifest = store.run_manifest
    assert set(manifest) == {
        "analysis_report_path",
        "analysis_root",
        "batch_id",
        "candidate_mode",
        "effect_config_hash",
        "effect_config_path",
        "effect_config_sha256",
        "format",
        "input_root",
        "max_physical_attempts_per_source",
        "panel_geometry_policy",
        "prompt_builder_sha256",
        "seed",
        "slots",
    }
    assert manifest["format"] == RUN_FORMAT
    assert manifest["max_physical_attempts_per_source"] == 3
    report = pipeline.status(store.root)
    assert report == {
        "format": RUN_FORMAT,
        "output_root": store.root.as_posix(),
        "batch_id": store.batch_id,
        "source_count": 1,
        "state_counts": {"not_generated": 1},
        "accepted_rule_distribution": {},
        "max_physical_simulation_attempts_per_source": 3,
        "simulation_workspace": {
            "present": False,
            "active": False,
            "simulation_id": None,
            "candidate_count": 0,
            "candidate_ids": [],
        },
    }


def _write_split(
    path: Path,
    *,
    first: list[Path],
    second: list[Path],
) -> Path:
    path.write_bytes(
        canonical_json_bytes(
            {
                "dataset_0_15": [value.as_posix() for value in first],
                "dataset_0_85": [value.as_posix() for value in second],
            }
        )
    )
    return path


def test_split_selection_validates_and_maps_disjoint_slots(
    tmp_path: Path,
) -> None:
    input_root = (tmp_path / "input").resolve()
    input_root.mkdir()
    first_source = input_root / "batch_0/random_body/rand_FIRST"
    second_source = input_root / "batch_0/random_body/rand_SECOND"
    split_path = _write_split(
        tmp_path / "split.json",
        first=[first_source],
        second=[second_source],
    )
    slots = {
        "slot_000": {"source_key": "batch_0/random_body/rand_FIRST"},
        "slot_001": {"source_key": "batch_0/random_body/rand_SECOND"},
    }

    first = load_split_selection(
        split_path=split_path,
        split_name="dataset_0_15",
        input_root=input_root,
    )
    second = load_split_selection(
        split_path=split_path,
        split_name="dataset_0_85",
        input_root=input_root,
    )

    assert first is not None
    assert second is not None
    assert selected_slot_ids(
        slots,
        input_root=input_root,
        selection=first,
    ) == frozenset({"slot_000"})
    assert selected_slot_ids(
        slots,
        input_root=input_root,
        selection=second,
    ) == frozenset({"slot_001"})

    relative_split = _write_split(
        tmp_path / "relative.json",
        first=[Path("batch_0/random_body/rand_FIRST")],
        second=[],
    )
    with pytest.raises(ValueError, match="normalized absolute path"):
        load_split_selection(
            split_path=relative_split,
            split_name="dataset_0_15",
            input_root=input_root,
        )

    overlapping_split = _write_split(
        tmp_path / "overlapping.json",
        first=[first_source],
        second=[first_source],
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        load_split_selection(
            split_path=overlapping_split,
            split_name="dataset_0_15",
            input_root=input_root,
        )


def test_split_generate_can_skip_then_resume_same_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = (tmp_path / "input").resolve()
    analysis_root = (tmp_path / "analysis").resolve()
    source_key = f"batch_0/random_body/{SOURCE_ID}"
    source_root = input_root.joinpath(*source_key.split("/"))
    source_root.mkdir(parents=True)
    analysis_root.mkdir()
    (source_root / "design_params.yaml").write_bytes(SOURCE_DESIGN)
    (source_root / "body_measurements.yaml").write_bytes(SOURCE_BODY)
    (source_root / f"{SOURCE_ID}_specification.json").write_bytes(
        SOURCE_SPECIFICATION
    )
    fixture_manifest = _manifest(tmp_path / "unused_builder.py")
    fixture_manifest["slots"][0]["source_key"] = source_key
    (analysis_root / "pilot_eligibility_report.json").write_bytes(
        canonical_json_bytes(
            {
                "format": PLAN_FORMAT,
                "batch_id": "batch_0",
                "config_path": EFFECT_CONFIG_PATH.as_posix(),
                "config_file_sha256": sha256_bytes(EFFECT_CONFIG_BYTES),
                "effective_config_hash": EFFECT_CONFIG_HASH,
                "prompt_builder_sha256": (
                    generation.APPROVED_PROMPT_BUILDER_SHA256
                ),
            }
        )
    )
    split_path = _write_split(
        tmp_path / "split.json",
        first=[input_root / "batch_0/random_body/rand_OTHER"],
        second=[source_root],
    )
    monkeypatch.setattr(
        generation,
        "build_run_manifest",
        lambda **_kwargs: {"slots": fixture_manifest["slots"]},
    )
    monkeypatch.setattr(
        registry_module,
        "build_semantic_registry",
        lambda _config: _effect_registry(),
    )
    _patch_candidate_build(monkeypatch)
    output_root = tmp_path / "output"

    skipped = pipeline.generate(
        input_root=input_root,
        analysis_root=analysis_root,
        output_root=output_root,
        seed="42",
        split_path=split_path,
        split_name="dataset_0_15",
    )
    store = SimpleEditStore.open(output_root)
    assert skipped["source_count"] == 1
    assert skipped["selected_source_count"] == 0
    assert skipped["new_candidate_count"] == 0
    assert store.load_slot_state(SLOT_ID)["attempts"] == []
    assert not (store.snapshots_root / SOURCE_ID).exists()
    skipped_status = pipeline.status(
        output_root,
        split_path=split_path,
        split_name="dataset_0_15",
    )
    pending_status = pipeline.status(
        output_root,
        split_path=split_path,
        split_name="dataset_0_85",
    )
    assert skipped_status["source_count"] == 0
    assert skipped_status["state_counts"] == {}
    assert pending_status["source_count"] == 1
    assert pending_status["state_counts"] == {"not_generated": 1}

    resumed = pipeline.generate(
        input_root=input_root,
        analysis_root=analysis_root,
        output_root=output_root,
        seed="42",
        split_path=split_path,
        split_name="dataset_0_85",
    )
    store = SimpleEditStore.open(output_root)
    assert resumed["source_count"] == 1
    assert resumed["selected_source_count"] == 1
    assert resumed["new_candidate_ids"] == [CANDIDATE_ID]
    assert store.source_snapshot(SOURCE_ID) == (SOURCE_DESIGN, SOURCE_BODY)
    assert store.load_slot_state(SLOT_ID)["attempts"][0]["status"] == (
        "in_simulation"
    )
    resumed_status = pipeline.status(
        output_root,
        split_path=split_path,
        split_name="dataset_0_85",
    )
    assert resumed_status["source_count"] == 1
    assert resumed_status["manifest_source_count"] == 1
    assert resumed_status["state_counts"] == {"in_simulation": 1}
