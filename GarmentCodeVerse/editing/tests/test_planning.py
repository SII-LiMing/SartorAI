"""Planning, source selection, and capability assignment behavior."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from editing.canonical_io import (
    canonical_hash,
    canonical_json_bytes,
    dump_yaml_document,
    load_json_bytes,
    sha256_bytes,
)
from editing.models import EffectResult, Operation, Rule, RuleContext
import editing.semantic_edit.prompt as prompt_module
import editing.semantic_edit.prompt_state as prompt_state_module
from editing.semantic_edit.prompt_state import ActivePaths
import editing.semantic_edit.registry as registry_module
import editing.semantic_edit.rules.godet as godet_module
from editing.models import (
    PlannedOperation,
    SemanticState,
    SemanticTransition,
    TopologyKind,
    canonical_operation_signature,
)
from editing.semantic_edit.planning import (
    Capability,
    build_run_manifest,
    plan,
    select_candidate,
)
from editing.semantic_edit.store import PLAN_FORMAT


SHA_A = "sha256:" + "a" * 64
ASYMMETRIC_LEFT_SLEEVE_PATH = "design.left.sleeve.length.v"
ASYMMETRIC_RIGHT_SLEEVE_PATH = "design.sleeve.length.v"
PLANNER_BASELINES = {
    "shirt": {
        "signature": (
            "sha256:3fccd5feb327b1cc60f5123fc8cb5bc7e4fe9a2abfd403c5afe910a0d5b76609"
        ),
        "payload_digest": (
            "sha256:99d9cf1472ab5ea284f912ded4258577ae504672e8d85e826472412e85705059"
        ),
    },
    "sleeve_first": {
        "signature": (
            "sha256:a9f2618d9bc4b3c352b10a9cba52ff7bcd40af744481255684ddf5d6d28eb679"
        ),
        "payload_digest": (
            "sha256:82c415b401efe97172225f06f8aa72cab13c431fd243107c97bb1fbff2081ad2"
        ),
    },
    "sleeve_second": {
        "signature": (
            "sha256:ec359f2abd33320db631ef5ade23cd5e5c1382ec51fd0b80b1beb67dd113a046"
        ),
        "payload_digest": (
            "sha256:342e1febf27a53243285081253c407bec81a4e9cb458651e0b0ca7393224526c"
        ),
    },
}


def _write_source(
    input_root: Path,
    sample_id: str,
    *,
    fixture_value: float = 1.0,
    flattened: bool = False,
    prefixed_yaml: bool = False,
) -> Path:
    body_root = input_root / "batch_0" / "random_body"
    sample_dir = (
        body_root / sample_id
        if flattened
        else body_root / "data" / sample_id
    )
    sample_dir.mkdir(parents=True)
    design_name = (
        f"{sample_id}_design_params.yaml"
        if prefixed_yaml
        else "design_params.yaml"
    )
    body_name = (
        f"{sample_id}_body_measurements.yaml"
        if prefixed_yaml
        else "body_measurements.yaml"
    )
    (sample_dir / design_name).write_bytes(
        dump_yaml_document(
            {
                "design": {
                    "fixture": {"value": fixture_value},
                    "sleeve": {"length": {"v": 0.40}},
                    "left": {"sleeve": {"length": {"v": 0.35}}},
                }
            }
        )
    )
    (sample_dir / body_name).write_bytes(
        dump_yaml_document({"body": {"body_sample": "body_001", "height": 170.0}})
    )
    specification = {
        "pattern": {
            "panels": {
                "front": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    "edges": [
                        {"endpoints": [0, 1]},
                        {"endpoints": [1, 2]},
                        {"endpoints": [2, 0]},
                    ],
                }
            },
            "stitches": [],
            "panel_order": ["front"],
        },
        "parameters": {},
        "parameter_order": [],
        "properties": {"units_in_meter": 100.0},
    }
    (sample_dir / f"{sample_id}_specification.json").write_bytes(
        canonical_json_bytes(specification)
    )
    return sample_dir


def _planned(
    capability: Capability,
    *,
    suffix: str = "fixture",
) -> PlannedOperation:
    path = f"design.fixture.{capability.rule_id}_{suffix}.v"
    operation = Operation(
        rule_id=capability.rule_id,
        semantic_group="test",
        scope="garment",
        direction=capability.direction,
        magnitude="structural",
        structural_level=0,
        reads=(path,),
        writes=(path,),
        conflicts=(),
        prerequisites=("fixture_applicable",),
        edit_mode="enum",
        effect_metric_id="fixture_metric",
        effect_contract_id="fixture_contract",
        before={path: capability.source_state_id},
        after={path: capability.target_state_id},
        relative_change=None,
        requested_magnitude="structural",
        realized_magnitude="structural",
        clamped=False,
        instruction_tokens={"feature": capability.rule_id},
    )
    transition = SemanticTransition(
        rule_id=capability.rule_id,
        scope="garment",
        channels=("garment",),
        feature_phrase=f"the {capability.rule_id.replace('_', ' ')}",
        source_state=SemanticState(
            rule_id=capability.rule_id,
            state_id=capability.source_state_id,
            phrase=capability.source_state_id.replace("-", " "),
            attributes={"fixture": "source"},
        ),
        target_state=SemanticState(
            rule_id=capability.rule_id,
            state_id=capability.target_state_id,
            phrase=capability.target_state_id.replace("-", " "),
            attributes={"fixture": "target"},
        ),
        direction=capability.direction,
    )
    return PlannedOperation(
        operation=operation,
        transitions=(transition,),
        topology_kind=TopologyKind.NONE,
        applicability_evidence={"eligible": True},
        conflict_evidence={"earlier_rule_ids": []},
    )


def _asymmetric_sleeve_planned(
    *,
    variant: str,
    left_target: float,
    right_target: float,
) -> PlannedOperation:
    writes = (ASYMMETRIC_LEFT_SLEEVE_PATH, ASYMMETRIC_RIGHT_SLEEVE_PATH)
    operation = Operation(
        rule_id="sleeve_length",
        semantic_group="sleeve",
        scope="both",
        direction="increase",
        magnitude="significantly",
        structural_level=0,
        reads=writes,
        writes=writes,
        conflicts=(),
        prerequisites=("asymmetric_sleeves_exist",),
        edit_mode="safe_span",
        effect_metric_id="sleeve_length_cm",
        effect_contract_id="topology_preserving",
        before={
            ASYMMETRIC_LEFT_SLEEVE_PATH: 0.35,
            ASYMMETRIC_RIGHT_SLEEVE_PATH: 0.40,
        },
        after={
            ASYMMETRIC_LEFT_SLEEVE_PATH: left_target,
            ASYMMETRIC_RIGHT_SLEEVE_PATH: right_target,
        },
        relative_change=0.5,
        requested_magnitude="significantly",
        realized_magnitude="significantly",
        clamped=False,
        instruction_tokens={"feature": "sleeve length", "variant": variant},
    )
    transition = SemanticTransition(
        rule_id="sleeve_length",
        scope="both",
        channels=("left", "right"),
        feature_phrase="both sleeves",
        source_state=SemanticState(
            rule_id="sleeve_length",
            state_id="short",
            phrase="short",
            attributes={"category": "short"},
        ),
        target_state=SemanticState(
            rule_id="sleeve_length",
            state_id="long",
            phrase="long",
            attributes={"category": "long"},
        ),
        direction="increase",
    )
    return PlannedOperation(
        operation=operation,
        transitions=(transition,),
        topology_kind=TopologyKind.NONE,
        applicability_evidence={"eligible": True, "variant": variant},
        conflict_evidence={"earlier_rule_ids": []},
    )


def _plan_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_root = tmp_path / "input"
    config_path = Path(__file__).resolve().parents[1] / "configs" / "semantic_edit.yaml"
    builder_path = tmp_path / "build_prompts.py"
    builder_path.write_text("# fixture\n", encoding="utf-8")
    return input_root, config_path, builder_path


def _patch_public_plan_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_streams: dict[str, tuple[PlannedOperation, ...]] | None = None,
    candidate_streams_by_source: (
        dict[
            int,
            dict[str, tuple[PlannedOperation, ...]],
        ]
        | None
    ) = None,
) -> None:
    def candidates_for(
        rule_id: str,
    ) -> Callable[[RuleContext], tuple[PlannedOperation, ...]]:
        def candidates(
            context: RuleContext,
        ) -> tuple[PlannedOperation, ...]:
            if candidate_streams_by_source is None:
                return (candidate_streams or {}).get(rule_id, ())
            source_index = int(context.design["fixture"]["value"])
            return candidate_streams_by_source[source_index].get(
                rule_id,
                (),
            )

        return candidates

    def unused_apply(
        _design: dict[str, Any],
        _candidate: Any,
    ) -> dict[str, Any]:
        raise AssertionError("planning must not apply candidates")

    def unused_validate(**_kwargs: Any) -> EffectResult:
        raise AssertionError("planning must not validate candidates")

    registry = {
        rule_id: Rule(
            rule_id,
            candidates_for(rule_id),
            unused_apply,
            unused_validate,
        )
        for rule_id in registry_module.RULE_ORDER
    }
    builder = SimpleNamespace(sha256=SHA_A)
    monkeypatch.setattr(
        registry_module,
        "build_semantic_registry",
        lambda _: registry,
    )
    monkeypatch.setattr(
        prompt_module.PromptBuilder,
        "load",
        staticmethod(lambda _: builder),
    )
    monkeypatch.setattr(
        prompt_state_module.PromptStateClassifier,
        "__call__",
        lambda *_: {},
    )
    monkeypatch.setattr(
        prompt_state_module,
        "resolve_active_paths",
        lambda design: ActivePaths(
            design_hash=canonical_hash(design),
            paths=frozenset(
                {
                    "design.fixture.value",
                    ASYMMETRIC_LEFT_SLEEVE_PATH,
                    ASYMMETRIC_RIGHT_SLEEVE_PATH,
                }
            ),
            concepts={},
            suppressed_paths=frozenset(),
            dispatch={},
        ),
    )
    monkeypatch.setattr(
        godet_module,
        "probe_constructor_facts",
        lambda *_: {},
    )


def _run_plan(
    *,
    input_root: Path,
    analysis_root: Path,
    config_path: Path,
    builder_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = plan(
        input_root=input_root,
        batch_id="batch_0",
        analysis_root=analysis_root,
        config_path=config_path,
        prompt_builder=builder_path,
    )
    report = load_json_bytes(
        (analysis_root / "pilot_eligibility_report.json").read_bytes()
    )
    return result, report


SHIRT_SHORT_LONG = Capability("shirt_length", "increase", "short", "long")
SHIRT_CROPPED_REGULAR = Capability("shirt_length", "increase", "cropped", "regular")
SLEEVE_SHORT_LONG = Capability("sleeve_length", "increase", "short", "long")


EXPECTED_REQUIRED_CAPABILITIES = (
    Capability("shirt_length", "increase", "*", "*"),
    Capability("shirt_length", "decrease", "*", "*"),
    Capability("sleeve_length", "increase", "*", "*"),
    Capability("sleeve_length", "decrease", "*", "*"),
    Capability("skirt_length", "increase", "*", "*"),
    Capability("skirt_length", "decrease", "*", "*"),
    Capability("pants_length", "increase", "*", "*"),
    Capability("pants_length", "decrease", "*", "*"),
    Capability("shirt_fit", "increase", "close-fitting", "loose"),
    Capability("shirt_fit", "decrease", "loose", "close-fitting"),
    Capability("sleeve_presence", "add", "sleeveless", "*"),
    Capability("sleeve_presence", "remove", "*", "sleeveless"),
    Capability("hood_presence", "add", "hoodless", "hooded"),
    Capability("hood_presence", "remove", "hooded", "hoodless"),
    Capability("collar_presence", "add", "collarless", "*"),
    Capability("collar_presence", "remove", "*", "collarless"),
    Capability("godet_insert_count", "increase", "*", "*"),
    Capability("godet_insert_count", "decrease", "*", "*"),
    Capability("neckline_type", "change", "*", "CircleNeckHalf"),
    Capability("neckline_type", "change", "*", "CurvyNeckHalf"),
    Capability("neckline_type", "change", "*", "VNeckHalf"),
    Capability("neckline_type", "change", "*", "SquareNeckHalf"),
    Capability("neckline_type", "change", "*", "CircleArcNeckHalf"),
    Capability("neckline_type", "change", "*", "Bezier2NeckHalf"),
)


def _concrete(capability: Capability) -> Capability:
    source_state_id = (
        f"{capability.rule_id}-source"
        if capability.source_state_id == "*"
        else capability.source_state_id
    )
    target_state_id = (
        f"{capability.rule_id}-target"
        if capability.target_state_id == "*"
        else capability.target_state_id
    )
    return Capability(
        capability.rule_id,
        capability.direction,
        source_state_id,
        target_state_id,
    )


def test_plan_discovers_sources_without_reading_specification_and_preserves_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root, config_path, builder_path = _plan_paths(tmp_path)
    _write_source(input_root, "rand_0001")
    shirt_path = "design.fixture.value"
    base = _planned(SHIRT_SHORT_LONG)
    shirt = replace(
        base,
        operation=replace(
            base.operation,
            reads=(shirt_path,),
            writes=(shirt_path,),
            before={shirt_path: 1.0},
            after={shirt_path: 2.0},
        ),
    )
    sleeve_first = _asymmetric_sleeve_planned(
        variant="first",
        left_target=0.75,
        right_target=0.80,
    )
    sleeve_second = _asymmetric_sleeve_planned(
        variant="second",
        left_target=0.85,
        right_target=0.90,
    )
    planned_by_name = {
        "shirt": shirt,
        "sleeve_first": sleeve_first,
        "sleeve_second": sleeve_second,
    }
    candidate_streams = {
        "shirt_length": (shirt,),
        "sleeve_length": (sleeve_first, sleeve_second),
    }

    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams=candidate_streams,
    )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name.endswith("_specification.json"):
            raise AssertionError("planning read the source specification")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    result, report = _run_plan(
        input_root=input_root,
        analysis_root=tmp_path / "analysis",
        config_path=config_path,
        builder_path=builder_path,
    )

    expected_row_order = (
        PLANNER_BASELINES["shirt"]["signature"],
        *sorted(
            (
                PLANNER_BASELINES["sleeve_first"]["signature"],
                PLANNER_BASELINES["sleeve_second"]["signature"],
            )
        ),
    )
    rows = report["source_probe_results"]["batch_0/random_body/data/rand_0001"][
        "probes"
    ]
    assert result["planning_failure_count"] == 0
    assert report["format"] == "semantic-edit-plan"
    assert tuple(row["operation_signature"] for row in rows) == expected_row_order
    rows_by_signature = {row["operation_signature"]: row for row in rows}
    for name, planned in planned_by_name.items():
        baseline = PLANNER_BASELINES[name]
        row = rows_by_signature[baseline["signature"]]
        assert set(row) == {
            "capability",
            "operation_signature",
            "planned_operation",
        }
        assert row["planned_operation"] == planned.to_payload()
        assert (
            sha256_bytes(canonical_json_bytes(row["planned_operation"]))
            == baseline["payload_digest"]
        )


def test_plan_records_and_skips_constructor_fact_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root, config_path, builder_path = _plan_paths(tmp_path)
    _write_source(input_root, "rand_0001", fixture_value=1.0)
    _write_source(input_root, "rand_0002", fixture_value=2.0)
    candidate = _planned(SHIRT_SHORT_LONG)
    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams={"shirt_length": (candidate,)},
    )

    def probe(design: dict[str, Any], _body: Any) -> dict[str, Any]:
        if design["fixture"]["value"] == 1.0:
            raise godet_module.ConstructorFactError(
                "failed to probe Godet constructor facts"
            )
        return {}

    monkeypatch.setattr(godet_module, "probe_constructor_facts", probe)
    result, report = _run_plan(
        input_root=input_root,
        analysis_root=tmp_path / "analysis",
        config_path=config_path,
        builder_path=builder_path,
    )

    failed_key = "batch_0/random_body/data/rand_0001"
    assert result["planning_failure_count"] == 1
    assert report["planning_failure_count"] == 1
    assert report["planning_failures"][failed_key] == {
        "error_type": "ConstructorFactError",
        "message": "failed to probe Godet constructor facts",
        "source_key": failed_key,
        "source_sample_id": "rand_0001",
    }
    assert failed_key not in report["source_probe_results"]
    assert all(
        preference["source_sample_id"] != "rand_0001"
        for preference in report["selected_slot_preferences"].values()
    )
    assert "skipped=rand_0001 error=ConstructorFactError" in capsys.readouterr().out


def test_plan_accepts_flattened_prefixed_garmentcode_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root, config_path, builder_path = _plan_paths(tmp_path)
    _write_source(
        input_root,
        "rand_0001",
        flattened=True,
        prefixed_yaml=True,
    )
    candidate = _planned(SHIRT_SHORT_LONG)
    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams={"shirt_length": (candidate,)},
    )

    result, report = _run_plan(
        input_root=input_root,
        analysis_root=tmp_path / "analysis",
        config_path=config_path,
        builder_path=builder_path,
    )

    source_key = "batch_0/random_body/rand_0001"
    assert result["source_count"] == 1
    assert result["eligible_source_count"] == 1
    assert tuple(report["source_probe_results"]) == (source_key,)
    assert report["selected_slot_preferences"]["slot_000"][
        "source_key"
    ] == source_key


def test_capability_matching_supports_only_state_wildcards() -> None:
    pattern = Capability("shirt_length", "increase", "*", "*")
    assert pattern.matches(SHIRT_SHORT_LONG)
    assert not pattern.matches(Capability("shirt_length", "decrease", "long", "short"))
    assert SHIRT_SHORT_LONG.concrete
    assert SHIRT_SHORT_LONG.to_payload()["target_state_id"] == "long"
    assert Capability.from_payload(SHIRT_SHORT_LONG.to_payload()) == SHIRT_SHORT_LONG
    assert SHIRT_SHORT_LONG.count_key == ("shirt_length:increase:short:long")

    with pytest.raises(ValueError, match="cannot be wildcards"):
        Capability("*", "increase", "short", "long")
    with pytest.raises(ValueError, match="state change"):
        Capability("shirt_length", "increase", "short", "short")
    with pytest.raises(TypeError, match="Capability"):
        pattern.matches("not-a-capability")  # type: ignore[arg-type]


def test_plan_accepts_exactly_the_24_required_capability_families(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(EXPECTED_REQUIRED_CAPABILITIES) == 24
    assert len(set(EXPECTED_REQUIRED_CAPABILITIES)) == 24
    assert all(
        not capability.concrete for capability in EXPECTED_REQUIRED_CAPABILITIES[:8]
    )
    input_root, config_path, builder_path = _plan_paths(tmp_path)
    concrete_capabilities: list[Capability] = []
    candidate_streams_by_source: dict[
        int,
        dict[str, tuple[PlannedOperation, ...]],
    ] = {}
    for index, required in enumerate(EXPECTED_REQUIRED_CAPABILITIES, start=1):
        sample_id = f"rand_{index:04d}"
        _write_source(
            input_root,
            sample_id,
            fixture_value=float(index),
        )
        concrete = _concrete(required)
        concrete_capabilities.append(concrete)
        candidate_streams_by_source[index] = {
            concrete.rule_id: (_planned(concrete, suffix=sample_id),)
        }
    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams_by_source=candidate_streams_by_source,
    )
    result, report = _run_plan(
        input_root=input_root,
        analysis_root=tmp_path / "analysis",
        config_path=config_path,
        builder_path=builder_path,
    )

    assert result["source_count"] == 24
    assert result["eligible_source_count"] == 24
    assert len(report["selected_slot_preferences"]) == 24
    selected = tuple(
        Capability.from_payload(preference["preferred_capability"])
        for preference in report["selected_slot_preferences"].values()
    )
    assert selected == tuple(concrete_capabilities)
    assert set(selected) == set(concrete_capabilities)


def test_plan_assignment_is_deterministic_and_balances_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root, config_path, builder_path = _plan_paths(tmp_path)
    reverse_input_root = tmp_path / "reverse_input"
    for index in range(8):
        sample_id = f"rand_{index:04d}"
        _write_source(input_root, sample_id)
    for index in reversed(range(8)):
        sample_id = f"rand_{index:04d}"
        _write_source(reverse_input_root, sample_id)
    candidate_streams = {
        "shirt_length": (
            _planned(SHIRT_SHORT_LONG, suffix="shirt-short"),
            _planned(SHIRT_CROPPED_REGULAR, suffix="shirt-cropped"),
        ),
        "sleeve_length": (
            _planned(SLEEVE_SHORT_LONG, suffix="sleeve-short"),
            _planned(
                Capability(
                    "sleeve_length",
                    "increase",
                    "elbow-length",
                    "long",
                ),
                suffix="sleeve-elbow",
            ),
        ),
    }
    assert any(len(stream) > 1 for stream in candidate_streams.values())
    assert all(
        len({canonical_operation_signature(item) for item in stream}) == len(stream)
        for stream in candidate_streams.values()
    )
    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams=candidate_streams,
    )
    _, forward = _run_plan(
        input_root=input_root,
        analysis_root=tmp_path / "analysis_forward",
        config_path=config_path,
        builder_path=builder_path,
    )
    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams={
            rule_id: tuple(reversed(stream))
            for rule_id, stream in candidate_streams.items()
        },
    )
    _, reverse = _run_plan(
        input_root=reverse_input_root,
        analysis_root=tmp_path / "analysis_reverse",
        config_path=config_path,
        builder_path=builder_path,
    )

    assert canonical_json_bytes(forward) == canonical_json_bytes(reverse)
    rules = Counter(
        preference["preferred_capability"]["rule_id"]
        for preference in forward["selected_slot_preferences"].values()
    )
    assert rules == {"shirt_length": 4, "sleeve_length": 4}
    assert forward["assigned_rule_distribution"] == dict(sorted(rules.items()))


def test_plan_assignment_balances_exact_transitions_within_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root, config_path, builder_path = _plan_paths(tmp_path)
    for index in range(6):
        sample_id = f"rand_{index:04d}"
        _write_source(input_root, sample_id)
    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams={
            "shirt_length": (
                _planned(SHIRT_SHORT_LONG, suffix="short"),
                _planned(SHIRT_CROPPED_REGULAR, suffix="cropped"),
            ),
        },
    )
    _, report = _run_plan(
        input_root=input_root,
        analysis_root=tmp_path / "analysis",
        config_path=config_path,
        builder_path=builder_path,
    )

    transitions = Counter(
        (
            preference["preferred_capability"]["source_state_id"],
            preference["preferred_capability"]["target_state_id"],
        )
        for preference in report["selected_slot_preferences"].values()
    )
    assert transitions == {
        ("short", "long"): 3,
        ("cropped", "regular"): 3,
    }


def test_plan_rejects_a_candidate_outside_the_required_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root, config_path, builder_path = _plan_paths(tmp_path)
    sample_id = "rand_0001"
    _write_source(input_root, sample_id)
    unknown = Capability("fixture_rule", "increase", "small", "large")
    _patch_public_plan_dependencies(
        monkeypatch,
        candidate_streams={"shirt_length": (_planned(unknown),)},
    )

    with pytest.raises(RuntimeError, match="no source has an applicable"):
        plan(
            input_root=input_root,
            batch_id="batch_0",
            analysis_root=tmp_path / "analysis",
            config_path=config_path,
            prompt_builder=builder_path,
        )


def _manifest_probe(capability: Capability, suffix: str) -> dict[str, Any]:
    planned = _planned(capability, suffix=suffix)
    return {
        "capability": capability.to_payload(),
        "operation_signature": canonical_operation_signature(planned),
        "planned_operation": planned.to_payload(),
    }


def _manifest_report() -> dict[str, Any]:
    source_key = "batch_0/random_body/data/rand_A"
    primary = _manifest_probe(SHIRT_SHORT_LONG, "primary")
    fallback = _manifest_probe(SLEEVE_SHORT_LONG, "fallback")
    return {
        "format": PLAN_FORMAT,
        "report_hash": "sha256:" + "d" * 64,
        "source_probe_results": {
            source_key: {
                "source_key": source_key,
                "probes": [fallback, primary],
            }
        },
        "selected_slot_preferences": {
            "slot_000": {
                "source_key": source_key,
                "source_sample_id": "rand_A",
                "preferred_capability": {
                    **primary["capability"],
                    "scope": "garment",
                    "operation_signature": primary["operation_signature"],
                },
            }
        },
    }


def _run_manifest(
    tmp_path: Path,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_path = tmp_path / "pilot_eligibility_report.json"
    report_path.write_bytes(canonical_json_bytes(report or _manifest_report()))
    return build_run_manifest(report_path, "seed")


def test_manifest_preserves_primary_then_fallback_candidates(
    tmp_path: Path,
) -> None:
    first = _run_manifest(tmp_path)
    second = _run_manifest(tmp_path)

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert set(first) == {"slots"}
    slot = first["slots"][0]
    assert slot["source_sample_id"] == "rand_A"
    assert [candidate["capability"]["rule_id"] for candidate in slot["candidates"]] == [
        "shirt_length",
        "sleeve_length",
    ]


def test_manifest_requires_one_exact_preferred_probe(tmp_path: Path) -> None:
    report = _manifest_report()
    report["selected_slot_preferences"]["slot_000"]["preferred_capability"][
        "operation_signature"
    ] = ("sha256:" + "e" * 64)

    with pytest.raises(ValueError, match="exactly one"):
        _run_manifest(tmp_path, report)


def test_candidate_selection_balances_and_never_retries(
    tmp_path: Path,
) -> None:
    slot = _run_manifest(tmp_path)["slots"][0]
    primary, fallback = slot["candidates"]

    assert select_candidate(slot, (), {}, {}) == primary
    assert (
        select_candidate(
            slot,
            (),
            {"shirt_length": 5, "sleeve_length": 0},
            {},
        )
        == fallback
    )
    assert (
        select_candidate(
            slot,
            {primary["operation_signature"]},
            {},
            {},
        )
        == fallback
    )
    assert (
        select_candidate(
            slot,
            {row["operation_signature"] for row in slot["candidates"]},
            {},
            {},
        )
        is None
    )
