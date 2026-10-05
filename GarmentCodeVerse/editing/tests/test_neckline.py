"""Public front-neckline constructor and boundary-effect behavior."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from editing.canonical_io import canonical_hash, design_dotted_get
from editing.config import load_config
from editing.models import (
    ProtectedStateSchemaError,
    RuleContext,
    SemanticState,
    StateUnavailableError,
    TopologyKind,
)
from editing.semantic_edit.prompt_state import (
    COMPONENT_STYLE_PATH,
    FRONT_NECKLINE_PATH,
    extract_prompt_states,
    neckline_inputs,
    protected_neckline_phrase,
)
from editing.semantic_edit.rules.neckline import (
    BACK_NECKLINE_PATH,
    FRONT_NECKLINE_READS,
    NECKLINE_TYPES,
    build_rule,
)
from editing.tests.support import (
    RAW_BODY_YAML,
    ROOT,
    make_rule_context,
    make_shirt_design,
)

CONFIG = load_config(ROOT / "editing/configs/semantic_edit.yaml")
RULE = build_rule(CONFIG)


def neckline_design(**updates: object) -> dict[str, Any]:
    return make_shirt_design(
        {
            "design.collar.f_collar.v": "CircleNeckHalf",
            "design.collar.b_collar.v": "CircleNeckHalf",
            "design.collar.width.v": 0.2,
            "design.collar.fc_depth.v": 0.4,
            "design.collar.bc_depth.v": 0.2,
            "design.collar.fc_angle.v": 95,
            "design.collar.bc_angle.v": 95,
            "design.collar.f_bezier_x.v": 0.3,
            "design.collar.f_bezier_y.v": 0.55,
            "design.collar.b_bezier_x.v": 0.15,
            "design.collar.b_bezier_y.v": 0.5,
            "design.collar.f_flip_curve.v": False,
            "design.collar.b_flip_curve.v": False,
            **updates,
        }
    )


def prompt_states_for(design: Mapping[str, Any]) -> dict[str, SemanticState]:
    try:
        phrase = protected_neckline_phrase(neckline_inputs(design))
    except StateUnavailableError:
        return {}
    text = f"A waist-length top with a {phrase}"
    document = {
        "body_prompt": "an adult body",
        "sample_prompt": text,
        "parts": {"upper": {"part_type": "top", "part_prompt": text}},
    }
    return {
        key: value
        for key, value in extract_prompt_states(design, document).items()
        if key != "shirt_fit"
    }


def classify_prompt(
    design: Mapping[str, Any],
    raw_body_yaml: bytes,
) -> Mapping[str, SemanticState]:
    assert raw_body_yaml == RAW_BODY_YAML
    return prompt_states_for(design)


def context_for(
    design: dict[str, Any],
    *,
    prompt_states: Mapping[str, SemanticState] | None = None,
) -> RuleContext:
    return make_rule_context(
        design,
        prompt_classifier=classify_prompt,
        prompt_states=(
            prompt_states_for(design) if prompt_states is None else prompt_states
        ),
        constructor_facts={},
    )


def candidate_to(target: str = "VNeckHalf"):
    source = neckline_design()
    candidates = RULE.candidates(context_for(source))
    return source, next(
        item
        for item in candidates
        if item.operation.after[FRONT_NECKLINE_PATH] == target
    )


def edge(
    first: int,
    second: int,
    *,
    label: str | None = None,
    curvature: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"endpoints": [first, second]}
    if label is not None:
        result["label"] = label
    if curvature is not None:
        result["curvature"] = curvature
    return result


def panel(vertices: list[list[float]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "vertices": vertices,
        "edges": edges,
        "label": "body",
    }


def stitch(
    left_panel: str,
    left_edge: int,
    right_panel: str,
    right_edge: int,
) -> list[dict[str, Any]]:
    return [
        {"panel": left_panel, "edge": left_edge},
        {"panel": right_panel, "edge": right_edge},
    ]


def specification(
    panels: Mapping[str, Mapping[str, Any]],
    stitches: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "pattern": {
            "panels": deepcopy(dict(panels)),
            "stitches": deepcopy(stitches),
            "panel_order": list(panels),
        },
        "parameters": {},
        "parameter_order": [],
        "properties": {"units_in_meter": 100.0},
    }


def front_panel(side: str, *, split: bool, displacement: float) -> dict[str, Any]:
    sign = 1.0 if side == "left" else -1.0
    collar = f"{side}_collar"
    shoulder = f"{side}_shoulder"
    if not split:
        return panel(
            [[0.0, 2.0], [2.0 * sign, 2.0], [2.0 * sign, 0.0], [0.0, 0.0]],
            [
                edge(0, 1, label=collar),
                edge(1, 2, label=shoulder),
                edge(2, 3, label=f"{side}_bottom"),
                edge(3, 0, label=f"{side}_center"),
            ],
        )
    return panel(
        [
            [0.0, 2.0],
            [1.0 * sign, 2.0 + displacement],
            [2.0 * sign, 2.0],
            [2.0 * sign, 0.0],
            [0.0, 0.0],
        ],
        [
            edge(0, 1, label=collar),
            edge(1, 2, label=collar),
            edge(2, 3, label=shoulder),
            edge(3, 4, label=f"{side}_bottom"),
            edge(4, 0, label=f"{side}_center"),
        ],
    )


def neckline_pair(
    *,
    displacement: float,
    split_target: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def make(split: bool, amount: float) -> dict[str, Any]:
        panels: dict[str, Mapping[str, Any]] = {}
        stitches = []
        for side in ("left", "right"):
            front = f"{side}_ftorso"
            back = f"{side}_btorso"
            panels[front] = front_panel(side, split=split, displacement=amount)
            panels[back] = panel(
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                [
                    edge(0, 1, label=f"{side}_collar"),
                    edge(1, 2, label=f"{side}_shoulder"),
                    edge(2, 3, label=f"{side}_back"),
                    edge(3, 0, label=f"{side}_side"),
                ],
            )
            stitches.append(stitch(front, 2 if split else 1, back, 1))
        center = 4 if split else 3
        stitches.append(stitch("left_ftorso", center, "right_ftorso", center))
        return specification(panels, stitches)

    return make(False, 0.0), make(split_target, displacement)


def config_with_threshold(value: float):
    return replace(
        CONFIG,
        rules={
            **CONFIG.rules,
            "neckline_type": replace(
                CONFIG.rules["neckline_type"], metric_threshold_cm=value
            ),
        },
    )


@pytest.mark.parametrize("source_constructor", NECKLINE_TYPES)
def test_constructor_allowlist_drives_ordered_state_transitions(
    source_constructor: str,
) -> None:
    design = neckline_design(**{FRONT_NECKLINE_PATH: source_constructor})
    candidates = RULE.candidates(context_for(design))

    assert [item.operation.after[FRONT_NECKLINE_PATH] for item in candidates] == [
        target for target in NECKLINE_TYPES if target != source_constructor
    ]
    assert all(
        item.transitions[0].source_state.state_id == source_constructor
        for item in candidates
    )
    assert all(
        item.transitions[0].source_state.phrase
        != item.transitions[0].target_state.phrase
        for item in candidates
    )


def test_candidate_contract_apply_and_evidence_are_consistent() -> None:
    source, candidate = candidate_to()
    operation = candidate.operation
    transition = candidate.transitions[0]

    assert operation.scope == "front"
    assert operation.direction == "change"
    assert operation.reads == FRONT_NECKLINE_READS
    assert operation.writes == (FRONT_NECKLINE_PATH,)
    assert BACK_NECKLINE_PATH in operation.reads
    assert BACK_NECKLINE_PATH not in operation.writes
    assert operation.conflicts == ("collar_presence", "hood_presence")
    assert operation.effect_metric_id == "bilateral_front_neckline_cm"
    assert operation.effect_contract_id == "boundary_topology_change"
    assert candidate.topology_kind is TopologyKind.BOUNDARY
    assert transition.source_state.phrase == "round neckline"
    assert transition.target_state.phrase == "V-neckline"

    evidence = candidate.applicability_evidence
    assert evidence["source_design_hash"] == canonical_hash(source)
    assert evidence["source_local_state"] == evidence["source_protected_state"]
    assert evidence["target_local_state"] == evidence["target_protected_state"]
    assert evidence["source_state_match"] and evidence["target_state_match"]
    assert candidate.to_payload()["conflict_evidence"]["declared_conflicts"] == [
        "collar_presence",
        "hood_presence",
    ]

    source["collar"]["f_collar"]["range"] = ["CircleNeckHalf"]
    original = deepcopy(source["collar"]["f_collar"])
    applied = RULE.apply(source, candidate)
    assert design_dotted_get(applied, FRONT_NECKLINE_PATH) == "VNeckHalf"
    assert applied["collar"]["f_collar"]["range"] == ["CircleNeckHalf"]
    assert source["collar"]["f_collar"] == original


@pytest.mark.parametrize(
    ("constructor", "updates", "phrase"),
    (
        ("CircleNeckHalf", {"design.collar.fc_depth.v": 0.60}, "deep scoop neckline"),
        ("CircleNeckHalf", {"design.collar.fc_depth.v": 0.50}, "scoop neckline"),
        ("CurvyNeckHalf", {"design.collar.fc_depth.v": 0.30}, "notched V-neckline"),
        ("VNeckHalf", {"design.collar.fc_depth.v": 0.70}, "deep V-neck"),
        ("CircleArcNeckHalf", {"design.collar.width.v": -0.10}, "V-neckline"),
        (
            "Bezier2NeckHalf",
            {"design.collar.f_flip_curve.v": True, "design.collar.f_bezier_y.v": 0.50},
            "split neckline featuring a V-shaped notch",
        ),
    ),
)
def test_parameter_dependent_transition_phrases_match_the_prompt_oracle(
    constructor: str,
    updates: dict[str, object],
    phrase: str,
) -> None:
    design = neckline_design(**{FRONT_NECKLINE_PATH: constructor, **updates})
    candidates = RULE.candidates(context_for(design))
    assert candidates
    assert {item.transitions[0].source_state.phrase for item in candidates} == {phrase}


@pytest.mark.parametrize(
    "updates",
    (
        {COMPONENT_STYLE_PATH: "Hood2Panels"},
        {COMPONENT_STYLE_PATH: "SimpleLapel"},
        {COMPONENT_STYLE_PATH: "Turtle"},
        {"design.left.enable_asym.v": True},
        {"design.meta.upper.v": None},
        {"design.meta.upper.v": "FittedShirt", "design.shirt.strapless.v": True},
        {FRONT_NECKLINE_PATH: "TrapezoidNeckHalf"},
        {FRONT_NECKLINE_PATH: "not-a-constructor"},
    ),
)
def test_rule_rejects_inapplicable_or_unapproved_constructor_states(
    updates: dict[str, object],
) -> None:
    design = neckline_design(**updates)
    assert RULE.candidates(context_for(design)) == ()


def test_rule_rejects_a_source_prompt_state_that_names_another_constructor() -> None:
    design = neckline_design()
    wrong = SemanticState(
        "neckline_type",
        "VNeckHalf",
        "V-neckline",
        {"constructor": "VNeckHalf"},
    )
    states = {**prompt_states_for(design), "neckline_type": wrong}
    with pytest.raises(ProtectedStateSchemaError, match="mismatch"):
        RULE.candidates(context_for(design, prompt_states=states))


@pytest.mark.parametrize(
    ("displacement", "threshold", "passed", "code"),
    (
        (3.0, None, True, "effect/passed"),
        (0.0, 0.1, False, "effect/metric_threshold"),
    ),
)
def test_effect_uses_bilateral_geometry_and_reports_public_evidence(
    displacement: float,
    threshold: float | None,
    passed: bool,
    code: str,
) -> None:
    _, candidate = candidate_to()
    source, target = neckline_pair(displacement=displacement)
    config = CONFIG if threshold is None else config_with_threshold(threshold)
    result = RULE.validate({}, {}, {}, source, target, candidate, config)

    assert result.passed is passed
    assert result.code == code
    details = result.evidence["details"]
    if passed:
        assert details["bilateral_front_neckline_cm"] > 0.0
        assert details["bilateral_front_neckline_cm"] >= details["threshold_cm"]
    else:
        assert details["bilateral_front_neckline_cm"] == pytest.approx(0.0)
    assert details["left_max_cm"] >= details["bilateral_front_neckline_cm"]
    assert details["right_max_cm"] >= details["bilateral_front_neckline_cm"]
    assert details["resample_points"] == 256
    assert details["topology_policy"] == "strict"


@pytest.mark.parametrize(
    ("mutation", "passed"),
    (
        ("partial_label", False),
        ("missing_shoulder_stitch", False),
        ("duplicate_label", False),
        ("branched_chain", False),
        ("changed_back", False),
        ("unrelated_stitch", False),
        ("reversed_chain_edges", True),
    ),
)
def test_effect_accepts_only_equivalent_boundary_topology(
    mutation: str,
    passed: bool,
) -> None:
    _, candidate = candidate_to()
    source, target = neckline_pair(displacement=3.0)
    front = target["pattern"]["panels"]["left_ftorso"]
    if mutation == "partial_label":
        front["edges"][0].pop("label")
    elif mutation == "missing_shoulder_stitch":
        target["pattern"]["stitches"] = target["pattern"]["stitches"][1:]
    elif mutation == "duplicate_label":
        front["edges"].append(edge(0, 1, label="left_collar"))
    elif mutation == "branched_chain":
        front["vertices"].append([1.0, 4.0])
        front["edges"].append(edge(1, 5, label="left_collar"))
    elif mutation == "changed_back":
        target["pattern"]["panels"]["left_btorso"]["vertices"][0][1] += 0.5
    elif mutation == "unrelated_stitch":
        target["pattern"]["stitches"].append(
            stitch("left_ftorso", 3, "right_ftorso", 3)
        )
    else:
        first, second = front["edges"][0], front["edges"][1]
        first["endpoints"].reverse()
        second["endpoints"].reverse()
        front["edges"][0], front["edges"][1] = second, first

    result = RULE.validate({}, {}, {}, source, target, candidate, CONFIG)
    assert result.passed is passed
    if passed:
        assert result.code == "effect/passed"
    else:
        assert result.code in {
            "effect/measurement_error",
            "effect/topology_mismatch",
        }


@pytest.mark.parametrize(
    ("anchor_drift", "passed"), ((1.0e-7, True), (1.0001e-7, False))
)
def test_shared_anchor_tolerance_is_inclusive_and_bounded(
    anchor_drift: float,
    passed: bool,
) -> None:
    _, candidate = candidate_to()
    source, target = neckline_pair(displacement=3.0)
    for side in ("left", "right"):
        target["pattern"]["panels"][f"{side}_ftorso"]["vertices"][0][1] += anchor_drift
    result = RULE.validate({}, {}, {}, source, target, candidate, CONFIG)
    assert result.passed is passed


@pytest.mark.parametrize(
    "curvature",
    (
        {"type": "quadratic", "params": [[0.5, 3.0]]},
        {"type": "quadratic", "params": []},
    ),
)
def test_serialized_curve_validation_is_exposed_through_rule_effects(
    curvature: dict[str, Any],
) -> None:
    _, candidate = candidate_to()
    source, target = neckline_pair(displacement=0.0, split_target=False)
    for side in ("left", "right"):
        target["pattern"]["panels"][f"{side}_ftorso"]["edges"][0]["curvature"] = (
            deepcopy(curvature)
        )
    result = RULE.validate(
        {}, {}, {}, source, target, candidate, config_with_threshold(0.1)
    )
    if curvature["params"]:
        assert result.passed
        assert result.evidence["details"]["bilateral_front_neckline_cm"] >= 2.0
    else:
        assert not result.passed
        assert result.code == "effect/measurement_error"
