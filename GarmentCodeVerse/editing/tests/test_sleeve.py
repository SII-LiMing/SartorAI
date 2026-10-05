"""Public behavior of sleeve-length and sleeve-presence rules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

import editing.pattern_geometry as pattern_geometry
from editing.models import Rule, TopologyKind
from editing.semantic_edit.registry import load_semantic_config
from editing.semantic_edit.rules.length import (
    build_rules as build_length_rules,
    classify_target,
)
from editing.semantic_edit.rules.sleeve import (
    SLEEVE_ADD_MAX_CONNECTING_WIDTH,
    SLEEVE_ADD_MAX_LENGTH,
    SLEEVE_PRESENCE_READS,
    SLEEVELESS_PATH,
    build_rules as build_presence_rules,
    classify_sleeve_state,
    evaluate_sleeve_presence_pilot_guardrail,
)
from editing.tests.support import BODY, make_rule_context, make_shirt_design


CONFIG = load_semantic_config(
    Path(__file__).resolve().parents[1] / "configs/semantic_edit.yaml"
)


def _design(**updates: Any) -> dict[str, Any]:
    return make_shirt_design(
        {
            "design.shirt.width.v": 0.85,
            "design.shirt.flare.v": 1.20,
            "design.shirt.length.v": 1.20,
            "design.sleeve.sleeveless.v": False,
            "design.sleeve.armhole_shape.v": "ArmholeCurve",
            "design.sleeve.cuff.type.v": None,
            "design.sleeve.length.v": 0.50,
            "design.sleeve.connecting_width.v": 0.20,
            **updates,
        }
    )


def _context(design: Mapping[str, Any]):
    return make_rule_context(
        design,
        prompt_classifier=lambda _design, _body: {},
        prompt_states={},
        constructor_facts={},
    )


@pytest.fixture(scope="module")
def sleeve_length_rule() -> Rule:
    return {rule.id: rule for rule in build_length_rules(CONFIG)}["sleeve_length"]


@pytest.fixture(scope="module")
def sleeve_presence_rule() -> Rule:
    (rule,) = build_presence_rules(CONFIG)
    return rule


def _candidate(
    rule: Rule,
    design: Mapping[str, Any],
    *,
    scope: str = "both",
    direction: str,
    magnitude: str,
):
    return next(
        candidate
        for candidate in rule.candidates(_context(design))
        if candidate.operation.scope == scope
        and candidate.operation.direction == direction
        and candidate.operation.requested_magnitude == magnitude
    )


@pytest.mark.parametrize(
    ("length", "width", "category"),
    [
        (0.349999, 0.20, "short"),
        (0.350000, 0.20, "elbow-length"),
        (0.719999, 0.20, "elbow-length"),
        (0.720000, 0.20, "three-quarter-length"),
        (0.939999, 0.20, "three-quarter-length"),
        (0.940000, 0.20, "long"),
        (0.340000, 1.00, "elbow-length"),
        (0.500000, 1.00, "three-quarter-length"),
        (0.800000, 1.00, "long"),
    ],
)
def test_sleeve_length_classification_boundaries(
    length: float,
    width: float,
    category: str,
) -> None:
    observation = classify_target(
        "sleeve_length",
        _design(
            **{
                "design.sleeve.length.v": length,
                "design.sleeve.connecting_width.v": width,
            }
        ),
        BODY,
        scope="both",
    )

    assert observation.scope == "both"
    assert tuple(channel.name for channel in observation.channels) == ("left", "right")
    assert {channel.category for channel in observation.channels} == {category}


@pytest.mark.parametrize(
    ("source", "direction", "magnitude", "source_state", "target_state"),
    [
        (0.34, "increase", "moderately", "short", "elbow-length"),
        (0.35, "decrease", "moderately", "elbow-length", "short"),
        (0.50, "increase", "significantly", "elbow-length", "three-quarter-length"),
        (0.94, "decrease", "moderately", "long", "elbow-length"),
    ],
)
def test_sleeve_length_candidates_transition_and_apply(
    sleeve_length_rule: Rule,
    source: float,
    direction: str,
    magnitude: str,
    source_state: str,
    target_state: str,
) -> None:
    design = _design(**{"design.sleeve.length.v": source})
    candidate = _candidate(
        sleeve_length_rule,
        design,
        direction=direction,
        magnitude=magnitude,
    )
    operation = candidate.operation
    (transition,) = candidate.transitions

    assert operation.rule_id == "sleeve_length"
    assert operation.writes == ("design.sleeve.length.v",)
    assert operation.direction == direction
    assert candidate.topology_kind is TopologyKind.NONE
    assert transition.channels == ("left", "right")
    assert transition.source_state.state_id == source_state
    assert transition.target_state.state_id == target_state

    target = sleeve_length_rule.apply(design, candidate)
    observation = classify_target("sleeve_length", target, BODY, scope="both")
    assert {channel.category for channel in observation.channels} == {target_state}
    evidence = candidate.applicability_evidence
    assert {state["state_id"] for state in evidence["source_states"]} == {source_state}
    assert {state["state_id"] for state in evidence["target_states"]} == {target_state}
    sleeveless = _context(_design(**{"design.sleeve.sleeveless.v": True}))
    assert sleeve_length_rule.candidates(sleeveless) == ()


@pytest.mark.parametrize(
    ("scope", "direction", "expected_writes", "expected_channels"),
    [
        (
            "both",
            "decrease",
            ("design.left.sleeve.length.v", "design.sleeve.length.v"),
            {("left",), ("right",)},
        ),
        ("left", "decrease", ("design.left.sleeve.length.v",), {("left",)}),
        ("right", "increase", ("design.sleeve.length.v",), {("right",)}),
    ],
)
def test_asymmetric_sleeve_length_scopes_only_declared_channels(
    sleeve_length_rule: Rule,
    scope: str,
    direction: str,
    expected_writes: tuple[str, ...],
    expected_channels: set[tuple[str, ...]],
) -> None:
    design = _design(
        **{
            "design.left.enable_asym.v": True,
            "design.left.sleeve.sleeveless.v": False,
            "design.left.sleeve.length.v": 0.50,
            "design.left.sleeve.connecting_width.v": 0.20,
            "design.sleeve.length.v": 0.80,
        }
    )
    candidate = _candidate(
        sleeve_length_rule,
        design,
        scope=scope,
        direction=direction,
        magnitude="moderately",
    )

    assert candidate.operation.writes == expected_writes
    assert {
        transition.channels for transition in candidate.transitions
    } == expected_channels
    target = sleeve_length_rule.apply(design, candidate)
    before = classify_target("sleeve_length", design, BODY, scope=scope)
    after = classify_target("sleeve_length", target, BODY, scope=scope)
    assert all(
        old.category != new.category
        for old, new in zip(before.channels, after.channels)
    )


def _panel(height: float, *, label: str) -> dict[str, Any]:
    labels = ("bottom", "side", "top", "center")
    return {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "vertices": [[0.0, 0.0], [2.0, 0.0], [2.0, height], [0.0, height]],
        "edges": [
            {"endpoints": [index, (index + 1) % 4], "label": edge_label}
            for index, edge_label in enumerate(labels)
        ],
        "label": label,
    }


def _specification(
    panels: Mapping[str, Mapping[str, Any]],
    stitches: list[list[Any]] | None = None,
) -> dict[str, Any]:
    return {
        "pattern": {
            "panels": deepcopy(dict(panels)),
            "stitches": deepcopy(stitches or []),
            "panel_order": list(panels),
        },
        "parameters": {},
        "parameter_order": [],
        "properties": {"units_in_meter": 100.0},
    }


def _sleeve_length_specification(
    left_height: float,
    right_height: float | None = None,
) -> dict[str, Any]:
    right_height = left_height if right_height is None else right_height
    panels = {
        f"{side}_sleeve_{half}": _panel(height, label="arm")
        for side, height in (("left", left_height), ("right", right_height))
        for half in ("f", "b")
    }
    return _specification(panels)


@pytest.mark.parametrize(
    ("metric_delta", "boundary_delta", "code", "stage"),
    [
        (4.0, 10.0, "effect/passed", "effect"),
        (4.0 - 5.0e-10, 10.0, "metric_threshold_failed", "metric"),
        (4.0, 10.0 - 5.0e-10, "boundary_threshold_failed", "boundary"),
    ],
)
def test_sleeve_length_effect_floors_and_evidence(
    sleeve_length_rule: Rule,
    monkeypatch: pytest.MonkeyPatch,
    metric_delta: float,
    boundary_delta: float,
    code: str,
    stage: str,
) -> None:
    planned = _candidate(
        sleeve_length_rule,
        _design(),
        direction="increase",
        magnitude="significantly",
    )
    monkeypatch.setattr(
        pattern_geometry,
        "boundary_displacement_cm",
        lambda *_args: boundary_delta,
    )
    result = sleeve_length_rule.validate(
        {},
        {},
        {},
        _sleeve_length_specification(50.0),
        _sleeve_length_specification(50.0 + metric_delta),
        planned,
        CONFIG,
    )

    assert result.code == code
    assert result.stage == stage
    payload = result.evidence["details"]["v4_effect_result"]
    evidence = payload["evidence"]
    assert evidence["metric_id"] == "sleeve_seam_length_cm"
    assert evidence["metric_threshold_cm"] == 4.0
    assert evidence["boundary_threshold_cm"] == 10.0
    assert evidence["details"]["target_boundary_displacement_cm"] == boundary_delta


def test_sleeve_length_effect_enforces_scope_and_topology(
    sleeve_length_rule: Rule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _design(
        **{
            "design.left.enable_asym.v": True,
            "design.left.sleeve.sleeveless.v": False,
            "design.left.sleeve.length.v": 0.50,
            "design.left.sleeve.connecting_width.v": 0.20,
            "design.sleeve.length.v": 0.80,
        }
    )
    planned = _candidate(
        sleeve_length_rule,
        design,
        scope="left",
        direction="increase",
        magnitude="significantly",
    )
    monkeypatch.setattr(pattern_geometry, "boundary_displacement_cm", lambda *_: 10.0)
    source = _sleeve_length_specification(50.0, 50.0)
    scoped = _sleeve_length_specification(54.0, 50.0)

    accepted = sleeve_length_rule.validate({}, {}, {}, source, scoped, planned, CONFIG)
    assert accepted.passed
    assert accepted.evidence["details"]["v4_effect_result"]["evidence"]["details"][
        "untargeted_side_delta_cm"
    ] == {"right": 0.0}

    outside_scope = _sleeve_length_specification(54.0, 54.0)
    rejected = sleeve_length_rule.validate(
        {}, {}, {}, source, outside_scope, planned, CONFIG
    )
    assert rejected.code == "effect_direction_failed"

    changed_topology = deepcopy(scoped)
    changed_topology["pattern"]["panels"]["left_sleeve_f"]["label"] = "changed"
    topology = sleeve_length_rule.validate(
        {}, {}, {}, source, changed_topology, planned, CONFIG
    )
    assert topology.code == "undeclared_topology_change"


@pytest.mark.parametrize(
    ("sleeveless", "length", "direction", "source_state", "target_state"),
    [
        (True, 0.30, "add", "sleeveless", "short"),
        (False, 0.95, "remove", "long", "sleeveless"),
    ],
)
def test_sleeve_presence_candidates_toggle_both_channels(
    sleeve_presence_rule: Rule,
    sleeveless: bool,
    length: float,
    direction: str,
    source_state: str,
    target_state: str,
) -> None:
    design = _design(
        **{
            "design.sleeve.sleeveless.v": sleeveless,
            "design.sleeve.length.v": length,
        }
    )
    (candidate,) = sleeve_presence_rule.candidates(_context(design))
    operation = candidate.operation
    transition = candidate.transitions[0]

    assert operation.direction == direction
    assert operation.structural_level == 2
    assert operation.magnitude == "structural"
    assert operation.reads == tuple(sorted(SLEEVE_PRESENCE_READS))
    assert operation.writes == (SLEEVELESS_PATH,)
    assert operation.before[SLEEVELESS_PATH] is sleeveless
    assert operation.after[SLEEVELESS_PATH] is (not sleeveless)
    assert candidate.topology_kind is TopologyKind.MEMBERSHIP
    assert transition.scope == "both"
    assert transition.channels == ("left", "right")
    assert transition.source_state.state_id == source_state
    assert transition.target_state.state_id == target_state

    target = sleeve_presence_rule.apply(design, candidate)
    assert classify_sleeve_state(target, BODY).state_id == target_state
    evidence = candidate.applicability_evidence
    assert evidence["source_local_state"]["state_id"] == source_state
    assert evidence["target_local_state"]["state_id"] == target_state
    assert evidence["symmetric_channels"] == ["left", "right"]
    assert candidate.conflict_evidence["declared_conflicts"] == ["sleeve_length"]


@pytest.mark.parametrize(
    ("updates", "applicable"),
    [
        ({}, True),
        ({"design.sleeve.armhole_shape.v": "ArmholeSquare"}, False),
        ({"design.sleeve.cuff.type.v": "CuffBand"}, False),
        ({"design.left.enable_asym.v": True}, False),
        (
            {"design.meta.upper.v": "FittedShirt", "design.shirt.strapless.v": False},
            True,
        ),
        (
            {"design.meta.upper.v": "FittedShirt", "design.shirt.strapless.v": True},
            False,
        ),
        ({"design.shirt.strapless.v": True}, True),
        ({"design.sleeve.sleeveless.v": True, "design.shirt.strapless.v": True}, False),
        (
            {
                "design.sleeve.sleeveless.v": True,
                "design.shirt.width.v": 1.20,
                "design.shirt.flare.v": 1.45,
            },
            False,
        ),
    ],
)
def test_sleeve_presence_applicability(
    sleeve_presence_rule: Rule,
    updates: Mapping[str, Any],
    applicable: bool,
) -> None:
    assert (
        bool(sleeve_presence_rule.candidates(_context(_design(**updates))))
        is applicable
    )


@pytest.mark.parametrize(
    ("updates", "passed", "violation"),
    [
        ({}, True, None),
        (
            {"design.sleeve.length.v": SLEEVE_ADD_MAX_LENGTH + 0.001},
            False,
            "length_above_max",
        ),
        (
            {
                "design.sleeve.connecting_width.v": SLEEVE_ADD_MAX_CONNECTING_WIDTH
                + 0.001
            },
            False,
            "connecting_width_above_max",
        ),
        ({"design.shirt.strapless.v": True}, False, "strapless_source"),
    ],
)
def test_sleeve_add_guardrail_boundaries_and_evidence(
    sleeve_presence_rule: Rule,
    updates: Mapping[str, Any],
    passed: bool,
    violation: str | None,
) -> None:
    values = {
        "design.sleeve.sleeveless.v": True,
        "design.sleeve.length.v": SLEEVE_ADD_MAX_LENGTH,
        "design.sleeve.connecting_width.v": SLEEVE_ADD_MAX_CONNECTING_WIDTH,
        **updates,
    }
    evidence = evaluate_sleeve_presence_pilot_guardrail(
        direction="add",
        strapless=values.get("design.shirt.strapless.v", False),
        sleeve_length=values["design.sleeve.length.v"],
        connecting_width=values["design.sleeve.connecting_width.v"],
    )

    assert evidence["passed"] is passed
    assert evidence["applied"] is True
    assert bool(sleeve_presence_rule.candidates(_context(_design(**values)))) is passed
    if passed:
        assert evidence["violations"] == ()
        assert evidence["observed"]["design.sleeve.length.v"] == SLEEVE_ADD_MAX_LENGTH
    else:
        assert any(violation in item["code"] for item in evidence["violations"])


def _stitch(first: str, first_edge: int, second: str, second_edge: int) -> list[Any]:
    return [
        {"panel": first, "edge": first_edge},
        {"panel": second, "edge": second_edge},
    ]


def _presence_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    torso: dict[str, Mapping[str, Any]] = {}
    for name in ("left_ftorso", "right_ftorso", "left_btorso", "right_btorso"):
        panel = _panel(2.0, label="body")
        panel["edges"][1]["label"] = f"{name.split('_', 1)[0]}_armhole"
        torso[name] = panel
    source = _specification(torso)
    present = deepcopy(torso)
    for side in ("left", "right"):
        present[f"{side}_sleeve_f"] = _panel(2.0, label="arm")
        present[f"{side}_sleeve_b"] = _panel(2.0, label="arm")
    stitches = []
    for side in ("left", "right"):
        stitches.extend(
            [
                _stitch(f"{side}_sleeve_f", 1, f"{side}_ftorso", 1),
                _stitch(f"{side}_sleeve_b", 1, f"{side}_btorso", 1),
                _stitch(f"{side}_sleeve_f", 2, f"{side}_sleeve_b", 3),
                _stitch(f"{side}_sleeve_f", 3, f"{side}_sleeve_b", 0),
            ]
        )
    return source, _specification(present, stitches)


@pytest.mark.parametrize("direction", ["add", "remove"])
def test_sleeve_presence_effect_accepts_exact_membership_and_attachments(
    sleeve_presence_rule: Rule,
    direction: str,
) -> None:
    absent, present = _presence_pair()
    if direction == "add":
        source, target = absent, present
        design = _design(
            **{
                "design.sleeve.sleeveless.v": True,
                "design.sleeve.length.v": 0.30,
            }
        )
    else:
        source, target = present, absent
        design = _design(**{"design.sleeve.length.v": 0.95})
    (planned,) = sleeve_presence_rule.candidates(_context(design))

    result = sleeve_presence_rule.validate({}, {}, {}, source, target, planned, CONFIG)

    assert result.passed
    assert result.code == "effect/passed"
    details = result.evidence["details"]
    assert details["changed_panels"] == 4
    assert details["armhole_stitch_membership"] == {"left": True, "right": True}
    assert all(details["sleeve_panel_attachments"].values())
    assert details["cuff_panel_names"] == ()


@pytest.mark.parametrize(
    "mutation", ["missing_panel", "cuff", "metadata", "cross_side"]
)
def test_sleeve_presence_effect_rejects_nonlocal_topology_changes(
    sleeve_presence_rule: Rule,
    mutation: str,
) -> None:
    source, target = _presence_pair()
    if mutation == "missing_panel":
        del target["pattern"]["panels"]["right_sleeve_b"]
        target["pattern"]["panel_order"].remove("right_sleeve_b")
    elif mutation == "cuff":
        for specification in (source, target):
            specification["pattern"]["panels"]["sl_left_cuff_f"] = _panel(
                1.0, label="arm"
            )
            specification["pattern"]["panel_order"].append("sl_left_cuff_f")
    elif mutation == "metadata":
        target["properties"]["units_in_meter"] = 200.0
    else:
        target["pattern"]["stitches"][0][1]["panel"] = "right_ftorso"
    design = _design(
        **{"design.sleeve.sleeveless.v": True, "design.sleeve.length.v": 0.30}
    )
    (planned,) = sleeve_presence_rule.candidates(_context(design))

    result = sleeve_presence_rule.validate({}, {}, {}, source, target, planned, CONFIG)

    assert not result.passed
    assert result.code == "effect/topology_mismatch"
