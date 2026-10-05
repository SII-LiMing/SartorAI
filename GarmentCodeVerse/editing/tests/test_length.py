"""Behavior tests for the four public semantic length rules."""

from __future__ import annotations

from copy import deepcopy

import pytest

from editing import pattern_geometry
from editing.canonical_io import canonical_hash, design_dotted_get
from editing.semantic_edit.registry import build_semantic_registry, load_semantic_config
from editing.semantic_edit.rules.length import classify_target
from editing.tests.support import ROOT, make_rule_context, make_shirt_design


CONFIG = load_semantic_config(ROOT / "editing/configs/semantic_edit.yaml")
REGISTRY = build_semantic_registry(CONFIG)
LENGTH_RULE_IDS = (
    "shirt_length",
    "sleeve_length",
    "skirt_length",
    "pants_length",
)


def _design(rule_id: str, **updates: object) -> dict:
    values: dict[str, object] = {
        "design.shirt.length.v": 1.2,
        "design.sleeve.sleeveless.v": False,
        "design.sleeve.length.v": 0.5,
        "design.sleeve.connecting_width.v": 0.2,
    }
    if rule_id == "skirt_length":
        values.update(
            {
                "design.meta.upper.v": None,
                "design.meta.bottom.v": "Skirt2",
                "design.skirt.length.v": 0.57,
                "design.skirt.rise.v": 0.75,
            }
        )
    elif rule_id == "pants_length":
        values.update(
            {
                "design.meta.upper.v": None,
                "design.meta.bottom.v": "Pants",
                "design.pants.length.v": 0.48,
            }
        )
    values.update(updates)
    return make_shirt_design(values)


def _context(rule_id: str, **updates: object):
    return make_rule_context(
        _design(rule_id, **updates),
        prompt_classifier=lambda _design, _body: {},
        constructor_facts={},
        prompt_states={},
    )


def _candidate(rule_id: str, direction: str):
    context = _context(rule_id)
    candidate = next(
        item
        for item in REGISTRY[rule_id].candidates(context)
        if item.operation.direction == direction
    )
    return context, candidate


def _increasing_candidate(rule_id: str):
    return _candidate(rule_id, "increase")


def test_cuffed_pants_length_uses_dynamic_lower_bound() -> None:
    context = _context(
        "pants_length",
        **{
            "design.pants.length.v": 0.396,
            "design.pants.cuff.type.v": "CuffBand",
            "design.pants.cuff.cuff_len.v": 0.108,
        },
    )

    candidate = next(
        item
        for item in REGISTRY["pants_length"].candidates(context)
        if item.operation.direction == "decrease"
    )
    target = REGISTRY["pants_length"].apply(context.design, candidate)

    assert candidate.operation.after["design.pants.length.v"] == pytest.approx(0.25)
    assert design_dotted_get(target, "design.pants.length.v") == pytest.approx(0.25)
    assert "design.pants.cuff.type.v" in candidate.operation.reads
    assert "design.pants.cuff.cuff_len.v" in candidate.operation.reads


def test_cuffless_pants_length_keeps_original_lower_bound() -> None:
    context = _context(
        "pants_length",
        **{
            "design.pants.length.v": 0.396,
            "design.pants.cuff.type.v": None,
        },
    )

    candidate = next(
        item
        for item in REGISTRY["pants_length"].candidates(context)
        if item.operation.direction == "decrease"
    )

    assert candidate.operation.after["design.pants.length.v"] == pytest.approx(0.2)


def _edge(first: int, second: int, label: str) -> dict:
    return {"endpoints": [first, second], "label": label}


def _panel(height: float, label: str = "body") -> dict:
    return {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "vertices": [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, height],
            [0.0, height],
        ],
        "edges": [
            _edge(0, 1, "bottom"),
            _edge(1, 2, "side"),
            _edge(2, 3, "top"),
            _edge(3, 0, "center"),
        ],
        "label": label,
    }


def _specification(rule_id: str, height: float) -> dict:
    if rule_id == "shirt_length":
        names = ("left_ftorso", "right_ftorso", "left_btorso", "right_btorso")
        label = "body"
    elif rule_id == "sleeve_length":
        names = (
            "left_sleeve_f",
            "left_sleeve_b",
            "right_sleeve_f",
            "right_sleeve_b",
        )
        label = "arm"
    elif rule_id == "skirt_length":
        names = ("skirt_front",)
        label = "leg"
    elif rule_id == "pants_length":
        names = ("pant_b_l", "pant_b_r", "pant_f_l", "pant_f_r")
        label = "leg"
    else:
        raise AssertionError(f"unsupported test rule: {rule_id}")
    panels = {name: _panel(height, label) for name in names}
    return {
        "pattern": {
            "panels": panels,
            "stitches": [],
            "panel_order": list(panels),
        },
        "parameters": {},
        "parameter_order": [],
        "properties": {"units_in_meter": 100.0},
    }


def _validate(
    rule_id: str,
    source: dict,
    target: dict,
    *,
    direction: str = "increase",
    topology_policy: str = "strict",
):
    context, candidate = _candidate(rule_id, direction)
    target_design = REGISTRY[rule_id].apply(context.design, candidate)
    return REGISTRY[rule_id].validate(
        context.design,
        target_design,
        context.body,
        source,
        target,
        candidate,
        CONFIG,
        topology_policy=topology_policy,
    )


@pytest.mark.parametrize("rule_id", LENGTH_RULE_IDS)
def test_length_candidates_apply_and_reclassify_without_mutating_source(
    rule_id: str,
) -> None:
    context, candidate = _increasing_candidate(rule_id)
    source_hash = canonical_hash(context.design)

    target = REGISTRY[rule_id].apply(context.design, candidate)

    assert canonical_hash(context.design) == source_hash
    assert candidate.operation.rule_id == rule_id
    assert candidate.operation.direction == "increase"
    for path, value in candidate.operation.after.items():
        assert design_dotted_get(target, path) == value
    observation = classify_target(
        rule_id,
        target,
        context.body,
        scope=candidate.operation.scope,
    )
    assert observation.rule_id == rule_id
    assert all(
        channel.category
        in {transition.target_state.phrase for transition in candidate.transitions}
        for channel in observation.channels
    )


@pytest.mark.parametrize(
    ("rule_id", "updates"),
    (
        ("shirt_length", {"design.shirt.length.v": 0.70}),
        (
            "sleeve_length",
            {
                "design.sleeve.length.v": 0.50,
                "design.sleeve.connecting_width.v": 0.20,
            },
        ),
        (
            "skirt_length",
            {
                "design.skirt.length.v": 0.57,
                "design.skirt.rise.v": 0.75,
            },
        ),
        ("pants_length", {"design.pants.length.v": 0.48}),
    ),
)
def test_length_candidates_skip_edits_that_do_not_change_category(
    rule_id: str,
    updates: dict[str, object],
) -> None:
    candidates = REGISTRY[rule_id].candidates(_context(rule_id, **updates))

    assert not any(
        item.operation.direction == "increase"
        and item.operation.requested_magnitude == "moderately"
        for item in candidates
    )
    assert any(
        item.operation.direction == "increase"
        and item.operation.requested_magnitude == "significantly"
        for item in candidates
    )


@pytest.mark.parametrize(
    ("rule_id", "base_height"),
    (("shirt_length", 2.0), ("skirt_length", 4.0), ("pants_length", 4.0)),
)
def test_length_metric_and_boundary_floors_are_inclusive_and_strict(
    monkeypatch: pytest.MonkeyPatch,
    rule_id: str,
    base_height: float,
) -> None:
    rule_config = CONFIG.rules[rule_id]
    metric_floor = rule_config.effective_metric_threshold_cm
    boundary_floor = rule_config.effective_boundary_threshold_cm
    source = _specification(rule_id, base_height)
    target = _specification(rule_id, base_height + metric_floor)
    monkeypatch.setattr(
        pattern_geometry,
        "boundary_displacement_cm",
        lambda *_args: boundary_floor,
    )

    exact = _validate(rule_id, source, target)
    metric_below = _validate(
        rule_id,
        source,
        _specification(rule_id, base_height + metric_floor - 5.0e-10),
    )
    monkeypatch.setattr(
        pattern_geometry,
        "boundary_displacement_cm",
        lambda *_args: boundary_floor - 5.0e-10,
    )
    boundary_below = _validate(rule_id, source, target)

    assert exact.passed
    details = exact.evidence["details"]["v4_effect_result"]["evidence"]["details"]
    assert details["metric_effect_size_cm"] == pytest.approx(metric_floor)
    assert details["target_boundary_displacement_cm"] == pytest.approx(boundary_floor)
    assert not metric_below.passed
    assert metric_below.stage == "metric"
    assert metric_below.code == "metric_threshold_failed"
    assert not boundary_below.passed
    assert boundary_below.stage == "boundary"
    assert boundary_below.code == "boundary_threshold_failed"


@pytest.mark.parametrize(
    ("rule_id", "base_height"),
    (("shirt_length", 2.0), ("skirt_length", 4.0), ("pants_length", 4.0)),
)
def test_length_validate_accepts_real_decrease_at_the_metric_floor(
    monkeypatch: pytest.MonkeyPatch,
    rule_id: str,
    base_height: float,
) -> None:
    rule_config = CONFIG.rules[rule_id]
    metric_floor = rule_config.effective_metric_threshold_cm
    monkeypatch.setattr(
        pattern_geometry,
        "boundary_displacement_cm",
        lambda *_args: rule_config.effective_boundary_threshold_cm,
    )

    result = _validate(
        rule_id,
        _specification(rule_id, base_height + metric_floor),
        _specification(rule_id, base_height),
        direction="decrease",
    )

    assert result.passed
    evidence = result.evidence["details"]["v4_effect_result"]["evidence"]
    assert evidence["expected_direction"] == "decrease"
    assert evidence["details"]["metric_effect_size_cm"] == pytest.approx(metric_floor)


def test_length_validate_rejects_noop_and_wrong_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule_id = "shirt_length"
    floor = CONFIG.rules[rule_id].effective_boundary_threshold_cm
    monkeypatch.setattr(
        pattern_geometry,
        "boundary_displacement_cm",
        lambda *_args: floor,
    )
    source = _specification(rule_id, 20.0)

    noop = _validate(rule_id, source, deepcopy(source))
    wrong_direction = _validate(
        rule_id,
        source,
        _specification(rule_id, 2.0),
    )

    assert not noop.passed
    assert noop.code == "operation_noop"
    assert not wrong_direction.passed
    assert wrong_direction.code == "effect_direction_failed"


def test_strict_topology_rejects_drift_after_measuring_the_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule_id = "shirt_length"
    source = _specification(rule_id, 2.0)
    target = _specification(rule_id, 20.0)
    target["pattern"]["panels"]["left_ftorso"]["label"] = "changed"
    monkeypatch.setattr(
        pattern_geometry,
        "boundary_displacement_cm",
        lambda *_args: CONFIG.rules[rule_id].effective_boundary_threshold_cm,
    )

    result = _validate(rule_id, source, target)

    assert not result.passed
    assert result.code == "undeclared_topology_change"
    details = result.evidence["details"]["v4_effect_result"]["evidence"]["details"]
    assert details["strict_topology_match"] is False
    assert details["metric_effect_size_cm"] == pytest.approx(18.0)


def test_local_topology_accepts_metadata_drift_but_rejects_open_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule_id = "skirt_length"
    source = _specification(rule_id, 4.0)
    target = _specification(rule_id, 24.0)
    target["artifact_note"] = "rebuilt"
    monkeypatch.setattr(
        pattern_geometry,
        "resampled_boundary_displacement_cm",
        lambda *_args: CONFIG.rules[rule_id].effective_boundary_threshold_cm,
    )

    accepted = _validate(rule_id, source, target, topology_policy="local")
    assert accepted.passed
    assert (
        accepted.evidence["details"]["v4_effect_result"]["evidence"]["details"][
            "strict_topology_match"
        ]
        is False
    )

    opened = deepcopy(target)
    opened["pattern"]["panels"]["skirt_front"]["edges"].pop()

    def reject_open_boundary(_before, after, _names):
        if len(after["pattern"]["panels"]["skirt_front"]["edges"]) < 4:
            raise pattern_geometry.TopologyMismatchError("open panel boundary")
        return CONFIG.rules[rule_id].effective_boundary_threshold_cm

    monkeypatch.setattr(
        pattern_geometry,
        "resampled_boundary_displacement_cm",
        reject_open_boundary,
    )
    rejected = _validate(rule_id, source, opened, topology_policy="local")

    assert not rejected.passed
    assert rejected.code == "effect_metric_failed"
    failure = rejected.evidence["details"]["v4_effect_result"]["failure"]
    assert "open panel boundary" in failure["message"]
