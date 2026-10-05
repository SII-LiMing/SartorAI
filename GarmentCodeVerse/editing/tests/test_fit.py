"""Public behavior of the ordinary-shirt fit rule."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from editing.models import Rule, SemanticState, TopologyKind
from editing.semantic_edit.registry import load_semantic_config
from editing.semantic_edit.rules.fit import (
    CLOSE_FIT_TARGET_WIDTH,
    FIT_READ_PATHS,
    FIT_WIDTH_PATH,
    build_rules,
    classify_design,
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
            **updates,
        }
    )


def _protected(state: SemanticState, *, state_id: str | None = None) -> SemanticState:
    identity = state.state_id if state_id is None else state_id
    return SemanticState(
        rule_id="shirt_fit",
        state_id=identity,
        phrase=identity,
        attributes={"protected_phrase_present": True},
    )


def _fit_context(
    design: Mapping[str, Any],
    *,
    body: Mapping[str, Any] = BODY,
    source_prompt: str = "agree",
    target_prompt: str = "agree",
):
    source = classify_design(design, body)
    prompt_states: dict[str, SemanticState] = {}
    if source_prompt == "agree" and source.state_id in {"close-fitting", "loose"}:
        prompt_states["shirt_fit"] = _protected(source)
    elif source_prompt == "wrong":
        prompt_states["shirt_fit"] = _protected(
            source,
            state_id=(
                "loose" if source.state_id == "close-fitting" else "close-fitting"
            ),
        )

    def prompt_classifier(
        target_design: dict[str, Any],
        _raw_body_yaml: bytes,
    ) -> Mapping[str, SemanticState]:
        target = classify_design(target_design, body)
        if target_prompt == "missing":
            return {}
        if target_prompt == "wrong":
            return {
                "shirt_fit": _protected(
                    target,
                    state_id=(
                        "loose"
                        if target.state_id == "close-fitting"
                        else "close-fitting"
                    ),
                )
            }
        return {"shirt_fit": _protected(target)}

    return make_rule_context(
        design,
        prompt_classifier=prompt_classifier,
        prompt_states=prompt_states,
        constructor_facts={},
        body=body,
    )


@pytest.fixture(scope="module")
def fit_rule() -> Rule:
    (rule,) = build_rules(CONFIG)
    return rule


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"design.shirt.width.v": 0.880, "design.shirt.flare.v": 1.249}, "unnamed"),
        ({"design.shirt.width.v": 1.151, "design.shirt.flare.v": 1.20}, "loose"),
        ({"design.shirt.flare.v": 1.450}, "loose"),
    ],
)
def test_fit_classification_boundaries(
    updates: Mapping[str, Any],
    expected: str,
) -> None:
    assert classify_design(_design(**updates), BODY).state_id == expected


@pytest.mark.parametrize(
    (
        "source_width",
        "source_state",
        "target_state",
        "direction",
        "target_width",
    ),
    [
        (0.85, "close-fitting", "loose", "increase", 1.20),
        (1.20, "loose", "close-fitting", "decrease", CLOSE_FIT_TARGET_WIDTH),
    ],
)
def test_fit_candidates_expose_named_transition_apply_and_evidence(
    fit_rule: Rule,
    source_width: float,
    source_state: str,
    target_state: str,
    direction: str,
    target_width: float,
) -> None:
    context = _fit_context(_design(**{"design.shirt.width.v": source_width}))
    (candidate,) = fit_rule.candidates(context)
    operation = candidate.operation
    (transition,) = candidate.transitions

    assert operation.rule_id == "shirt_fit"
    assert operation.scope == "garment"
    assert operation.direction == direction
    assert operation.reads == tuple(sorted(FIT_READ_PATHS))
    assert operation.writes == (FIT_WIDTH_PATH,)
    assert operation.after[FIT_WIDTH_PATH] == pytest.approx(target_width)
    assert candidate.topology_kind is TopologyKind.NONE
    assert transition.channels == ("garment",)
    assert transition.feature_phrase == "the shirt"
    assert transition.source_state.state_id == source_state
    assert transition.target_state.state_id == target_state

    target = fit_rule.apply(context.design, candidate)
    assert classify_design(target, context.body).state_id == target_state
    assert target["shirt"]["width"]["v"] == pytest.approx(target_width)

    evidence = candidate.applicability_evidence
    assert evidence["source_state_match"] is True
    assert evidence["target_state_match"] is True
    assert evidence["metric_id"] == operation.effect_metric_id
    assert evidence["metric_value_cm"] >= evidence["metric_threshold_cm"]
    assert candidate.conflict_evidence["declared_conflicts"] == ["shirt_length"]


@pytest.mark.parametrize(
    ("updates", "source_prompt", "target_prompt"),
    [
        ({"design.meta.upper.v": "FittedShirt"}, "agree", "agree"),
        ({"design.left.enable_asym.v": True}, "agree", "agree"),
        ({"design.shirt.length.v": 0.999}, "agree", "agree"),
        ({"design.shirt.flare.v": 1.45}, "agree", "agree"),
        ({"design.shirt.width.v": 1.00}, "agree", "agree"),
        ({}, "missing", "agree"),
        ({}, "wrong", "agree"),
        ({}, "agree", "missing"),
        ({}, "agree", "wrong"),
        (
            {
                "design.sleeve.sleeveless.v": True,
                "design.shirt.width.v": 1.20,
                "design.shirt.flare.v": 1.45,
            },
            "agree",
            "agree",
        ),
    ],
)
def test_fit_rejects_inapplicable_or_unconfirmed_endpoints(
    fit_rule: Rule,
    updates: Mapping[str, Any],
    source_prompt: str,
    target_prompt: str,
) -> None:
    context = _fit_context(
        _design(**updates),
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )
    assert fit_rule.candidates(context) == ()


@pytest.mark.parametrize(
    ("metric_delta", "applicable"),
    [(0.0, True), (-1.0e-8, False)],
)
def test_fit_candidate_metric_floor_is_inclusive(
    fit_rule: Rule,
    metric_delta: float,
    applicable: bool,
) -> None:
    width_delta = 1.20 - 0.85
    bust = (
        CONFIG.rules["shirt_fit"].effective_metric_threshold_cm + metric_delta
    ) / width_delta
    body = {
        **BODY,
        "bust": bust,
        "waist": 0.70 * bust,
        "hips": 0.90 * bust,
    }
    candidates = fit_rule.candidates(_fit_context(_design(), body=body))

    assert bool(candidates) is applicable
    if candidates:
        assert candidates[0].applicability_evidence["metric_value_cm"] == pytest.approx(
            10.0
        )


def _panel(width: float) -> dict[str, Any]:
    labels = ("bottom", "side", "top", "center")
    return {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "vertices": [[0.0, 0.0], [width, 0.0], [width, 2.0], [0.0, 2.0]],
        "edges": [
            {"endpoints": [index, (index + 1) % 4], "label": label}
            for index, label in enumerate(labels)
        ],
        "label": "body",
    }


def _fit_specification(width: float) -> dict[str, Any]:
    names = ("left_ftorso", "right_ftorso", "left_btorso", "right_btorso")
    return {
        "pattern": {
            "panels": {name: _panel(width) for name in names},
            "stitches": [],
            "panel_order": list(names),
        },
        "parameters": {},
        "parameter_order": [],
        "properties": {"units_in_meter": 100.0},
    }


@pytest.mark.parametrize(
    ("target_width", "expected_code"),
    [(0.95, "effect/passed"), (0.94999999, "effect/metric_threshold")],
)
def test_fit_effect_metric_floor_and_evidence(
    fit_rule: Rule,
    target_width: float,
    expected_code: str,
) -> None:
    (planned,) = fit_rule.candidates(_fit_context(_design()))
    result = fit_rule.validate(
        source_design={"shirt": {"width": {"v": 0.85}}},
        target_design={"shirt": {"width": {"v": target_width}}},
        body={"bust": 100.0},
        source_specification=_fit_specification(2.0),
        target_specification=_fit_specification(2.2),
        planned_operation=planned,
        config=CONFIG,
    )

    assert result.code == expected_code
    assert result.passed is (expected_code == "effect/passed")
    details = result.evidence["details"]
    assert details["full_torso_width_cm"] == pytest.approx(
        abs(target_width - 0.85) * 100.0
    )
    assert details["threshold_cm"] == 10.0
    assert details["torso_topology_unchanged"] is True


@pytest.mark.parametrize(
    ("topology_policy", "passed", "code"),
    [
        ("strict", False, "effect/topology_mismatch"),
        ("local", True, "effect/passed"),
    ],
)
def test_fit_effect_topology_policy_is_explicit(
    fit_rule: Rule,
    topology_policy: str,
    passed: bool,
    code: str,
) -> None:
    (planned,) = fit_rule.candidates(_fit_context(_design()))
    changed = _fit_specification(2.2)
    changed["pattern"]["panels"]["left_ftorso"]["edges"].append(
        {"endpoints": [0, 2], "label": "unexpected"}
    )

    result = fit_rule.validate(
        {"shirt": {"width": {"v": 0.85}}},
        {"shirt": {"width": {"v": 0.95}}},
        {"bust": 100.0},
        _fit_specification(2.0),
        changed,
        planned,
        CONFIG,
        topology_policy=topology_policy,
    )

    assert result.passed is passed
    assert result.code == code
    assert result.evidence["details"]["topology_policy"] == topology_policy
