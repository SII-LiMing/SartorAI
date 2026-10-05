"""Public collar/hood candidate, transition, evidence, and effect behavior."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
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
    extract_prompt_states,
    neckline_inputs,
    protected_neckline_phrase,
)
from editing.semantic_edit.rules.collar_hood import (
    BACK_NECKLINE_PATH,
    COLLAR_PRESENCE_READS,
    COMPONENT_STYLE_PATH,
    FRONT_NECKLINE_PATH,
    HOOD_ADD_ACCEPT_REASON,
    HOOD_ADD_GUARDRAIL_ID,
    HOOD_ADD_MAX_HOOD_LENGTH,
    HOOD_ADD_REJECTION_REASON,
    HOOD_LENGTH_PATH,
    HOOD_PRESENCE_READS,
    HOOD_REMOVE_GUARDRAIL_REASON,
    build_rules,
    classify_hood_state,
)
from editing.tests.support import (
    BODY,
    RAW_BODY_YAML,
    ROOT,
    make_rule_context,
    make_shirt_design,
)

CONFIG = load_config(ROOT / "editing/configs/semantic_edit.yaml")
RULES = {rule.id: rule for rule in build_rules(CONFIG)}


def shirt_design(**updates: object) -> dict[str, Any]:
    """Return a deterministic shirt whose collar parameters are explicit."""

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
            "design.collar.component.depth.v": 2.0,
            "design.collar.component.lapel_standing.v": False,
            **updates,
        }
    )


def prompt_states_for(design: Mapping[str, Any]) -> dict[str, SemanticState]:
    """Create the protected prompt states observed for one test design."""

    phrase = ""
    try:
        hood_state = classify_hood_state(design)
    except StateUnavailableError:
        hood_state = None
    if hood_state is not None and hood_state.state_id == "hooded":
        phrase = "hood"
    else:
        try:
            phrase = protected_neckline_phrase(neckline_inputs(design))
        except StateUnavailableError:
            phrase = ""

    text = "A waist-length top"
    if phrase:
        text += f" with a {phrase}"
    document = {
        "body_prompt": "an adult body",
        "sample_prompt": text,
        "parts": {"upper": {"part_type": "top", "part_prompt": text}},
    }
    return {
        rule_id: state
        for rule_id, state in extract_prompt_states(design, document).items()
        if rule_id != "shirt_fit"
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
    protected = prompt_states_for(design) if prompt_states is None else prompt_states
    return make_rule_context(
        design,
        prompt_classifier=classify_prompt,
        prompt_states=protected,
        constructor_facts={},
    )


def explicit_collar_design(*, present: bool) -> dict[str, Any]:
    return shirt_design(
        **{
            COMPONENT_STYLE_PATH: "SimpleLapel" if present else None,
            FRONT_NECKLINE_PATH: "SquareNeckHalf",
            "design.collar.component.lapel_standing.v": True,
        }
    )


def presence_candidate(rule_id: str, direction: str):
    if rule_id == "hood_presence":
        design = shirt_design(
            **{COMPONENT_STYLE_PATH: ("Hood2Panels" if direction == "remove" else None)}
        )
    else:
        design = explicit_collar_design(present=direction == "remove")
    candidates = RULES[rule_id].candidates(context_for(design))
    assert len(candidates) == 1
    return design, candidates[0]


def pattern_edge(
    first: int,
    second: int,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    edge: dict[str, Any] = {"endpoints": [first, second]}
    if label is not None:
        edge["label"] = label
    return edge


def pattern_panel(*, label: str = "body") -> dict[str, Any]:
    return {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "vertices": [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ],
        "edges": [
            pattern_edge(0, 1, label="bottom"),
            pattern_edge(1, 2, label="side"),
            pattern_edge(2, 3, label="top"),
            pattern_edge(3, 0, label="center"),
        ],
        "label": label,
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


def pattern_specification(
    panels: Mapping[str, Mapping[str, Any]],
    stitches: list[list[dict[str, Any]]] | None = None,
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


def upper_panels(*, collar_edges: bool) -> dict[str, dict[str, Any]]:
    panels: dict[str, dict[str, Any]] = {}
    for name, side in (
        ("left_ftorso", "left"),
        ("right_ftorso", "right"),
        ("left_btorso", "left"),
        ("right_btorso", "right"),
    ):
        panel = pattern_panel()
        if collar_edges:
            panel["edges"][2]["label"] = f"{side}_collar"
        panels[name] = panel
    panels["pocket"] = pattern_panel(label="pocket")
    return panels


def hood_effect_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    source_panels = upper_panels(collar_edges=False)
    target_panels = deepcopy(source_panels)
    target_panels["left_hood"] = pattern_panel(label="hood")
    target_panels["right_hood"] = pattern_panel(label="hood")
    target = pattern_specification(
        target_panels,
        [
            stitch("left_hood", 0, "right_hood", 0),
            stitch("left_hood", 1, "left_ftorso", 2),
            stitch("right_hood", 1, "right_ftorso", 2),
        ],
    )
    return pattern_specification(source_panels), target


def collar_effect_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    source_panels = upper_panels(collar_edges=True)
    target_panels = deepcopy(source_panels)
    for side in ("left", "right"):
        for half in ("front", "back"):
            target_panels[f"{side}_collar_{half}"] = pattern_panel(label="collar")
    target = pattern_specification(
        target_panels,
        [
            stitch("left_ftorso", 2, "left_collar_front", 0),
            stitch("right_ftorso", 2, "right_collar_front", 0),
            stitch("left_btorso", 2, "left_collar_back", 0),
            stitch("right_btorso", 2, "right_collar_back", 0),
        ],
    )
    return pattern_specification(source_panels), target


@pytest.mark.parametrize(
    (
        "rule_id",
        "direction",
        "source_value",
        "target_value",
        "source_phrase",
        "target_phrase",
        "reads",
        "conflicts",
        "metric",
    ),
    (
        (
            "hood_presence",
            "add",
            None,
            "Hood2Panels",
            "hoodless",
            "hooded",
            HOOD_PRESENCE_READS,
            ("collar_presence", "neckline_type"),
            "hood_membership",
        ),
        (
            "hood_presence",
            "remove",
            "Hood2Panels",
            None,
            "hooded",
            "hoodless",
            HOOD_PRESENCE_READS,
            ("collar_presence", "neckline_type"),
            "hood_membership",
        ),
        (
            "collar_presence",
            "add",
            None,
            "SimpleLapel",
            "collarless",
            "having a split stand collar",
            COLLAR_PRESENCE_READS,
            ("hood_presence", "neckline_type"),
            "collar_membership",
        ),
        (
            "collar_presence",
            "remove",
            "SimpleLapel",
            None,
            "having a split stand collar",
            "collarless",
            COLLAR_PRESENCE_READS,
            ("hood_presence", "neckline_type"),
            "collar_membership",
        ),
    ),
)
def test_presence_state_transitions_have_one_public_operation_contract(
    rule_id: str,
    direction: str,
    source_value: object,
    target_value: object,
    source_phrase: str,
    target_phrase: str,
    reads: tuple[str, ...],
    conflicts: tuple[str, ...],
    metric: str,
) -> None:
    source, candidate = presence_candidate(rule_id, direction)
    operation = candidate.operation
    (transition,) = candidate.transitions

    assert operation.rule_id == rule_id
    assert operation.semantic_group == "collar"
    assert operation.scope == "garment"
    assert operation.direction == direction
    assert operation.magnitude == "structural"
    assert operation.structural_level == 2
    assert operation.reads == reads
    assert operation.writes == (COMPONENT_STYLE_PATH,)
    assert operation.conflicts == conflicts
    assert operation.effect_metric_id == metric
    assert operation.effect_contract_id == "exact_membership_change"
    assert dict(operation.before) == {COMPONENT_STYLE_PATH: source_value}
    assert dict(operation.after) == {COMPONENT_STYLE_PATH: target_value}
    assert candidate.topology_kind is TopologyKind.MEMBERSHIP
    assert transition.feature_phrase == "the top"
    assert transition.channels == ("garment",)
    assert transition.source_state.phrase == source_phrase
    assert transition.target_state.phrase == target_phrase

    applied = RULES[rule_id].apply(deepcopy(source), candidate)
    assert design_dotted_get(applied, COMPONENT_STYLE_PATH) == target_value
    assert design_dotted_get(source, COMPONENT_STYLE_PATH) == source_value


@pytest.mark.parametrize(
    ("rule_id", "direction"),
    (
        ("hood_presence", "add"),
        ("hood_presence", "remove"),
        ("collar_presence", "add"),
        ("collar_presence", "remove"),
    ),
)
def test_presence_evidence_proves_prompt_and_local_state_consistency(
    rule_id: str,
    direction: str,
) -> None:
    source, candidate = presence_candidate(rule_id, direction)
    evidence = candidate.applicability_evidence
    transition = candidate.transitions[0]

    assert evidence["rule_id"] == rule_id
    assert evidence["scope"] == "garment"
    assert evidence["source_design_hash"] == canonical_hash(source)
    assert evidence["body_hash"] == canonical_hash(BODY)
    assert evidence["source_state_match"] is True
    assert evidence["target_state_match"] is True
    assert evidence["source_local_state"] == transition.source_state.to_payload()
    assert evidence["target_local_state"] == transition.target_state.to_payload()
    for endpoint in ("source", "target"):
        local = evidence[f"{endpoint}_local_state"]
        protected = evidence[f"{endpoint}_protected_state"]
        assert protected["rule_id"] == local["rule_id"]
        assert protected["state_id"] == local["state_id"]
        assert protected["phrase"] == local["phrase"]
        assert protected["attributes"]["present"] == local["attributes"]["present"]
    assert evidence["read_contract_hash"] == canonical_hash(
        {path: design_dotted_get(source, path) for path in candidate.operation.reads}
    )

    assert candidate.to_payload()["conflict_evidence"] == {
        "rule_id": rule_id,
        "declared_conflicts": list(candidate.operation.conflicts),
        "earlier_rule_ids": [],
        "earlier_operation_hashes": [],
        "earlier_write_paths": [],
        "overlap_write_paths": [],
    }


@pytest.mark.parametrize(
    ("parameters", "subtype"),
    (
        (
            {
                "design.collar.component.lapel_standing.v": True,
                FRONT_NECKLINE_PATH: "CircleNeckHalf",
                "design.collar.width.v": 0.10,
                "design.collar.fc_depth.v": 0.70,
            },
            "split stand V-neck collar",
        ),
        (
            {
                FRONT_NECKLINE_PATH: "CircleNeckHalf",
                "design.collar.fc_depth.v": 0.60,
            },
            "fold-over collar framing a deep V-shaped neckline",
        ),
        (
            {
                "design.collar.component.lapel_standing.v": False,
                "design.collar.width.v": 0.70,
            },
            "wide fold-over collar framing a shallow V-shaped neckline",
        ),
        (
            {
                "design.collar.component.lapel_standing.v": False,
                "design.collar.component.depth.v": 3.00,
            },
            "soft fold-over V-neck collar",
        ),
        (
            {
                "design.collar.component.lapel_standing.v": True,
                FRONT_NECKLINE_PATH: "SquareNeckHalf",
            },
            "split stand collar",
        ),
    ),
)
@pytest.mark.parametrize("direction", ("add", "remove"))
def test_collar_transition_preserves_the_prompt_visible_subtype(
    parameters: dict[str, object],
    subtype: str,
    direction: str,
) -> None:
    source = shirt_design(
        **{
            **parameters,
            COMPONENT_STYLE_PATH: ("SimpleLapel" if direction == "remove" else None),
        }
    )
    (candidate,) = RULES["collar_presence"].candidates(context_for(source))
    transition = candidate.transitions[0]

    assert candidate.operation.direction == direction
    if direction == "add":
        assert transition.source_state.phrase == "collarless"
        assert transition.target_state.phrase == f"having a {subtype}"
    else:
        assert transition.source_state.phrase == f"having a {subtype}"
        assert transition.target_state.phrase == "collarless"


@pytest.mark.parametrize(
    ("length", "direction", "candidate_count", "status", "reason"),
    (
        (
            HOOD_ADD_MAX_HOOD_LENGTH,
            "add",
            1,
            "passed",
            HOOD_ADD_ACCEPT_REASON,
        ),
        (
            HOOD_ADD_MAX_HOOD_LENGTH + 1.0e-9,
            "add",
            0,
            None,
            HOOD_ADD_REJECTION_REASON,
        ),
        (
            HOOD_ADD_MAX_HOOD_LENGTH + 10.0,
            "remove",
            1,
            "not_applicable",
            HOOD_REMOVE_GUARDRAIL_REASON,
        ),
    ),
)
def test_hood_add_boundary_and_unrestricted_remove_are_publicly_observable(
    length: float,
    direction: str,
    candidate_count: int,
    status: str | None,
    reason: str,
) -> None:
    design = shirt_design(
        **{
            HOOD_LENGTH_PATH: length,
            COMPONENT_STYLE_PATH: ("Hood2Panels" if direction == "remove" else None),
        }
    )
    candidates = RULES["hood_presence"].candidates(context_for(design))

    assert len(candidates) == candidate_count
    if not candidates:
        assert reason == HOOD_ADD_REJECTION_REASON
        return
    guardrail = candidates[0].applicability_evidence["pilot_guardrail"]
    assert guardrail["id"] == HOOD_ADD_GUARDRAIL_ID
    assert guardrail["direction"] == direction
    assert guardrail["parameter_path"] == HOOD_LENGTH_PATH
    assert guardrail["threshold"] == HOOD_ADD_MAX_HOOD_LENGTH
    assert guardrail["threshold_inclusive"] is True
    assert guardrail["observed_value"] == length
    assert guardrail["status"] == status
    assert guardrail["reason_code"] == reason


@pytest.mark.parametrize(
    ("rule_id", "updates"),
    (
        ("hood_presence", {FRONT_NECKLINE_PATH: "VNeckHalf"}),
        ("hood_presence", {BACK_NECKLINE_PATH: "VNeckHalf"}),
        ("hood_presence", {COMPONENT_STYLE_PATH: "SimpleLapel"}),
        ("hood_presence", {COMPONENT_STYLE_PATH: "Turtle"}),
        ("hood_presence", {"design.left.enable_asym.v": True}),
        ("hood_presence", {"design.meta.upper.v": None}),
        (
            "hood_presence",
            {
                "design.meta.upper.v": "FittedShirt",
                "design.shirt.strapless.v": True,
            },
        ),
        ("collar_presence", {BACK_NECKLINE_PATH: "VNeckHalf"}),
        ("collar_presence", {COMPONENT_STYLE_PATH: "Hood2Panels"}),
        ("collar_presence", {COMPONENT_STYLE_PATH: "Turtle"}),
        ("collar_presence", {"design.left.enable_asym.v": True}),
        ("collar_presence", {"design.meta.upper.v": None}),
        (
            "collar_presence",
            {
                "design.meta.upper.v": "FittedShirt",
                "design.shirt.strapless.v": True,
            },
        ),
        (
            "collar_presence",
            {
                COMPONENT_STYLE_PATH: "SimpleLapel",
                "design.collar.component.lapel_standing.v": False,
                "design.collar.component.depth.v": 2.0,
                "design.collar.width.v": 0.2,
                "design.collar.fc_depth.v": 0.4,
            },
        ),
    ),
)
def test_presence_rules_reject_out_of_scope_constructor_states(
    rule_id: str,
    updates: dict[str, object],
) -> None:
    design = shirt_design(**updates)
    assert RULES[rule_id].candidates(context_for(design)) == ()


@pytest.mark.parametrize("rule_id", ("hood_presence", "collar_presence"))
def test_presence_rule_rejects_a_prompt_state_that_disagrees_with_source(
    rule_id: str,
) -> None:
    if rule_id == "hood_presence":
        design = shirt_design()
        wrong = SemanticState(
            "hood_presence",
            "hooded",
            "hooded",
            {"present": True, "protected_phrase": "hood"},
        )
        states = {**prompt_states_for(design), rule_id: wrong}
    else:
        design = explicit_collar_design(present=True)
        wrong = SemanticState(
            "collar_presence",
            "collared",
            "having a soft fold-over V-neck collar",
            {
                "present": True,
                "component_style": "SimpleLapel",
                "subtype_phrase": "soft fold-over V-neck collar",
            },
        )
        states = {**prompt_states_for(design), rule_id: wrong}

    with pytest.raises(ProtectedStateSchemaError):
        RULES[rule_id].candidates(context_for(design, prompt_states=states))


@pytest.mark.parametrize("rule_id", ("hood_presence", "collar_presence"))
@pytest.mark.parametrize("direction", ("add", "remove"))
def test_membership_effect_accepts_exact_add_and_remove_topology(
    rule_id: str,
    direction: str,
) -> None:
    _, candidate = presence_candidate(rule_id, direction)
    absent, present = (
        hood_effect_pair() if rule_id == "hood_presence" else collar_effect_pair()
    )
    source, target = (absent, present) if direction == "add" else (present, absent)

    result = RULES[rule_id].validate({}, {}, {}, source, target, candidate, CONFIG)

    assert result.passed
    assert result.stage == "effect"
    assert result.code == "effect/passed"
    assert result.evidence["rule_id"] == rule_id
    details = result.evidence["details"]
    if rule_id == "hood_presence":
        observed = details["v4_effect_result"]
        assert observed["passed"] is True
        assert observed["failure"] is None
        assert observed["evidence"]["expected_direction"] == direction
        delta = observed["evidence"]["delta"]
        assert set(delta["added_panels"] or delta["removed_panels"]) == {
            "left_hood",
            "right_hood",
        }
    else:
        assert details["membership_ok"] is True
        assert details["localized_stitch_change"] is True
        assert details["stitch_direction_ok"] is True
        assert all(details["collar_stitch_membership"].values())
        assert all(details["collar_panel_attachments"].values())


@pytest.mark.parametrize("rule_id", ("hood_presence", "collar_presence"))
def test_strict_topology_rejects_but_local_policy_accepts_geometry_only_drift(
    rule_id: str,
) -> None:
    _, candidate = presence_candidate(rule_id, "add")
    source, target = (
        hood_effect_pair() if rule_id == "hood_presence" else collar_effect_pair()
    )
    target["pattern"]["panels"]["pocket"]["vertices"][0][0] += 0.25

    strict = RULES[rule_id].validate(
        {}, {}, {}, source, target, candidate, CONFIG, topology_policy="strict"
    )
    local = RULES[rule_id].validate(
        {}, {}, {}, source, target, candidate, CONFIG, topology_policy="local"
    )

    assert not strict.passed
    assert strict.code == "effect/topology_mismatch"
    assert local.passed
    assert local.code == "effect/passed"
    assert local.evidence["details"]["topology_policy"] == "local"


@pytest.mark.parametrize(
    ("rule_id", "mutation"),
    (
        ("hood_presence", "missing_component_panel"),
        ("hood_presence", "missing_component_stitches"),
        ("hood_presence", "unrelated_panel_membership"),
        ("collar_presence", "missing_attachment"),
        ("collar_presence", "cross_side_attachment"),
        ("collar_presence", "unrelated_panel_membership"),
    ),
)
def test_membership_effect_rejects_wrong_or_nonlocal_topology(
    rule_id: str,
    mutation: str,
) -> None:
    _, candidate = presence_candidate(rule_id, "add")
    source, target = (
        hood_effect_pair() if rule_id == "hood_presence" else collar_effect_pair()
    )
    if mutation == "missing_component_panel":
        target["pattern"]["panels"].pop("right_hood")
        target["pattern"]["panel_order"].remove("right_hood")
        target["pattern"]["stitches"] = [
            item
            for item in target["pattern"]["stitches"]
            if all(endpoint["panel"] != "right_hood" for endpoint in item[:2])
        ]
    elif mutation == "missing_component_stitches":
        target["pattern"]["stitches"] = []
    elif mutation == "missing_attachment":
        target["pattern"]["stitches"].pop()
    elif mutation == "cross_side_attachment":
        target["pattern"]["stitches"] = [
            stitch("right_ftorso", 2, "left_collar_front", 0),
            stitch("left_ftorso", 2, "right_collar_front", 0),
            stitch("right_btorso", 2, "left_collar_back", 0),
            stitch("left_btorso", 2, "right_collar_back", 0),
        ]
    else:
        target["pattern"]["panels"]["unexpected"] = pattern_panel()
        target["pattern"]["panel_order"].append("unexpected")

    result = RULES[rule_id].validate({}, {}, {}, source, target, candidate, CONFIG)

    assert not result.passed
    assert result.code != "effect/passed"
    assert result.evidence["rule_id"] == rule_id
