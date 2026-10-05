"""Behavior tests for the public Godet insert-count rule."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from editing.canonical_io import canonical_hash, design_dotted_get
from editing.semantic_edit.registry import build_semantic_registry, load_semantic_config
from editing.semantic_edit.rules.godet import (
    GODET_COUNT_PATH,
    GODET_INSERT_DEPTH_PATH,
    GODET_INSERT_WIDTH_PATH,
)
from editing.tests.support import ROOT, make_design, make_rule_context


CONFIG = load_semantic_config(ROOT / "editing/configs/semantic_edit.yaml")
RULE = build_semantic_registry(CONFIG)["godet_insert_count"]


def _design(
    count: object = 8,
    *,
    bottom: object = "GodetSkirt",
    base: object = "Skirt2",
    insert_width: object = 15,
    insert_depth: object = 20,
) -> dict:
    return make_design(
        {
            "design.meta.upper.v": None,
            "design.meta.bottom.v": bottom,
            "design.meta.wb.v": None,
            "design.godet-skirt.base.v": base,
            GODET_COUNT_PATH: count,
            GODET_INSERT_WIDTH_PATH: insert_width,
            GODET_INSERT_DEPTH_PATH: insert_depth,
        }
    )


def _context(
    count: object = 8,
    *,
    bottom: object = "GodetSkirt",
    interface_count: object | None = 2,
    insert_width: object = 15,
    insert_depth: object = 20,
):
    facts = (
        {}
        if interface_count is None
        else {"godet_base_bottom_interface_count": interface_count}
    )
    return make_rule_context(
        _design(
            count,
            bottom=bottom,
            insert_width=insert_width,
            insert_depth=insert_depth,
        ),
        prompt_classifier=lambda _design, _body: {},
        constructor_facts=facts,
        prompt_states={},
    )


def _edge(first: int, second: int, label: str | None = None) -> dict[str, Any]:
    edge: dict[str, Any] = {"endpoints": [first, second]}
    if label is not None:
        edge["label"] = label
    return edge


def _panel(
    *,
    vertices: list[list[float]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    label: str = "body",
) -> dict[str, Any]:
    return {
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "vertices": vertices or [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        "edges": edges
        or [
            _edge(0, 1, "bottom"),
            _edge(1, 2, "side"),
            _edge(2, 3, "top"),
            _edge(3, 0, "center"),
        ],
        "label": label,
    }


def _stitch(
    left_panel: str,
    left_edge: int,
    right_panel: str,
    right_edge: int,
) -> list[dict[str, Any]]:
    return [
        {"panel": left_panel, "edge": left_edge},
        {"panel": right_panel, "edge": right_edge},
    ]


def _specification(count: int, retained_width: float = 1.0) -> dict[str, Any]:
    panels: dict[str, dict[str, Any]] = {"waistband": _panel()}
    stitches: list[list[dict[str, Any]]] = []
    half_counts = {"front": (count + 1) // 2, "back": count // 2}

    for half, insert_count in half_counts.items():
        width = 20.0
        cut_width = (width - retained_width * (insert_count + 1)) / insert_count
        vertices = [[0.0, 0.0]]
        edges: list[dict[str, Any]] = []
        cut_edge_pairs: list[tuple[int, int]] = []
        cursor = 0.0
        for _ in range(insert_count):
            if retained_width:
                cursor += retained_width
                vertices.append([cursor, 0.0])
                edges.append(_edge(len(vertices) - 2, len(vertices) - 1))
            vertices.append([cursor + cut_width / 2.0, 2.0])
            first_cut = len(edges)
            edges.append(_edge(len(vertices) - 2, len(vertices) - 1))
            cursor += cut_width
            vertices.append([cursor, 0.0])
            second_cut = len(edges)
            edges.append(_edge(len(vertices) - 2, len(vertices) - 1))
            cut_edge_pairs.append((first_cut, second_cut))
        if retained_width:
            vertices.append([width, 0.0])
            edges.append(_edge(len(vertices) - 2, len(vertices) - 1))
        vertices.extend([[width, 8.0], [0.0, 8.0]])
        edges.extend(
            [
                _edge(len(vertices) - 3, len(vertices) - 2, f"{half}_side"),
                _edge(len(vertices) - 2, len(vertices) - 1, f"{half}_waist"),
                _edge(len(vertices) - 1, 0, f"{half}_side"),
            ]
        )
        base_name = f"skirt_{half}"
        panels[base_name] = _panel(vertices=vertices, edges=edges, label="leg")

        for index, (first_cut, second_cut) in enumerate(cut_edge_pairs):
            insert_name = f"ins_skirt_{half}_{index}"
            panels[insert_name] = _panel(
                vertices=[[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]],
                edges=[_edge(0, 1), _edge(1, 2), _edge(2, 0)],
                label="leg",
            )
            insert_edges = (1, 2) if half == "front" else (0, 1)
            stitches.extend(
                [
                    _stitch(insert_name, insert_edges[0], base_name, first_cut),
                    _stitch(insert_name, insert_edges[1], base_name, second_cut),
                ]
            )

    return {
        "pattern": {
            "panels": panels,
            "stitches": stitches,
            "panel_order": list(panels),
        },
        "parameters": {},
        "parameter_order": [],
        "properties": {"units_in_meter": 100.0},
    }


def _case(target_count: int, retained_width: float = 1.0):
    context = _context(8)
    candidate = next(
        item
        for item in RULE.candidates(context)
        if item.operation.after[GODET_COUNT_PATH] == target_count
    )
    target_design = RULE.apply(context.design, candidate)
    return (
        context,
        candidate,
        target_design,
        _specification(8, retained_width),
        _specification(target_count, retained_width),
    )


def _decrease_case():
    return _case(4)


def _validate_case(
    source_design: dict,
    target_design: dict,
    source_specification: dict,
    target_specification: dict,
    candidate,
    *,
    topology_policy: str = "strict",
):
    return RULE.validate(
        source_design,
        target_design,
        {},
        source_specification,
        target_specification,
        candidate,
        CONFIG,
        topology_policy=topology_policy,
    )


@pytest.mark.parametrize(
    ("source_count", "interface_count", "expected_targets"),
    (
        (4, 2, {10, 12}),
        (6, 2, {10, 12}),
        (8, 2, {4, 12}),
        (10, 2, {4}),
        (12, 2, {4, 6, 8}),
        (4, 3, set()),
    ),
)
def test_godet_candidates_obey_interface_divisibility_and_safety_limits(
    source_count: int,
    interface_count: int,
    expected_targets: set[int],
) -> None:
    candidates = RULE.candidates(
        _context(source_count, interface_count=interface_count)
    )

    assert {
        item.operation.after[GODET_COUNT_PATH] for item in candidates
    } == expected_targets
    assert all(
        abs(item.operation.after[GODET_COUNT_PATH] - source_count) >= 4
        for item in candidates
    )


def test_godet_candidate_complexity_limit_excludes_only_the_oversized_target() -> None:
    at_limit = RULE.candidates(_context(8, insert_width=50, insert_depth=20))
    over_limit = RULE.candidates(_context(8, insert_width=50, insert_depth=21))

    assert {item.operation.after[GODET_COUNT_PATH] for item in at_limit} == {4, 12}
    assert {item.operation.after[GODET_COUNT_PATH] for item in over_limit} == {4}


@pytest.mark.parametrize(
    "context",
    (
        pytest.param(_context(8, bottom="Skirt2"), id="different-bottom"),
        pytest.param(_context(8, interface_count=None), id="missing-constructor-fact"),
        pytest.param(_context(5), id="unsupported-source-count"),
    ),
)
def test_godet_is_inapplicable_without_an_active_supported_source(context) -> None:
    assert RULE.candidates(context) == ()


def test_godet_apply_changes_only_the_selected_count() -> None:
    context, candidate, target, *_ = _decrease_case()
    source_hash = canonical_hash(context.design)

    assert design_dotted_get(target, GODET_COUNT_PATH) == 4
    assert canonical_hash(context.design) == source_hash
    assert candidate.operation.direction == "decrease"
    assert candidate.operation.before == {GODET_COUNT_PATH: 8}
    assert candidate.operation.after == {GODET_COUNT_PATH: 4}


@pytest.mark.parametrize("target_count", (4, 12))
def test_godet_validate_accepts_real_attached_insert_topology_with_evidence(
    target_count: int,
) -> None:
    context, candidate, target_design, source, target = _case(target_count)

    result = _validate_case(
        context.design,
        target_design,
        source,
        target,
        candidate,
    )

    assert result.passed
    assert result.code == "effect/passed"
    details = result.evidence["details"]
    assert details["source_insert_panel_count"] == 8
    assert details["target_insert_panel_count"] == target_count
    assert details["insert_panel_delta"] == 4
    assert details["declared_count_delta"] == 4
    assert details["insert_attachments_consistent"] is True
    assert details["direction_ok"] is True


@pytest.mark.parametrize("target_count", (4, 12))
def test_godet_validate_accepts_adjacent_insert_cuts_without_retained_edges(
    target_count: int,
) -> None:
    context, candidate, target_design, source, target = _case(
        target_count,
        retained_width=0.0,
    )

    result = _validate_case(
        context.design,
        target_design,
        source,
        target,
        candidate,
    )

    assert result.passed
    assert result.evidence["details"]["insert_attachments_consistent"] is True


@pytest.mark.parametrize(
    ("mutation", "failed_evidence"),
    (
        ("declared-count", "declared_count_delta"),
        ("base-dispatch", "base_dispatch_unchanged"),
        ("detached-insert", "insert_attachments_consistent"),
        ("base-geometry", "base_geometry_unchanged_except_insert_cut_edges"),
        ("nonbase-topology", "nonbase_panel_topology_unchanged"),
        ("noninsert-stitch", "noninsert_stitches_unchanged"),
    ),
)
def test_godet_validate_rejects_critical_topology_mutations_with_evidence(
    mutation: str,
    failed_evidence: str,
) -> None:
    context, candidate, target_design, source, target = _decrease_case()
    target_design = deepcopy(target_design)
    target = deepcopy(target)

    if mutation == "declared-count":
        target_design["godet-skirt"]["num_inserts"]["v"] = 6
    elif mutation == "base-dispatch":
        target_design["godet-skirt"]["base"]["v"] = "PencilSkirt"
    elif mutation == "detached-insert":
        target["pattern"]["stitches"].pop()
    elif mutation == "base-geometry":
        target["pattern"]["panels"]["skirt_front"]["rotation"][0] = 1.0
    elif mutation == "nonbase-topology":
        target["pattern"]["panels"]["waistband"]["edges"][0]["label"] = "changed"
    elif mutation == "noninsert-stitch":
        target["pattern"]["stitches"].append(_stitch("waistband", 0, "skirt_front", 0))

    result = _validate_case(
        context.design,
        target_design,
        source,
        target,
        candidate,
    )

    assert not result.passed
    assert result.code == (
        "effect/measurement_error"
        if mutation == "detached-insert"
        else "effect/topology_mismatch"
    )
    details = result.evidence["details"]
    if mutation == "detached-insert":
        assert "Godet cut edges" in details["message"]
    elif mutation == "declared-count":
        assert details[failed_evidence] == 2
    else:
        assert details[failed_evidence] is False


def test_godet_local_policy_allows_coordinate_rebuild_but_not_topology_drift() -> None:
    context, candidate, target_design, source, target = _decrease_case()
    for specification in (source, target):
        specification["pattern"]["panels"]["skirt_pocket"] = _panel()
        specification["pattern"]["panel_order"].append("skirt_pocket")
    target["pattern"]["panels"]["skirt_pocket"]["vertices"][0][0] += 0.5
    target["pattern"]["panels"]["skirt_front"]["vertices"][-2][1] += 0.5

    strict = _validate_case(
        context.design,
        target_design,
        source,
        target,
        candidate,
    )
    local = _validate_case(
        context.design,
        target_design,
        source,
        target,
        candidate,
        topology_policy="local",
    )

    assert not strict.passed
    assert local.passed
    assert local.evidence["details"]["nonbase_panel_topology_unchanged"] is True

    changed_topology = deepcopy(target)
    changed_topology["pattern"]["panels"]["skirt_pocket"]["edges"][0][
        "label"
    ] = "changed"
    rejected = _validate_case(
        context.design,
        target_design,
        source,
        changed_topology,
        candidate,
        topology_policy="local",
    )

    assert not rejected.passed
    assert rejected.code == "effect/topology_mismatch"
    assert rejected.evidence["details"]["nonbase_panel_topology_unchanged"] is False
