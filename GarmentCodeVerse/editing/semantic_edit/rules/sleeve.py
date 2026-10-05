"""Self-contained symmetric sleeve-presence rule."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any

from editing.semantic_edit.prompt_state import ActivePathError, resolve_active_paths
from editing.canonical_io import (
    canonical_hash,
    canonical_json_bytes,
    design_dotted_get,
    design_dotted_set,
    plain_copy,
)
from editing.config import EditConfig, RuleConfig
from editing.models import (
    EffectResult,
    Operation,
    PlannedOperation,
    Rule,
    RuleContext,
    SemanticState,
    SemanticTransition,
    StateUnavailableError,
    TopologyKind,
    empty_conflict_evidence,
    rule_effect_result,
)
from editing.pattern_compare import edge_type
from editing.pattern_geometry import (
    PatternGeometryError,
    edge_topology,
    specification_metadata,
    stitch_records,
)

SLEEVELESS_PATH = "design.sleeve.sleeveless.v"
SLEEVE_ADD_MAX_LENGTH = 0.95
SLEEVE_ADD_MAX_CONNECTING_WIDTH = 1.20
SLEEVE_ADD_GUARDRAIL_SCHEMA_VERSION = "gcd-sleeve-presence-add-pilot-guardrail-v1"
SLEEVE_PRESENCE_READS = (
    "design.meta.upper.v",
    "design.left.enable_asym.v",
    "design.shirt.strapless.v",
    "design.shirt.width.v",
    "design.shirt.flare.v",
    "design.sleeve.sleeveless.v",
    "design.sleeve.armhole_shape.v",
    "design.sleeve.cuff.type.v",
    "design.sleeve.length.v",
    "design.sleeve.connecting_width.v",
)

_SLEEVE_CONFLICTS = ("sleeve_length",)
_SLEEVE_PREREQUISITES = (
    "curved_armhole",
    "null_cuff",
    "symmetric_upper",
)

SLEEVE_PANEL = re.compile(r"^(left|right)_sleeve_[fb]$")
_SLEEVE_CUFF_PANEL = re.compile(r"^sl_(left|right)_cuff(?:_skirt)?_[fb]$")
_TORSO_PANELS = (
    "left_ftorso",
    "right_ftorso",
    "left_btorso",
    "right_btorso",
)
_GEOMETRY_ATOL = 1.0e-7


class SleeveEffectError(ValueError):
    """A constructed pattern cannot support sleeve validation."""


def _unavailable(code: str, message: str) -> StateUnavailableError:
    return StateUnavailableError(code, message)


def _state_value(design: Mapping[str, Any], path: str) -> Any:
    try:
        return design_dotted_get(design, path)
    except (KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "state_schema_unavailable",
            f"active state path is missing or invalid: {path}",
        ) from exc


def _state_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise _unavailable("state_schema_unavailable", f"{field_name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _unavailable("state_schema_unavailable", f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise _unavailable("state_schema_unavailable", f"{field_name} must be finite")
    return result


def classify_sleeve_state(
    design: Mapping[str, Any],
    body: Mapping[str, Any],
) -> SemanticState:
    """Classify global sleeve presence without inventing side states."""

    try:
        active = resolve_active_paths(design)
    except (ActivePathError, KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "active_path_unavailable",
            f"cannot resolve active constructor paths: {exc}",
        ) from exc
    upper = active.dispatch.get("upper")
    if upper is None:
        raise _unavailable(
            "sleeve_presence_inapplicable",
            "sleeve presence requires an active upper garment",
        )
    if upper == "FittedShirt" and bool(
        _state_value(design, "design.shirt.strapless.v")
    ):
        raise _unavailable(
            "strapless_sleeve_path_suppressed",
            "strapless construction suppresses the sleeve component",
        )
    if active.dispatch.get("asymmetric") is True:
        raise _unavailable(
            "asymmetric_sleeve_presence_unrepresented",
            "asymmetric construction has no approved global sleeve-presence state",
        )
    right = active.dispatch.get("right_sleeveless")
    if (
        upper == "Shirt"
        and right is True
        and _state_number(
            _state_value(design, "design.shirt.width.v"),
            "design.shirt.width.v",
        )
        >= 1.20
        and _state_number(
            _state_value(design, "design.shirt.flare.v"),
            "design.shirt.flare.v",
        )
        >= 1.45
    ):
        raise _unavailable(
            "cold_shoulder_sleeve_state_suppressed",
            "cold-shoulder cape-sleeve construction is not a sleeveless state",
        )
    if not isinstance(right, bool):
        raise _unavailable(
            "sleeve_presence_unavailable",
            "the active sleeve presence flag is unavailable",
        )
    if right:
        return SemanticState(
            rule_id="sleeve_presence",
            state_id="sleeveless",
            phrase="sleeveless",
            attributes={"present": False, "channels": ("left", "right")},
        )

    try:
        from .length import LengthCategoryError, classify_target

        observation = classify_target("sleeve_length", design, body, scope="both")
    except (LengthCategoryError, ValueError, TypeError) as exc:
        raise _unavailable(
            "sleeve_length_state_unavailable",
            f"active sleeve length cannot be classified: {exc}",
        ) from exc
    categories = tuple(channel.category for channel in observation.channels)
    if len(set(categories)) != 1:
        raise _unavailable(
            "sleeve_length_state_unrepresented",
            "global sleeve presence requires one shared sleeve-length category",
        )
    category = categories[0]
    return SemanticState(
        rule_id="sleeve_presence",
        state_id=category,
        phrase=f"having {category} sleeves",
        attributes={
            "present": True,
            "channels": tuple(channel.name for channel in observation.channels),
            "length_categories": categories,
        },
    )


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _effect_coordinate(value: Any, path: str) -> float:
    """Read one constructed-pattern coordinate as a finite number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SleeveEffectError(f"{path} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise SleeveEffectError(f"{path} must be finite") from exc
    if not math.isfinite(result):
        raise SleeveEffectError(f"{path} must be finite")
    return result


def evaluate_sleeve_presence_pilot_guardrail(
    *,
    direction: str,
    strapless: Any,
    sleeve_length: Any,
    connecting_width: Any,
) -> dict[str, Any]:
    """Return complete evidence for the pilot-derived sleeve-add envelope."""

    if direction not in {"add", "remove"}:
        raise ValueError("direction must be add or remove")
    if type(strapless) is not bool:
        raise ValueError("design.shirt.strapless.v must be bool")
    length = _finite_number(
        sleeve_length,
        "design.sleeve.length.v",
    )
    width = _finite_number(
        connecting_width,
        "design.sleeve.connecting_width.v",
    )
    observed = {
        "design.shirt.strapless.v": strapless,
        "design.sleeve.length.v": length,
        "design.sleeve.connecting_width.v": width,
    }
    limits = {
        "design.shirt.strapless.v": False,
        "design.sleeve.length.v": SLEEVE_ADD_MAX_LENGTH,
        "design.sleeve.connecting_width.v": (SLEEVE_ADD_MAX_CONNECTING_WIDTH),
    }
    if direction == "remove":
        return {
            "schema_version": SLEEVE_ADD_GUARDRAIL_SCHEMA_VERSION,
            "direction": direction,
            "applied": False,
            "passed": True,
            "code": "sleeve_add_guardrail/not_applicable_remove",
            "reason": (
                "pilot-derived sleeve-add guardrail does not apply " "to removal"
            ),
            "observed": observed,
            "limits": limits,
            "violations": (),
        }

    violations: list[dict[str, Any]] = []
    if strapless:
        violations.append(
            {
                "code": "sleeve_add_guardrail/strapless_source",
                "path": "design.shirt.strapless.v",
                "observed": strapless,
                "allowed_operator": "==",
                "limit": False,
            }
        )
    if length > SLEEVE_ADD_MAX_LENGTH:
        violations.append(
            {
                "code": "sleeve_add_guardrail/length_above_max",
                "path": "design.sleeve.length.v",
                "observed": length,
                "allowed_operator": "<=",
                "limit": SLEEVE_ADD_MAX_LENGTH,
            }
        )
    if width > SLEEVE_ADD_MAX_CONNECTING_WIDTH:
        violations.append(
            {
                "code": ("sleeve_add_guardrail/connecting_width_above_max"),
                "path": "design.sleeve.connecting_width.v",
                "observed": width,
                "allowed_operator": "<=",
                "limit": SLEEVE_ADD_MAX_CONNECTING_WIDTH,
            }
        )
    violation_codes = [item["code"] for item in violations]
    passed = not violations
    return {
        "schema_version": SLEEVE_ADD_GUARDRAIL_SCHEMA_VERSION,
        "direction": direction,
        "applied": True,
        "passed": passed,
        "code": (
            "sleeve_add_guardrail/pass"
            if passed
            else "sleeve_add_guardrail/rejected"
        ),
        "reason": (
            "source design is inside the pilot-derived sleeve-add " "envelope"
            if passed
            else "sleeve add rejected: " + ", ".join(violation_codes)
        ),
        "observed": observed,
        "limits": limits,
        "violations": tuple(violations),
    }


def _validate_config(config: RuleConfig) -> None:
    if not isinstance(config, RuleConfig):
        raise TypeError("rule_config must be a RuleConfig")
    if type(config.enabled) is not bool or config.enabled is not True:
        raise ValueError("sleeve_presence rule_config must be enabled")
    if (
        config.semantic_group != "sleeve"
        or type(config.structural_level) is not int
        or config.structural_level != 2
    ):
        raise ValueError("sleeve_presence rule_config has an invalid semantic contract")
    if (
        config.effect_metric != "sleeve_armhole_membership"
        or config.effect_contract != "exact_membership_change"
    ):
        raise ValueError("sleeve_presence rule_config has an invalid effect contract")
    if set(config.parameter_specs) != {"sleeveless"}:
        raise ValueError("sleeve_presence rule_config must contain only sleeveless")
    spec = config.parameter_specs["sleeveless"]
    if (
        spec.edit_mode != "toggle"
        or spec.value_type != "bool"
        or not isinstance(spec.transitions, tuple)
        or len(spec.transitions) != 2
        or type(spec.transitions[0]) is not bool
        or type(spec.transitions[1]) is not bool
        or spec.transitions[0] is not False
        or spec.transitions[1] is not True
    ):
        raise ValueError("sleeve_presence config must use the exact false/true toggle")


def _candidate(
    context: RuleContext,
    rule_config: RuleConfig,
) -> PlannedOperation | None:
    if context.active_paths.design_hash != canonical_hash(context.design):
        return None
    upper = context.active_paths.dispatch.get("upper")
    if upper not in {"Shirt", "FittedShirt"}:
        return None
    if context.active_paths.dispatch.get("asymmetric") is not False:
        return None

    main_active = {
        SLEEVELESS_PATH,
        "design.sleeve.armhole_shape.v",
        "design.sleeve.connecting_width.v",
    }
    if not main_active.issubset(context.active_paths.paths):
        return None
    if any(path.startswith("design.left.sleeve.") for path in context.active_paths.paths):
        return None

    read_values = {
        path: design_dotted_get(context.design, path)
        for path in SLEEVE_PRESENCE_READS
    }
    if read_values["design.meta.upper.v"] != upper:
        return None
    if read_values["design.left.enable_asym.v"] is not False:
        return None
    strapless = read_values["design.shirt.strapless.v"]
    if type(strapless) is not bool:
        return None
    if upper == "FittedShirt" and strapless is not False:
        return None
    sleeveless = read_values[SLEEVELESS_PATH]
    if not isinstance(sleeveless, bool):
        return None
    if context.active_paths.dispatch.get("right_sleeveless") is not sleeveless:
        return None
    if read_values["design.sleeve.armhole_shape.v"] != "ArmholeCurve":
        return None
    if read_values["design.sleeve.cuff.type.v"] is not None:
        return None
    sleeve_length = _finite_number(
        read_values["design.sleeve.length.v"], "design.sleeve.length.v"
    )
    connecting_width = _finite_number(
        read_values["design.sleeve.connecting_width.v"],
        "design.sleeve.connecting_width.v",
    )
    direction = "add" if sleeveless else "remove"
    pilot_guardrail = evaluate_sleeve_presence_pilot_guardrail(
        direction=direction,
        strapless=strapless,
        sleeve_length=sleeve_length,
        connecting_width=connecting_width,
    )
    if not pilot_guardrail["passed"]:
        return None

    source_state = classify_sleeve_state(context.design, context.body)
    target_value = not sleeveless
    target_design = plain_copy(
        design_dotted_set(context.design, SLEEVELESS_PATH, target_value),
        tuples_as_lists=True,
    )
    target_state = classify_sleeve_state(target_design, context.body)
    for state in (source_state, target_state):
        if tuple(state.attributes.get("channels", ())) != ("left", "right"):
            return None
    if source_state.state_id == target_state.state_id:
        return None
    if direction == "add":
        if source_state.state_id != "sleeveless" or target_state.state_id == "sleeveless":
            return None
    elif source_state.state_id == "sleeveless" or target_state.state_id != "sleeveless":
        return None

    transition = SemanticTransition(
        rule_id="sleeve_presence",
        scope="both",
        channels=("left", "right"),
        feature_phrase="the top",
        source_state=source_state,
        target_state=target_state,
        direction=direction,
    )
    operation = Operation(
        rule_id="sleeve_presence",
        semantic_group=rule_config.semantic_group,
        scope="both",
        direction=direction,
        magnitude="structural",
        structural_level=rule_config.structural_level,
        reads=tuple(sorted(SLEEVE_PRESENCE_READS)),
        writes=(SLEEVELESS_PATH,),
        conflicts=tuple(sorted(_SLEEVE_CONFLICTS)),
        prerequisites=tuple(sorted(_SLEEVE_PREREQUISITES)),
        edit_mode=rule_config.parameter_specs["sleeveless"].edit_mode,
        effect_metric_id=rule_config.effect_metric,
        effect_contract_id=rule_config.effect_contract,
        before={SLEEVELESS_PATH: sleeveless},
        after={SLEEVELESS_PATH: target_value},
        relative_change=None,
        requested_magnitude="structural",
        realized_magnitude="structural",
        clamped=False,
        instruction_tokens={
            "feature": "the top",
            "source_state_id": source_state.state_id,
            "target_state_id": target_state.state_id,
        },
    )
    return PlannedOperation(
        operation=operation,
        transitions=(transition,),
        topology_kind=TopologyKind.MEMBERSHIP,
        applicability_evidence={
            "rule_id": "sleeve_presence",
            "source_design_hash": canonical_hash(context.design),
            "target_design_hash": canonical_hash(target_design),
            "body_hash": canonical_hash(context.body),
            "active_paths_design_hash": context.active_paths.design_hash,
            "read_contract_hash": canonical_hash(read_values),
            "source_local_state": source_state.to_payload(),
            "target_local_state": target_state.to_payload(),
            "symmetric_channels": ["left", "right"],
            "armhole_shape": "ArmholeCurve",
            "cuff_type": None,
            "pilot_guardrail": pilot_guardrail,
        },
        conflict_evidence=empty_conflict_evidence(
            "sleeve_presence", _SLEEVE_CONFLICTS
        ),
    )


def _candidates(
    context: RuleContext,
    rule_config: RuleConfig,
) -> tuple[PlannedOperation, ...]:
    try:
        candidate = _candidate(context, rule_config)
    except (KeyError, TypeError, ValueError, StateUnavailableError):
        return ()
    return () if candidate is None else (candidate,)


def apply(design: dict[str, Any], operation: PlannedOperation) -> dict[str, Any]:
    """Apply the one declared sleeve-presence toggle."""

    return plain_copy(
        design_dotted_set(
            design,
            SLEEVELESS_PATH,
            operation.operation.after[SLEEVELESS_PATH],
        ),
        tuples_as_lists=True,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SleeveEffectError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SleeveEffectError(f"{name} must be a sequence")
    return value


def _panels(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return _mapping(specification["pattern"]["panels"], "pattern.panels")
    except KeyError as exc:
        raise SleeveEffectError("pattern.panels is required") from exc


def _edge(
    specification: Mapping[str, Any], panel_name: str, edge_index: int
) -> Mapping[str, Any]:
    panel = _mapping(_panels(specification).get(panel_name), panel_name)
    edges = _sequence(panel.get("edges"), f"{panel_name}.edges")
    if (
        isinstance(edge_index, bool)
        or not isinstance(edge_index, int)
        or edge_index < 0
        or edge_index >= len(edges)
    ):
        raise SleeveEffectError(f"invalid edge reference {panel_name}[{edge_index!r}]")
    return _mapping(edges[edge_index], f"{panel_name}.edges[{edge_index}]")


def _edge_label(edge: Mapping[str, Any]) -> str | None:
    label = edge.get("label")
    if label is not None and (type(label) is not str or not label):
        raise SleeveEffectError("edge labels must be non-empty strings")
    return label


def _stitch_records(
    specification: Mapping[str, Any],
) -> tuple[tuple[tuple[str, int], tuple[str, int], tuple[Any, ...]], ...]:
    try:
        return stitch_records(specification)
    except PatternGeometryError as exc:
        raise SleeveEffectError(str(exc)) from exc


def _edge_topology(panel: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return edge_topology(panel)
    except PatternGeometryError as exc:
        raise SleeveEffectError(str(exc)) from exc


def _panel_vertices(panel: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    result = []
    for raw in _sequence(panel.get("vertices"), "panel.vertices"):
        point = _sequence(raw, "panel vertex")
        if len(point) != 2:
            raise SleeveEffectError("panel vertices must be 2-D")
        result.append(
            (
                _effect_coordinate(point[0], "vertex.x"),
                _effect_coordinate(point[1], "vertex.y"),
            )
        )
    if not result:
        raise SleeveEffectError("panel must contain vertices")
    return tuple(result)


def _edge_endpoints(panel: Mapping[str, Any], edge_index: int) -> tuple[int, int]:
    vertices = _panel_vertices(panel)
    edges = _sequence(panel.get("edges"), "panel.edges")
    if edge_index < 0 or edge_index >= len(edges):
        raise SleeveEffectError("edge index is out of range")
    endpoints = _sequence(
        _mapping(edges[edge_index], "panel edge").get("endpoints"),
        "edge.endpoints",
    )
    if (
        len(endpoints) != 2
        or any(isinstance(index, bool) or not isinstance(index, int) for index in endpoints)
        or any(index < 0 or index >= len(vertices) for index in endpoints)
        or endpoints[0] == endpoints[1]
    ):
        raise SleeveEffectError("edge endpoints are invalid")
    return int(endpoints[0]), int(endpoints[1])


def _adjacency_signature(
    panel: Mapping[str, Any], vertex: int, chain_edges: frozenset[int]
) -> tuple[Any, ...]:
    records = []
    for index, raw in enumerate(_sequence(panel.get("edges"), "panel.edges")):
        if index not in chain_edges and vertex in _edge_endpoints(panel, index):
            edge = _mapping(raw, "panel edge")
            records.append((_edge_label(edge), edge_type(edge)))
    return tuple(sorted(records, key=repr))


def _canonical_equal(first: Any, second: Any) -> bool:
    return canonical_json_bytes(first) == canonical_json_bytes(second)


def _spec_metadata(specification: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return specification_metadata(specification)
    except PatternGeometryError as exc:
        raise SleeveEffectError(str(exc)) from exc


@dataclass(frozen=True)
class _SleeveArmholeChain:
    panel_name: str
    edge_indices: frozenset[int]
    directed_edges: tuple[int, ...]
    anchor_vertices: tuple[int, int]
    anchor_points: tuple[tuple[float, float], tuple[float, float]]
    anchor_adjacency: tuple[tuple[Any, ...], tuple[Any, ...]]


def _sleeve_armhole_chain(
    panel_name: str,
    panel: Mapping[str, Any],
    raw_edge_indices: Sequence[int],
) -> _SleeveArmholeChain:
    edge_indices = frozenset(raw_edge_indices)
    if not edge_indices:
        raise SleeveEffectError(f"{panel_name} has no armhole attachment edges")
    edges = _sequence(panel.get("edges"), f"{panel_name}.edges")
    vertices = _panel_vertices(panel)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(edges)
        for index in edge_indices
    ):
        raise SleeveEffectError(f"{panel_name} armhole edge index is invalid")

    adjacency: dict[int, list[int]] = defaultdict(list)
    endpoint_pairs: set[frozenset[int]] = set()
    incident_vertices: set[int] = set()
    for index in range(len(edges)):
        first, second = _edge_endpoints(panel, index)
        incident_vertices.update((first, second))
        if index not in edge_indices:
            continue
        pair = frozenset((first, second))
        if pair in endpoint_pairs:
            raise SleeveEffectError(f"{panel_name} armhole chain reuses an endpoint pair")
        endpoint_pairs.add(pair)
        adjacency[first].append(index)
        adjacency[second].append(index)
    if incident_vertices != set(range(len(vertices))):
        raise SleeveEffectError(f"{panel_name} contains an isolated panel vertex")
    if any(len(indices) > 2 for indices in adjacency.values()):
        raise SleeveEffectError(f"{panel_name} armhole chain is branched")
    endpoints = tuple(vertex for vertex, indices in adjacency.items() if len(indices) == 1)
    if len(endpoints) != 2:
        raise SleeveEffectError(f"{panel_name} armhole chain must have two outer anchors")

    directed = None
    anchors = None
    for start in endpoints:
        current = start
        ordered = []
        unused = set(edge_indices)
        while unused:
            candidates = [index for index in adjacency[current] if index in unused]
            if len(candidates) != 1:
                break
            index = candidates[0]
            first, second = _edge_endpoints(panel, index)
            if first != current:
                break
            ordered.append(index)
            unused.remove(index)
            current = second
        if not unused and current != start:
            directed = tuple(ordered)
            anchors = (start, current)
            break
    if directed is None or anchors is None:
        raise SleeveEffectError(f"{panel_name} armhole chain is not a directed boundary run")
    adjacency_records = tuple(
        _adjacency_signature(panel, vertex, edge_indices) for vertex in anchors
    )
    if any(not record for record in adjacency_records):
        raise SleeveEffectError(f"{panel_name} armhole anchors lack protected adjacency")
    return _SleeveArmholeChain(
        panel_name,
        edge_indices,
        directed,
        anchors,
        tuple(vertices[index] for index in anchors),
        adjacency_records,
    )


def _absent_sleeve_armhole_chains(
    specification: Mapping[str, Any],
) -> Mapping[str, _SleeveArmholeChain]:
    panels = _panels(specification)
    chains = {}
    for panel_name in _TORSO_PANELS:
        side = panel_name.split("_", 1)[0]
        panel = _mapping(panels.get(panel_name), panel_name)
        indices = tuple(
            index
            for index, raw in enumerate(
                _sequence(panel.get("edges"), f"{panel_name}.edges")
            )
            if _edge_label(_mapping(raw, "panel edge")) == f"{side}_armhole"
        )
        chains[panel_name] = _sleeve_armhole_chain(panel_name, panel, indices)
    return dict(sorted(chains.items()))


def _present_sleeve_topology(
    specification: Mapping[str, Any],
) -> tuple[Mapping[str, _SleeveArmholeChain], Mapping[str, Any]]:
    panels = _panels(specification)
    expected_names = {
        f"{side}_sleeve_{half}"
        for side in ("left", "right")
        for half in ("f", "b")
    }
    if {name for name in panels if SLEEVE_PANEL.fullmatch(name)} != expected_names:
        raise SleeveEffectError("present sleeve topology has the wrong component membership")

    external_relations: Counter[tuple[str, str, str]] = Counter()
    attachment_counts: Counter[str] = Counter()
    internal_relations: Counter[str] = Counter()
    torso_edges: dict[str, set[int]] = defaultdict(set)
    used_sleeve_endpoints: set[tuple[str, int]] = set()
    used_torso_endpoints: set[tuple[str, int]] = set()
    labelled_attachment = {"left": False, "right": False}
    for stitch in _stitch_records(specification):
        endpoints = stitch[:2]
        component = tuple(
            endpoint for endpoint in endpoints if SLEEVE_PANEL.fullmatch(endpoint[0])
        )
        if not component:
            continue
        if stitch[2]:
            raise SleeveEffectError("sleeve stitches must not carry extra markers")
        for endpoint in component:
            if endpoint in used_sleeve_endpoints:
                raise SleeveEffectError("a sleeve edge participates in multiple stitches")
            used_sleeve_endpoints.add(endpoint)
        if len(component) == 2:
            matches = tuple(SLEEVE_PANEL.fullmatch(name) for name, _ in component)
            if any(match is None for match in matches):
                raise SleeveEffectError("sleeve panel attribution is invalid")
            sides = {match.group(1) for match in matches if match is not None}
            halves = {
                match.group(0).rsplit("_", 1)[-1]
                for match in matches
                if match is not None
            }
            if len(sides) != 1 or halves != {"f", "b"}:
                raise SleeveEffectError(
                    "internal sleeve seams must join same-side front/back panels"
                )
            internal_relations[next(iter(sides))] += 1
            continue
        if len(component) != 1:
            raise SleeveEffectError("sleeve stitch has ambiguous component incidence")

        component_name, _ = component[0]
        match = SLEEVE_PANEL.fullmatch(component_name)
        if match is None:
            raise SleeveEffectError("sleeve panel attribution drifted")
        side = match.group(1)
        sleeve_half = component_name.rsplit("_", 1)[-1]
        external_name, external_edge = (
            endpoints[1] if endpoints[0] == component[0] else endpoints[0]
        )
        allowed_external = (
            {f"{side}_ftorso"}
            if sleeve_half == "f"
            else {f"{side}_ftorso", f"{side}_btorso"}
        )
        if external_name not in allowed_external:
            raise SleeveEffectError("sleeve attachment crosses side or torso relation")
        external_endpoint = (external_name, external_edge)
        if external_endpoint in used_torso_endpoints:
            raise SleeveEffectError(
                "a torso armhole edge participates in multiple sleeve stitches"
            )
        used_torso_endpoints.add(external_endpoint)
        torso_half = "f" if external_name.endswith("_ftorso") else "b"
        external_relations[(side, sleeve_half, torso_half)] += 1
        attachment_counts[component_name] += 1
        torso_edges[external_name].add(external_edge)
        label = _edge_label(_edge(specification, external_name, external_edge))
        if label not in {None, f"{side}_armhole"}:
            raise SleeveEffectError("sleeve attachment uses a non-armhole torso label")
        if label == f"{side}_armhole":
            labelled_attachment[side] = True

    required = {
        (side, half, half)
        for side in ("left", "right")
        for half in ("f", "b")
    }
    allowed = required | {(side, "b", "f") for side in ("left", "right")}
    if (
        not required.issubset(external_relations)
        or not set(external_relations).issubset(allowed)
        or any(count < 1 for count in external_relations.values())
    ):
        raise SleeveEffectError(
            "sleeve external attachment relations are incomplete or invalid"
        )
    if internal_relations != Counter({"left": 2, "right": 2}):
        raise SleeveEffectError("each sleeve side requires exactly two front/back seams")
    if not all(labelled_attachment.values()):
        raise SleeveEffectError("each sleeve side requires a protected armhole label")

    chains = {}
    for panel_name in _TORSO_PANELS:
        panel = _mapping(panels.get(panel_name), panel_name)
        chain = _sleeve_armhole_chain(
            panel_name,
            panel,
            tuple(sorted(torso_edges.get(panel_name, set()))),
        )
        side = panel_name.split("_", 1)[0]
        labelled = {
            index
            for index, raw in enumerate(
                _sequence(panel.get("edges"), f"{panel_name}.edges")
            )
            if _edge_label(_mapping(raw, "panel edge")) == f"{side}_armhole"
        }
        if not labelled.issubset(chain.edge_indices):
            raise SleeveEffectError(
                "an armhole-labelled torso edge is not sleeve-incident"
            )
        chains[panel_name] = chain
    topology = {
        "armhole_stitch_membership": {
            side: all(torso_edges.get(f"{side}_{half}torso") for half in ("f", "b"))
            for side in ("left", "right")
        },
        "attachment_relations_valid": True,
        "external_panel_attachment_counts": {
            name: attachment_counts.get(name, 0) for name in sorted(expected_names)
        },
        "external_relations": {
            repr(key): value
            for key, value in sorted(external_relations.items(), key=lambda item: repr(item[0]))
        },
        "internal_relations": dict(sorted(internal_relations.items())),
        "labelled_attachment": labelled_attachment,
    }
    return dict(sorted(chains.items())), topology


def _non_chain_geometry(
    panel: Mapping[str, Any], chain: _SleeveArmholeChain
) -> tuple[bytes, tuple[Any, ...]]:
    vertices = _sequence(panel.get("vertices"), "panel.vertices")
    static = {key: value for key, value in panel.items() if key not in {"vertices", "edges"}}
    records = []
    for index, raw in enumerate(_sequence(panel.get("edges"), "panel.edges")):
        if index in chain.edge_indices:
            continue
        edge = _mapping(raw, "panel edge")
        first, second = _edge_endpoints(panel, index)
        records.append((
            canonical_json_bytes(vertices[first]),
            canonical_json_bytes(vertices[second]),
            canonical_json_bytes({key: value for key, value in edge.items() if key != "endpoints"}),
        ))
    return canonical_json_bytes(static), tuple(sorted(records))


def _panel_static_topology(panel: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes({
        key: value
        for key, value in panel.items()
        if key not in {"vertices", "edges", "translation", "rotation"}
    })


def _panel_topology(panel: Mapping[str, Any]) -> tuple[bytes, int, tuple[Any, ...]]:
    return (
        _panel_static_topology(panel),
        len(_sequence(panel.get("vertices"), "panel.vertices")),
        _edge_topology(panel),
    )


def _non_chain_relations(
    panel: Mapping[str, Any], chain: _SleeveArmholeChain
) -> Mapping[int, tuple[Any, ...]]:
    edges = _sequence(panel.get("edges"), "panel.edges")
    topology = _edge_topology(panel)
    unused = set(range(len(edges))) - chain.edge_indices
    current = chain.anchor_vertices[1]
    relations = {}
    for position in range(len(unused)):
        candidates = [
            index for index in unused if _edge_endpoints(panel, index)[0] == current
        ]
        if len(candidates) != 1:
            raise SleeveEffectError(
                f"{chain.panel_name} non-armhole boundary is not a directed run"
            )
        index = candidates[0]
        first, second = _edge_endpoints(panel, index)
        if first != current:
            raise SleeveEffectError(
                f"{chain.panel_name} non-armhole boundary direction drifted"
            )
        kind, _endpoints, label = topology[index]
        relations[index] = ("non_armhole_edge", position, kind, label)
        unused.remove(index)
        current = second
    if unused or current != chain.anchor_vertices[0]:
        raise SleeveEffectError(
            f"{chain.panel_name} non-armhole boundary does not connect the armhole anchors"
        )
    return relations


def _non_chain_topology(
    panel: Mapping[str, Any], chain: _SleeveArmholeChain
) -> tuple[bytes, tuple[Any, ...]]:
    relations = _non_chain_relations(panel, chain)
    return (
        _panel_static_topology(panel),
        tuple(
            relation
            for _index, relation in sorted(relations.items(), key=lambda item: item[1][1])
        ),
    )


def _chains_match(
    source: _SleeveArmholeChain,
    target: _SleeveArmholeChain,
    *,
    require_coordinate_match: bool = True,
) -> tuple[bool, float | None]:
    if source.anchor_adjacency != target.anchor_adjacency:
        return False, None
    drift = tuple(
        math.hypot(target_point[0] - source_point[0], target_point[1] - source_point[1])
        for source_point, target_point in zip(source.anchor_points, target.anchor_points)
    )
    if any(not math.isfinite(value) for value in drift):
        raise SleeveEffectError("sleeve armhole anchor drift must be finite")
    maximum = max(drift)
    return not require_coordinate_match or maximum <= _GEOMETRY_ATOL, maximum


def _normalized_non_sleeve_stitches(
    specification: Mapping[str, Any],
    chains: Mapping[str, _SleeveArmholeChain],
    *,
    topology_only: bool = False,
) -> Counter[Any]:
    panels = _panels(specification)
    result: Counter[Any] = Counter()
    topology_relations = (
        {
            name: _non_chain_relations(_mapping(panels.get(name), name), chain)
            for name, chain in chains.items()
        }
        if topology_only
        else {}
    )
    for stitch in _stitch_records(specification):
        if any(SLEEVE_PANEL.fullmatch(name) for name, _ in stitch[:2]):
            continue
        normalized = []
        for panel_name, edge_index in stitch[:2]:
            chain = chains.get(panel_name)
            if chain is not None and edge_index in chain.edge_indices:
                relation: Any = ("armhole_chain",)
            elif topology_only:
                panel = _mapping(panels.get(panel_name), panel_name)
                edge = _edge(specification, panel_name, edge_index)
                if chain is not None:
                    relation = topology_relations[panel_name].get(edge_index)
                    if relation is None:
                        raise SleeveEffectError(
                            "non-sleeve stitch references an unclassified torso edge"
                        )
                else:
                    relation = ("panel_edge", edge_index, edge_type(edge), _edge_label(edge))
            else:
                panel = _mapping(panels.get(panel_name), panel_name)
                edge = _edge(specification, panel_name, edge_index)
                first, second = _edge_endpoints(panel, edge_index)
                vertices = _sequence(panel.get("vertices"), "panel.vertices")
                relation = (
                    canonical_json_bytes(vertices[first]),
                    canonical_json_bytes(vertices[second]),
                    canonical_json_bytes(
                        {
                            key: value
                            for key, value in edge.items()
                            if key != "endpoints"
                        }
                    ),
                )
            normalized.append((panel_name, relation))
        normalized.sort(key=repr)
        result[(normalized[0], normalized[1], stitch[2])] += 1
    return result


def _membership_effect(
    direction: str,
    source_specification: Mapping[str, Any],
    target_specification: Mapping[str, Any],
    topology_policy: str,
) -> EffectResult:
    try:
        if direction not in {"add", "remove"}:
            raise SleeveEffectError("sleeve direction must be exactly add or remove")
        source_panels = _panels(source_specification)
        target_panels = _panels(target_specification)
        expected_names = {
            f"{side}_sleeve_{half}"
            for side in ("left", "right")
            for half in ("f", "b")
        }
        source_component = {
            name for name in source_panels if SLEEVE_PANEL.fullmatch(name)
        }
        target_component = {
            name for name in target_panels if SLEEVE_PANEL.fullmatch(name)
        }
        membership_ok = (
            not source_component and target_component == expected_names
            if direction == "add"
            else source_component == expected_names and not target_component
        )
        source_unrelated = set(source_panels) - source_component
        target_unrelated = set(target_panels) - target_component
        unrelated_membership_ok = source_unrelated == target_unrelated
        if not membership_ok or not unrelated_membership_ok:
            return rule_effect_result("sleeve_presence", False, "effect/topology_mismatch", {
                "source_component_panels": sorted(source_component),
                "target_component_panels": sorted(target_component),
                "changed_panels": len(source_component.symmetric_difference(target_component)),
                "membership_ok": membership_ok,
                "unrelated_membership_ok": unrelated_membership_ok,
                "localized_armhole_recut": False,
                "normalized_unrelated_stitches_match": False,
            })

        absent = source_specification if direction == "add" else target_specification
        present = target_specification if direction == "add" else source_specification
        absent_chains = _absent_sleeve_armhole_chains(absent)
        present_chains, present_topology = _present_sleeve_topology(present)
        source_chains = absent_chains if direction == "add" else present_chains
        target_chains = present_chains if direction == "add" else absent_chains

        metadata_match = _canonical_equal(
            _spec_metadata(source_specification), _spec_metadata(target_specification)
        )
        non_torso = source_unrelated - set(_TORSO_PANELS)
        unrelated_panel_topology_match = all(
            _panel_topology(source_panels[name]) == _panel_topology(target_panels[name])
            for name in non_torso
        )
        unrelated_panels_equal = all(
            _canonical_equal(source_panels[name], target_panels[name]) for name in non_torso
        )
        strict_localized_panels = []
        local_topology_panels = []
        anchor_drift: dict[str, float | None] = {}
        for panel_name in _TORSO_PANELS:
            source_panel = _mapping(source_panels.get(panel_name), panel_name)
            target_panel = _mapping(target_panels.get(panel_name), panel_name)
            static_match = _non_chain_geometry(
                source_panel, source_chains[panel_name]
            ) == _non_chain_geometry(target_panel, target_chains[panel_name])
            strict_chains_match, maximum_drift = _chains_match(
                source_chains[panel_name], target_chains[panel_name]
            )
            local_static_match = _non_chain_topology(
                source_panel, source_chains[panel_name]
            ) == _non_chain_topology(target_panel, target_chains[panel_name])
            local_chains_match, _ = _chains_match(
                source_chains[panel_name],
                target_chains[panel_name],
                require_coordinate_match=False,
            )
            anchor_drift[panel_name] = maximum_drift
            if static_match and strict_chains_match:
                strict_localized_panels.append(panel_name)
            if local_static_match and local_chains_match:
                local_topology_panels.append(panel_name)
        strict_localized_armhole_recut = set(strict_localized_panels) == set(_TORSO_PANELS)
        local_armhole_topology_match = set(local_topology_panels) == set(_TORSO_PANELS)
        strict_stitches_match = _normalized_non_sleeve_stitches(
            source_specification, source_chains
        ) == _normalized_non_sleeve_stitches(target_specification, target_chains)
        local_stitches_match = _normalized_non_sleeve_stitches(
            source_specification, source_chains, topology_only=True
        ) == _normalized_non_sleeve_stitches(
            target_specification, target_chains, topology_only=True
        )
        localized_armhole_recut = (
            local_armhole_topology_match
            if topology_policy == "local"
            else strict_localized_armhole_recut
        )
        normalized_stitches_match = (
            local_stitches_match if topology_policy == "local" else strict_stitches_match
        )
    except SleeveEffectError as exc:
        return rule_effect_result(
            "sleeve_presence",
            False,
            "effect/topology_mismatch",
            {"message": str(exc)},
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return rule_effect_result(
            "sleeve_presence",
            False,
            "effect/measurement_error",
            {"message": str(exc)},
        )

    details = {
        "source_component_panels": sorted(source_component),
        "target_component_panels": sorted(target_component),
        "changed_panels": len(source_component.symmetric_difference(target_component)),
        "membership_ok": membership_ok,
        "unrelated_membership_ok": unrelated_membership_ok,
        "unrelated_panels_equal": unrelated_panels_equal,
        "unrelated_panel_topology_match": unrelated_panel_topology_match,
        "metadata_match": metadata_match,
        "localized_armhole_recut": localized_armhole_recut,
        "localized_armhole_panels": sorted(
            local_topology_panels if topology_policy == "local" else strict_localized_panels
        ),
        "strict_localized_armhole_recut": strict_localized_armhole_recut,
        "local_armhole_topology_match": local_armhole_topology_match,
        "armhole_anchor_drift": anchor_drift,
        "armhole_anchor_atol": _GEOMETRY_ATOL,
        "strict_normalized_unrelated_stitches_match": strict_stitches_match,
        "normalized_unrelated_stitch_topology_match": local_stitches_match,
        "normalized_unrelated_stitches_match": normalized_stitches_match,
        "present_topology": present_topology,
        "localized_stitch_change": normalized_stitches_match,
        "stitch_direction_ok": True,
        "topology_policy": topology_policy,
    }
    passed = (
        all((
            membership_ok,
            unrelated_membership_ok,
            unrelated_panel_topology_match,
            local_armhole_topology_match,
            localized_armhole_recut,
            normalized_stitches_match,
        ))
        if topology_policy == "local"
        else all((
            membership_ok,
            unrelated_membership_ok,
            unrelated_panels_equal,
            metadata_match,
            localized_armhole_recut,
            normalized_stitches_match,
        ))
    )
    return rule_effect_result(
        "sleeve_presence",
        passed,
        "effect/passed" if passed else "effect/topology_mismatch",
        details,
    )


def validate(
    source_design: Mapping[str, Any],
    target_design: Mapping[str, Any],
    body: Mapping[str, Any],
    source_specification: Mapping[str, Any],
    target_specification: Mapping[str, Any],
    planned_operation: PlannedOperation,
    config: EditConfig,
    topology_policy: str = "strict",
) -> EffectResult:
    """Validate exact sleeve membership, localized armholes, and attachments."""

    del source_design, target_design, body
    if topology_policy not in {"strict", "local"}:
        raise ValueError("topology_policy must be exactly 'strict' or 'local'")
    operation = planned_operation.operation
    try:
        rule_config = config.rules["sleeve_presence"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("config must contain sleeve_presence") from exc
    if operation.rule_id != "sleeve_presence":
        raise ValueError("operation names a foreign rule")
    if operation.effect_metric_id != rule_config.effect_metric:
        raise ValueError("operation names the wrong sleeve effect metric")
    if operation.effect_contract_id != rule_config.effect_contract:
        raise ValueError("operation names the wrong sleeve effect contract")
    if operation.scope != "both":
        raise ValueError("sleeve presence requires scope 'both'")
    if operation.direction not in {"add", "remove"}:
        raise ValueError("sleeve direction must be exactly add or remove")
    direction = operation.direction
    result = _membership_effect(
        direction, source_specification, target_specification, topology_policy
    )
    if not result.passed:
        return result
    present = target_specification if direction == "add" else source_specification
    panels = _panels(present)
    topology = result.evidence["details"]["present_topology"]
    cuff_names = tuple(
        sorted(name for name in panels if _SLEEVE_CUFF_PANEL.fullmatch(name))
    )
    armholes = dict(topology["armhole_stitch_membership"])
    attachment_counts = topology["external_panel_attachment_counts"]
    panel_attachments = {
        name: attachment_counts.get(name, 0) > 0
        for name in sorted(panels)
        if SLEEVE_PANEL.fullmatch(name)
    }
    exact_attachment_relations = bool(topology["attachment_relations_valid"])
    attachment_mismatches = tuple(
        sorted(
            [name for name, attached in panel_attachments.items() if not attached]
            + [
                f"{side}_armhole"
                for side, attached in armholes.items()
                if not attached
            ]
        )
    )
    details = dict(result.evidence["details"])
    details.update({
        "cuff_panel_names": cuff_names,
        "armhole_stitch_membership": armholes,
        "sleeve_panel_attachments": panel_attachments,
        "exact_attachment_relations": exact_attachment_relations,
        "attachment_mismatches": attachment_mismatches,
    })
    passed = (
        not cuff_names
        and all(armholes.values())
        and all(panel_attachments.values())
        and exact_attachment_relations
    )
    return rule_effect_result(
        "sleeve_presence",
        passed,
        "effect/passed" if passed else "effect/topology_mismatch",
        details,
    )


def build_rules(config: EditConfig) -> tuple[Rule, ...]:
    """Bind the sleeve rule to one loaded configuration."""

    if not isinstance(config, EditConfig):
        raise TypeError("config must be an EditConfig")
    try:
        rule_config = config.rules["sleeve_presence"]
    except KeyError as exc:
        raise ValueError("config must contain sleeve_presence") from exc
    _validate_config(rule_config)

    def candidates(context: RuleContext) -> tuple[PlannedOperation, ...]:
        return _candidates(context, rule_config)

    return (Rule("sleeve_presence", candidates, apply, validate),)
