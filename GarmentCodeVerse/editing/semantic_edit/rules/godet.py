"""Self-contained Godet insert-count semantic rule."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any

from editing.semantic_edit.prompt_state import ActivePathError, resolve_active_paths
from editing.canonical_io import (
    canonical_hash,
    canonical_json_bytes,
    design_dotted_get,
    plain_copy,
)
from editing.config import ConfigError, EditConfig, ParameterSpec, RuleConfig
from editing.models import (
    BodySnapshot,
    EffectResult,
    Operation,
    OperationContractError,
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

GODET_COUNT_PATH = "design.godet-skirt.num_inserts.v"
GODET_COUNTS = (4, 6, 8, 10, 12)
GODET_INSERT_WIDTH_PATH = "design.godet-skirt.insert_w.v"
GODET_INSERT_DEPTH_PATH = "design.godet-skirt.insert_depth.v"
GODET_PILOT_EXCLUDED_TRANSITIONS = frozenset({(4, 8), (10, 6)})
GODET_PILOT_MAX_TARGET_COMPLEXITY = 12_000.0
GODET_PILOT_GUARDRAIL_SCHEMA = "gcd-godet-insert-count-pilot-guardrail-v1"
GODET_BASES = frozenset({"Skirt2", "PencilSkirt"})
GODET_READ_PATHS = (
    "design.godet-skirt.base.v",
    "design.godet-skirt.cuts_distance.v",
    "design.godet-skirt.insert_depth.v",
    "design.godet-skirt.insert_w.v",
    GODET_COUNT_PATH,
    "design.meta.bottom.v",
)
GODET_PREREQUISITES = (
    "base_bottom_interface_count_positive",
    "bottom_is_GodetSkirt",
    "godet_base_supported",
    "source_and_target_counts_divisible",
    "pilot_transition_allowed",
    "pilot_target_complexity_at_most_12000",
)
_FACT_NAME = "godet_base_bottom_interface_count"
_NUMBER_WORDS = {
    4: "four",
    6: "six",
    8: "eight",
    10: "ten",
    12: "twelve",
}

GODET_PANEL = re.compile(r"^ins_skirt_(front|back)_[0-9]+$")
_GEOMETRY_ATOL = 1.0e-7


class ConstructorFactError(ValueError):
    """A Godet constructor fact could not be established."""


class EffectMeasurementError(ValueError):
    """A constructed pattern cannot support Godet validation."""


def _unavailable(code: str, message: str) -> StateUnavailableError:
    return StateUnavailableError(code, message)


def classify_godet_state(design: Mapping[str, Any]) -> SemanticState:
    """Classify the exact active GodetSkirt insert count."""

    try:
        active = resolve_active_paths(design)
    except (ActivePathError, KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "active_path_unavailable",
            f"cannot resolve active constructor paths: {exc}",
        ) from exc
    if (
        active.dispatch.get("bottom") != "GodetSkirt"
        or GODET_COUNT_PATH not in active.paths
    ):
        raise _unavailable(
            "godet_insert_count_inapplicable",
            "godet insert count requires an active GodetSkirt",
        )
    try:
        count = design_dotted_get(design, GODET_COUNT_PATH)
    except (KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "state_schema_unavailable",
            f"active state path is missing or invalid: {GODET_COUNT_PATH}",
        ) from exc
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count not in _NUMBER_WORDS
    ):
        raise _unavailable(
            "godet_insert_count_unavailable",
            "godet insert count must be one of 4, 6, 8, 10, or 12",
        )
    return SemanticState(
        rule_id="godet_insert_count",
        state_id=f"{count}-inserts",
        phrase=f"{_NUMBER_WORDS[count]} godet inserts",
        attributes={"count": count},
    )


def _load_constructor_fact_dependencies() -> tuple[type, Any]:
    """Load the untouched GarmentCode constructor classes on demand."""

    from assets.bodies.body_params import BodyParameters
    from assets.garment_programs import skirt_paneled

    return BodyParameters, skirt_paneled


def probe_constructor_facts(
    design: Mapping[str, Any],
    body_snapshot: BodySnapshot,
) -> dict[str, Any]:
    """Probe the Godet base's bottom-interface count on private copies."""

    if not isinstance(body_snapshot, BodySnapshot):
        raise ConstructorFactError(
            "constructor facts require an immutable body snapshot"
        )
    if not isinstance(design, Mapping):
        raise ConstructorFactError("constructor facts require a canonical design")
    working_design = plain_copy(design)

    try:
        bottom_name = working_design["meta"]["bottom"]["v"]
    except (KeyError, TypeError) as exc:
        raise ConstructorFactError(
            "constructor facts require a valid bottom dispatch"
        ) from exc
    if bottom_name != "GodetSkirt":
        return {}

    try:
        base_name = working_design["godet-skirt"]["base"]["v"]
    except (KeyError, TypeError) as exc:
        raise ConstructorFactError(
            "constructor facts require a valid Godet base dispatch"
        ) from exc
    if not isinstance(base_name, str) or base_name not in GODET_BASES:
        raise ConstructorFactError(f"unsupported Godet base: {base_name!r}")

    working_body = plain_copy(body_snapshot.body)
    try:
        from ..construction import instantiate_body_parameters

        body_parameters_cls, skirt_paneled = _load_constructor_fact_dependencies()
        body = instantiate_body_parameters(body_parameters_cls)
        body.load_from_dict(working_body)
        base_class = getattr(skirt_paneled, base_name)
        upper = working_design["meta"]["upper"]["v"]
        rise = 1.0 if upper and "Fitted" in upper else None
        base = base_class(body, working_design, rise=rise, slit=False)
        count = len(base.interfaces["bottom"])
    except ConstructorFactError:
        raise
    except Exception as exc:
        raise ConstructorFactError("failed to probe Godet constructor facts") from exc
    if count <= 0:
        raise ConstructorFactError("Godet base has no bottom interfaces")
    return {_FACT_NAME: count}


def _validate_rule_config(config: RuleConfig) -> ParameterSpec:
    """Reject Godet config drift from the frozen structural contract."""

    rule_id = "godet_insert_count"
    if type(config) is not RuleConfig:
        raise ConfigError(f"{rule_id} rule_config must be an exact RuleConfig")
    if type(config.enabled) is not bool or config.enabled is not True:
        raise ConfigError(f"{rule_id} rule_config must be enabled")
    if (
        config.semantic_group != "skirt"
        or type(config.structural_level) is not int
        or config.structural_level != 2
        or config.effect_metric != "exact_godet_insert_membership"
        or config.effect_contract != "exact_membership_change"
    ):
        raise ConfigError(f"{rule_id} rule_config has an invalid operation contract")
    if set(config.parameter_specs) != {"num_inserts"}:
        raise ConfigError(f"{rule_id} rule_config must contain only num_inserts")
    spec = config.parameter_specs["num_inserts"]
    if (
        type(spec) is not ParameterSpec
        or spec.edit_mode != "enum"
        or spec.value_type != "int"
        or spec.transitions != GODET_COUNTS
    ):
        raise ConfigError(f"{rule_id} num_inserts config has invalid transitions")
    return spec


def _applicability_evidence(
    context: RuleContext,
    operation: Operation,
    transitions: tuple[SemanticTransition, ...],
) -> dict[str, Any]:
    return {
        "rule_id": operation.rule_id,
        "scope": operation.scope,
        "source_design_hash": canonical_hash(context.design),
        "body_hash": canonical_hash(context.body),
        "active_paths_design_hash": context.active_paths.design_hash,
        "requested_magnitude": operation.requested_magnitude,
        "active_path_match": (
            context.active_paths.design_hash == canonical_hash(context.design)
        ),
        "source_states": [
            transition.source_state.to_payload() for transition in transitions
        ],
        "target_states": [
            transition.target_state.to_payload() for transition in transitions
        ],
    }


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _pilot_guardrail_evidence(
    *,
    source_count: int,
    target_count: int,
    insert_width: float,
    insert_depth: float,
) -> dict[str, Any]:
    transition = (source_count, target_count)
    transition_excluded = transition in GODET_PILOT_EXCLUDED_TRANSITIONS
    target_complexity = target_count * insert_width * insert_depth
    complexity_within_limit = target_complexity <= GODET_PILOT_MAX_TARGET_COMPLEXITY
    rejections: list[dict[str, Any]] = []
    if transition_excluded:
        rejections.append(
            {
                "code": "pilot_excluded_transition",
                "message": (
                    "pilot simulation excluded godet transition "
                    f"{source_count}->{target_count}"
                ),
            }
        )
    if not complexity_within_limit:
        rejections.append(
            {
                "code": "pilot_target_complexity_exceeded",
                "message": (
                    "target_count * insert_w * insert_depth exceeds "
                    f"{GODET_PILOT_MAX_TARGET_COMPLEXITY:g}"
                ),
            }
        )
    return {
        "schema_version": GODET_PILOT_GUARDRAIL_SCHEMA,
        "source_count": source_count,
        "target_count": target_count,
        "insert_w": insert_width,
        "insert_depth": insert_depth,
        "target_complexity": target_complexity,
        "maximum_target_complexity": GODET_PILOT_MAX_TARGET_COMPLEXITY,
        "transition_excluded": transition_excluded,
        "complexity_within_limit": complexity_within_limit,
        "accepted": not rejections,
        "rejections": tuple(rejections),
    }


def _write_count(
    design: Mapping[str, Any],
    value: int,
) -> dict[str, Any]:
    """Write only ``.v`` without treating persisted sampling ranges as policy."""

    if not isinstance(design, Mapping):
        raise OperationContractError("Godet design must be a mapping")
    working = plain_copy(design)
    try:
        parameter = working["godet-skirt"]["num_inserts"]
        if (
            not isinstance(parameter, dict)
            or "v" not in parameter
            or type(parameter.get("type")) is not str
            or parameter["type"] != "select"
        ):
            raise TypeError
        parameter["v"] = value
    except (KeyError, TypeError) as exc:
        raise OperationContractError(
            f"candidate write path is absent from the design: " f"{GODET_COUNT_PATH}"
        ) from exc
    return working


@dataclass(frozen=True, slots=True)
class GodetInsertCountRule:
    """Enumerate exact, constructor-safe Godet insert membership changes."""

    rule_config: RuleConfig = field(repr=False)
    id: str = field(default="godet_insert_count", init=False)
    _parameter_spec: ParameterSpec = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Materialize the init=False identity explicitly on CPython 3.10.
        object.__setattr__(self, "id", "godet_insert_count")
        object.__setattr__(self, "_parameter_spec", _validate_rule_config(self.rule_config))

    def _source_inputs(
        self,
        context: RuleContext,
    ) -> tuple[str, int, int, float, float] | None:
        """Return validated constructor inputs or ordinary inapplicability."""

        try:
            active_bottom = context.active_paths.dispatch.get("bottom")
            stored_bottom = design_dotted_get(
                context.design,
                "design.meta.bottom.v",
            )
            if (
                type(active_bottom) is not str
                or active_bottom != "GodetSkirt"
                or type(stored_bottom) is not str
                or stored_bottom != "GodetSkirt"
            ):
                return None
            base_name = design_dotted_get(
                context.design,
                "design.godet-skirt.base.v",
            )
            source_count = design_dotted_get(
                context.design,
                GODET_COUNT_PATH,
            )
            count_parameter = context.design["godet-skirt"]["num_inserts"]
            if (
                not isinstance(count_parameter, Mapping)
                or type(count_parameter.get("type")) is not str
                or count_parameter["type"] != "select"
            ):
                return None
            # These values are invariants, but reading them is part of the
            # candidate identity because they affect the actual constructor.
            insert_width = _finite_number(
                design_dotted_get(context.design, GODET_INSERT_WIDTH_PATH),
                GODET_INSERT_WIDTH_PATH,
            )
            insert_depth = _finite_number(
                design_dotted_get(context.design, GODET_INSERT_DEPTH_PATH),
                GODET_INSERT_DEPTH_PATH,
            )
            design_dotted_get(context.design, "design.godet-skirt.cuts_distance.v")
            bottom_interface_count = context.constructor_facts.get(_FACT_NAME)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if type(base_name) is not str or base_name not in GODET_BASES:
            return None
        if (
            type(source_count) is not int
            or source_count not in self._parameter_spec.transitions
        ):
            return None
        if (
            type(bottom_interface_count) is not int
            or bottom_interface_count <= 0
            or source_count % bottom_interface_count != 0
        ):
            return None
        return (
            base_name,
            source_count,
            bottom_interface_count,
            insert_width,
            insert_depth,
        )

    def candidates(
        self,
        context: RuleContext,
    ) -> tuple[PlannedOperation, ...]:
        source_inputs = self._source_inputs(context)
        if source_inputs is None:
            return ()
        (
            base_name,
            source_count,
            bottom_interface_count,
            insert_width,
            insert_depth,
        ) = source_inputs

        # Every classifier input has already passed the exact active dispatch,
        # metadata, scalar-type, and allowlist contract.  A classifier failure
        # here is therefore hard schema/program drift, not inapplicability.
        source_state = classify_godet_state(context.design)

        result: list[PlannedOperation] = []
        for target_count in self._parameter_spec.transitions:
            absolute_delta = abs(target_count - source_count)
            if absolute_delta < 4 or target_count % bottom_interface_count != 0:
                continue
            guardrail = _pilot_guardrail_evidence(
                source_count=source_count,
                target_count=target_count,
                insert_width=insert_width,
                insert_depth=insert_depth,
            )
            if guardrail["accepted"] is not True:
                continue
            direction = "increase" if target_count > source_count else "decrease"
            target_design = _write_count(
                context.design,
                target_count,
            )
            target_state = classify_godet_state(target_design)
            transition = SemanticTransition(
                rule_id=self.id,
                scope="garment",
                channels=("garment",),
                feature_phrase="the skirt",
                source_state=source_state,
                target_state=target_state,
                direction=direction,
            )
            operation = Operation(
                rule_id=self.id,
                semantic_group=self.rule_config.semantic_group,
                scope="garment",
                direction=direction,
                magnitude="structural",
                structural_level=self.rule_config.structural_level,
                reads=tuple(sorted(GODET_READ_PATHS)),
                writes=tuple(sorted((GODET_COUNT_PATH,))),
                conflicts=(),
                prerequisites=tuple(sorted(GODET_PREREQUISITES)),
                edit_mode=self._parameter_spec.edit_mode,
                effect_metric_id=self.rule_config.effect_metric,
                effect_contract_id=self.rule_config.effect_contract,
                before={GODET_COUNT_PATH: source_count},
                after={GODET_COUNT_PATH: target_count},
                relative_change=None,
                requested_magnitude="structural",
                realized_magnitude="structural",
                clamped=False,
                instruction_tokens={
                    "feature": "godet inserts",
                    "source_count": source_count,
                    "target_count": target_count,
                },
            )
            transitions = (transition,)
            applicability = _applicability_evidence(
                context,
                operation,
                transitions,
            )
            applicability.update(
                {
                    "absolute_count_delta": absolute_delta,
                    "godet_base": base_name,
                    _FACT_NAME: bottom_interface_count,
                    "minimum_absolute_count_delta": 4,
                    "source_count_divisible": True,
                    "target_count_divisible": True,
                    "pilot_guardrail": guardrail,
                }
            )
            result.append(
                PlannedOperation(
                    operation=operation,
                    transitions=transitions,
                    topology_kind=TopologyKind.MEMBERSHIP,
                    applicability_evidence=applicability,
                    conflict_evidence=empty_conflict_evidence(operation.rule_id),
                )
            )
        return tuple(result)


def apply(design: dict[str, Any], planned: PlannedOperation) -> dict[str, Any]:
    """Apply one planned Godet count while preserving parameter metadata."""

    return _write_count(design, planned.operation.after[GODET_COUNT_PATH])


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EffectMeasurementError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise EffectMeasurementError(f"{field_name} must be finite") from exc
    if not math.isfinite(result):
        raise EffectMeasurementError(f"{field_name} must be finite")
    return result


def _canonical_equal(first: Any, second: Any) -> bool:
    return canonical_json_bytes(first) == canonical_json_bytes(second)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EffectMeasurementError(f"{field_name} must be a mapping")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EffectMeasurementError(f"{field_name} must be a sequence")
    return value


def _panels(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return _mapping(specification["pattern"]["panels"], "pattern.panels")
    except KeyError as exc:
        raise EffectMeasurementError("pattern.panels is required") from exc


def _edge(
    specification: Mapping[str, Any],
    panel_name: str,
    edge_index: int,
) -> Mapping[str, Any]:
    panel = _mapping(
        _panels(specification).get(panel_name),
        f"pattern.panels.{panel_name}",
    )
    edges = _sequence(panel.get("edges"), f"pattern.panels.{panel_name}.edges")
    if (
        isinstance(edge_index, bool)
        or not isinstance(edge_index, int)
        or edge_index < 0
        or edge_index >= len(edges)
    ):
        raise EffectMeasurementError(
            f"invalid edge reference {panel_name}[{edge_index!r}]"
        )
    return _mapping(
        edges[edge_index],
        f"pattern.panels.{panel_name}.edges[{edge_index}]",
    )


def _edge_label(edge: Mapping[str, Any]) -> str | None:
    label = edge.get("label")
    if label is None:
        return None
    if type(label) is not str or not label:
        raise EffectMeasurementError("edge labels must be non-empty strings")
    return label


def _stitch_records(
    specification: Mapping[str, Any],
) -> tuple[tuple[tuple[str, int], tuple[str, int], tuple[Any, ...]], ...]:
    try:
        return stitch_records(specification)
    except PatternGeometryError as exc:
        raise EffectMeasurementError(str(exc)) from exc


def _spec_metadata(specification: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return specification_metadata(specification)
    except PatternGeometryError as exc:
        raise EffectMeasurementError(str(exc)) from exc


def _edge_topology(panel: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return edge_topology(panel)
    except PatternGeometryError as exc:
        raise EffectMeasurementError(str(exc)) from exc


def _insert_stitch(stitch: Any) -> bool:
    return any(GODET_PANEL.fullmatch(panel_name) for panel_name, _ in stitch[:2])


def _semantic_stitch_counter(
    specification: Mapping[str, Any],
    *,
    topology_only: bool = False,
) -> Counter[Any]:
    result: Counter[Any] = Counter()
    for stitch in _stitch_records(specification):
        if _insert_stitch(stitch):
            continue
        normalized = []
        for panel_name, edge_index in stitch[:2]:
            edge = _edge(specification, panel_name, edge_index)
            relation: Any = _edge_label(edge)
            if topology_only:
                relation = (relation, edge_type(edge))
            elif relation is None:
                curve_type = edge_type(edge)
                relation = (
                    curve_type,
                    canonical_json_bytes(edge.get("curvature")),
                )
            normalized.append((panel_name, relation))
        normalized.sort(key=repr)
        result[(normalized[0], normalized[1], stitch[2])] += 1
    return result


def _panel_vertices(panel: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    result = []
    for raw in _sequence(panel.get("vertices"), "panel.vertices"):
        point = _sequence(raw, "panel vertex")
        if len(point) != 2:
            raise EffectMeasurementError("panel vertices must be 2-D")
        result.append(
            (
                _number(point[0], "vertex.x"),
                _number(point[1], "vertex.y"),
            )
        )
    if not result:
        raise EffectMeasurementError("panel must contain vertices")
    return tuple(result)


def _edge_endpoints(
    panel: Mapping[str, Any],
    edge_index: int,
) -> tuple[int, int]:
    vertices = _panel_vertices(panel)
    edges = _sequence(panel.get("edges"), "panel.edges")
    if edge_index < 0 or edge_index >= len(edges):
        raise EffectMeasurementError("edge index is out of range")
    edge = _mapping(edges[edge_index], "panel edge")
    endpoints = _sequence(edge.get("endpoints"), "edge.endpoints")
    if (
        len(endpoints) != 2
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in endpoints
        )
        or any(index < 0 or index >= len(vertices) for index in endpoints)
        or endpoints[0] == endpoints[1]
    ):
        raise EffectMeasurementError("edge endpoints are invalid")
    return int(endpoints[0]), int(endpoints[1])


@dataclass(frozen=True)
class _GodetAttachmentSummary:
    valid: bool
    base_panels: frozenset[str]
    uniform_degree: int | None
    peers_by_half: tuple[tuple[str, tuple[str, ...]], ...]
    cut_edges_by_panel: tuple[tuple[str, tuple[int, ...]], ...]


def _godet_attachment_summary(
    specification: Mapping[str, Any],
    insert_names: set[str],
) -> _GodetAttachmentSummary:
    degrees: Counter[str] = Counter()
    insert_edge_membership: dict[str, set[int]] = defaultdict(set)
    peer_edge_membership: dict[str, set[tuple[str, int]]] = defaultdict(set)
    base_panels: set[str] = set()
    peers_by_half: dict[str, set[str]] = defaultdict(set)
    cut_edges_by_panel: dict[str, set[int]] = defaultdict(set)
    valid = True
    for stitch in _stitch_records(specification):
        insert_endpoints = [
            endpoint
            for endpoint in stitch[:2]
            if GODET_PANEL.fullmatch(endpoint[0])
        ]
        if not insert_endpoints:
            continue
        if len(insert_endpoints) != 1:
            valid = False
            continue
        insert_name, insert_edge_index = insert_endpoints[0]
        peer = stitch[1] if stitch[0] == insert_endpoints[0] else stitch[0]
        match = GODET_PANEL.fullmatch(insert_name)
        expected_peer = None if match is None else f"skirt_{match.group(1)}"
        if match is None or peer[0] in insert_names or peer[0] != expected_peer:
            valid = False
            continue
        degrees[insert_name] += 1
        insert_edge_membership[insert_name].add(insert_edge_index)
        peer_edge_membership[insert_name].add(peer)
        base_panels.add(peer[0])
        peers_by_half[match.group(1)].add(peer[0])
        cut_edges_by_panel[peer[0]].add(peer[1])
    degree_values = {degrees.get(name, 0) for name in insert_names}
    uniform_degree = (
        next(iter(degree_values))
        if len(degree_values) == 1 and next(iter(degree_values)) > 0
        else None
    )
    valid = (
        valid
        and set(degrees) == insert_names
        and uniform_degree == 2
        and all(
            len(insert_edge_membership[name]) == 2
            and len(peer_edge_membership[name]) == 2
            for name in insert_names
        )
    )
    return _GodetAttachmentSummary(
        valid=valid,
        base_panels=frozenset(base_panels),
        uniform_degree=uniform_degree,
        peers_by_half=tuple(
            (half, tuple(sorted(peers)))
            for half, peers in sorted(peers_by_half.items())
        ),
        cut_edges_by_panel=tuple(
            (panel_name, tuple(sorted(edge_indices)))
            for panel_name, edge_indices in sorted(cut_edges_by_panel.items())
        ),
    )


def _godet_base_panel_records(
    panel: Mapping[str, Any],
    cut_edges: frozenset[int],
) -> tuple[bytes, tuple[Any, ...], tuple[Any, ...]]:
    vertices = _sequence(panel.get("vertices"), "panel.vertices")
    numeric_vertices = _panel_vertices(panel)
    edges = _sequence(panel.get("edges"), "panel.edges")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(edges)
        for index in cut_edges
    ):
        raise EffectMeasurementError("Godet cut edge index is invalid")
    if not cut_edges:
        raise EffectMeasurementError(
            "Godet base panel requires insert-incident cut edges"
        )

    groups: list[list[int]] = []
    for edge_index in sorted(cut_edges):
        if not groups or edge_index - groups[-1][-1] > 2:
            groups.append([edge_index])
        else:
            groups[-1].append(edge_index)
    affected_edges: set[int] = set()
    collapsed_chains = []
    previous_end = -1
    for group in groups:
        first_cut_start = _edge_endpoints(panel, group[0])[0]
        last_cut_end = _edge_endpoints(panel, group[-1])[1]
        baseline_start = numeric_vertices[first_cut_start]
        baseline_end = numeric_vertices[last_cut_end]
        baseline_x = baseline_end[0] - baseline_start[0]
        baseline_y = baseline_end[1] - baseline_start[1]
        baseline_length = math.hypot(baseline_x, baseline_y)
        if (
            not math.isfinite(baseline_length)
            or baseline_length <= _GEOMETRY_ATOL
        ):
            raise EffectMeasurementError(
                "Godet cut feet do not define a valid bottom baseline"
            )

        def on_baseline(vertex_index: int) -> bool:
            point = numeric_vertices[vertex_index]
            cross = (
                baseline_x * (point[1] - baseline_start[1])
                - baseline_y * (point[0] - baseline_start[0])
            )
            distance = abs(cross) / baseline_length
            return math.isfinite(distance) and distance <= _GEOMETRY_ATOL

        def is_residual(edge_index: int) -> bool:
            edge = _mapping(edges[edge_index], "panel edge")
            first, second = _edge_endpoints(panel, edge_index)
            return (
                edge_type(edge) == "line"
                and _edge_label(edge) is None
                and on_baseline(first)
                and on_baseline(second)
            )

        for internal_index in range(group[0], group[-1] + 1):
            if internal_index not in cut_edges and not is_residual(internal_index):
                raise EffectMeasurementError(
                    "Godet cut edges are separated by a non-baseline edge"
                )

        first_index = group[0]
        if first_index > 0:
            previous_index = first_index - 1
            previous_vertices = _edge_endpoints(panel, previous_index)
            if previous_vertices[1] == first_cut_start and is_residual(previous_index):
                first_index = previous_index

        last_index = group[-1]
        if last_index + 1 < len(edges):
            next_index = last_index + 1
            next_vertices = _edge_endpoints(panel, next_index)
            if last_cut_end == next_vertices[0] and is_residual(next_index):
                last_index = next_index

        if first_index <= previous_end:
            raise EffectMeasurementError("Godet cut chains overlap ambiguously")
        chain_indices = tuple(range(first_index, last_index + 1))
        for left_index, right_index in zip(chain_indices, chain_indices[1:]):
            left_end = _edge_endpoints(panel, left_index)[1]
            right_start = _edge_endpoints(panel, right_index)[0]
            if left_end != right_start:
                raise EffectMeasurementError(
                    "Godet cut chain is not a directed contiguous boundary run"
                )
        first_start = _edge_endpoints(panel, first_index)[0]
        last_end = _edge_endpoints(panel, last_index)[1]
        collapsed_chains.append(
            (
                canonical_json_bytes(vertices[first_start]),
                canonical_json_bytes(vertices[last_end]),
            )
        )
        affected_edges.update(chain_indices)
        previous_end = last_index

    static_panel = {
        key: value for key, value in panel.items() if key not in {"vertices", "edges"}
    }
    records = []
    for edge_index, raw in enumerate(edges):
        if edge_index in affected_edges:
            continue
        edge = _mapping(raw, "panel edge")
        first, second = _edge_endpoints(panel, edge_index)
        records.append(
            (
                canonical_json_bytes(vertices[first]),
                canonical_json_bytes(vertices[second]),
                canonical_json_bytes(
                    {key: value for key, value in edge.items() if key != "endpoints"}
                ),
            )
        )
    return (
        canonical_json_bytes(static_panel),
        tuple(sorted(records)),
        tuple(sorted(collapsed_chains)),
    )


def _godet_base_geometry_unchanged(
    source_panels: Mapping[str, Any],
    target_panels: Mapping[str, Any],
    source_attachments: _GodetAttachmentSummary,
    target_attachments: _GodetAttachmentSummary,
) -> tuple[bool, tuple[str, ...]]:
    if source_attachments.base_panels != target_attachments.base_panels:
        return False, ("base_panel_membership",)
    source_cut_edges = {
        name: frozenset(indices)
        for name, indices in source_attachments.cut_edges_by_panel
    }
    target_cut_edges = {
        name: frozenset(indices)
        for name, indices in target_attachments.cut_edges_by_panel
    }
    mismatches = []
    for panel_name in sorted(source_attachments.base_panels):
        source_panel = _mapping(source_panels.get(panel_name), panel_name)
        target_panel = _mapping(target_panels.get(panel_name), panel_name)
        if _godet_base_panel_records(
            source_panel,
            source_cut_edges.get(panel_name, frozenset()),
        ) != _godet_base_panel_records(
            target_panel,
            target_cut_edges.get(panel_name, frozenset()),
        ):
            mismatches.append(panel_name)
    return not mismatches, tuple(mismatches)


def _godet_nonbase_topology_unchanged(
    source_panels: Mapping[str, Any],
    target_panels: Mapping[str, Any],
    panel_names: set[str],
) -> tuple[bool, tuple[str, ...]]:
    mismatches = []
    for panel_name in sorted(panel_names):
        source_panel = _mapping(source_panels.get(panel_name), panel_name)
        target_panel = _mapping(target_panels.get(panel_name), panel_name)
        source_vertices = _sequence(
            source_panel.get("vertices"), f"{panel_name}.vertices"
        )
        target_vertices = _sequence(
            target_panel.get("vertices"), f"{panel_name}.vertices"
        )
        if (
            len(source_vertices),
            _edge_topology(source_panel),
        ) != (
            len(target_vertices),
            _edge_topology(target_panel),
        ):
            mismatches.append(panel_name)
    return not mismatches, tuple(mismatches)


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
    """Validate insert count, attachments, and unchanged unrelated topology."""

    del body
    if topology_policy not in {"strict", "local"}:
        raise ValueError("topology_policy must be exactly 'strict' or 'local'")
    operation = planned_operation.operation
    if operation.rule_id != "godet_insert_count":
        raise ValueError("operation names a foreign rule")
    try:
        rule_config = config.rules["godet_insert_count"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("config must contain godet_insert_count") from exc
    if operation.effect_metric_id != rule_config.effect_metric:
        raise ValueError("operation names the wrong Godet effect metric")
    if operation.effect_contract_id != rule_config.effect_contract:
        raise ValueError("operation names the wrong Godet effect contract")
    if operation.scope != "garment":
        raise ValueError("Godet operation scope must be 'garment'")
    if operation.direction not in {"increase", "decrease"}:
        raise ValueError("Godet direction must be exactly increase or decrease")

    try:
        source_count = design_dotted_get(source_design, GODET_COUNT_PATH)
        target_count = design_dotted_get(target_design, GODET_COUNT_PATH)
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or isinstance(target_count, bool)
            or not isinstance(target_count, int)
        ):
            raise EffectMeasurementError("Godet counts must be integers")
        source_base = design_dotted_get(source_design, "design.godet-skirt.base.v")
        target_base = design_dotted_get(target_design, "design.godet-skirt.base.v")
        source_panels = _panels(source_specification)
        target_panels = _panels(target_specification)
        source_inserts = {
            name for name in source_panels if GODET_PANEL.fullmatch(name)
        }
        target_inserts = {
            name for name in target_panels if GODET_PANEL.fullmatch(name)
        }
        source_unrelated = set(source_panels) - source_inserts
        target_unrelated = set(target_panels) - target_inserts
        source_attachments = _godet_attachment_summary(
            source_specification, source_inserts
        )
        target_attachments = _godet_attachment_summary(
            target_specification, target_inserts
        )
        source_nonbase = source_unrelated - source_attachments.base_panels
        target_nonbase = target_unrelated - target_attachments.base_panels
        nonbase_topology_equal, nonbase_topology_mismatches = (
            _godet_nonbase_topology_unchanged(
                source_panels,
                target_panels,
                source_nonbase.intersection(target_nonbase),
            )
        )
        if topology_policy == "strict":
            nonbase_equal: bool | None = all(
                _canonical_equal(source_panels[name], target_panels[name])
                for name in source_nonbase.intersection(target_nonbase)
            )
            base_geometry_equal, base_geometry_mismatches = (
                _godet_base_geometry_unchanged(
                    source_panels,
                    target_panels,
                    source_attachments,
                    target_attachments,
                )
            )
            metadata_match: bool | None = _canonical_equal(
                _spec_metadata(source_specification),
                _spec_metadata(target_specification),
            )
        else:
            nonbase_equal = None
            base_geometry_equal = None
            base_geometry_mismatches = ()
            metadata_match = None
        stitches_equal = _semantic_stitch_counter(
            source_specification,
            topology_only=topology_policy == "local",
        ) == _semantic_stitch_counter(
            target_specification,
            topology_only=topology_policy == "local",
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        ArithmeticError,
        EffectMeasurementError,
    ) as exc:
        return rule_effect_result(
            "godet_insert_count",
            False,
            "effect/measurement_error",
            {"message": str(exc)},
        )

    count_delta = target_count - source_count
    observed_delta = len(target_inserts) - len(source_inserts)
    direction_ok = (
        count_delta > 0 and operation.direction == "increase"
    ) or (
        count_delta < 0 and operation.direction == "decrease"
    )
    membership_direction_ok = (
        observed_delta > 0 and operation.direction == "increase"
    ) or (
        observed_delta < 0 and operation.direction == "decrease"
    )
    attachments_consistent = (
        source_attachments.valid
        and target_attachments.valid
        and source_attachments.base_panels == target_attachments.base_panels
        and source_attachments.uniform_degree == target_attachments.uniform_degree
        and source_attachments.peers_by_half == target_attachments.peers_by_half
    )
    details = {
        "source_insert_panel_count": len(source_inserts),
        "target_insert_panel_count": len(target_inserts),
        "insert_panel_delta": abs(observed_delta),
        "declared_count_delta": abs(count_delta),
        "base_dispatch_unchanged": _canonical_equal(source_base, target_base),
        "noninsert_panel_membership_unchanged": source_unrelated == target_unrelated,
        "base_attachment_panels_unchanged": (
            source_attachments.base_panels == target_attachments.base_panels
        ),
        "nonbase_panel_membership_unchanged": source_nonbase == target_nonbase,
        "nonbase_panels_unchanged": source_nonbase == target_nonbase and nonbase_equal,
        "nonbase_panel_topology_unchanged": (
            source_nonbase == target_nonbase and nonbase_topology_equal
        ),
        "nonbase_topology_mismatches": nonbase_topology_mismatches,
        "base_geometry_unchanged_except_insert_cut_edges": base_geometry_equal,
        "base_geometry_mismatches": base_geometry_mismatches,
        "noninsert_stitches_unchanged": stitches_equal,
        "noninsert_stitch_comparison": (
            "topology_only" if topology_policy == "local" else "exact_curvature_semantics"
        ),
        "insert_attachments_consistent": attachments_consistent,
        "source_insert_attachment_degree": source_attachments.uniform_degree,
        "target_insert_attachment_degree": target_attachments.uniform_degree,
        "metadata_match": metadata_match,
        "direction_ok": direction_ok and membership_direction_ok,
        "topology_policy": topology_policy,
    }
    core_checks = (
        len(source_inserts) == source_count,
        len(target_inserts) == target_count,
        abs(count_delta) >= 4,
        abs(observed_delta) == abs(count_delta),
        _canonical_equal(source_base, target_base),
        source_unrelated == target_unrelated,
        source_nonbase == target_nonbase,
        stitches_equal,
        attachments_consistent,
        direction_ok,
        membership_direction_ok,
    )
    passed = (
        all(core_checks)
        and nonbase_topology_equal
        and (
            topology_policy == "local"
            or all((nonbase_equal, base_geometry_equal, metadata_match))
        )
    )
    return rule_effect_result(
        "godet_insert_count",
        passed,
        "effect/passed" if passed else "effect/topology_mismatch",
        details,
    )


def build_rule(config: EditConfig) -> Rule:
    """Bind the complete Godet rule to one loaded configuration."""

    if not isinstance(config, EditConfig):
        raise TypeError("config must be an EditConfig")
    try:
        rule_config = config.rules["godet_insert_count"]
    except KeyError as exc:
        raise ValueError("config must contain godet_insert_count") from exc
    implementation = GodetInsertCountRule(rule_config)
    return Rule("godet_insert_count", implementation.candidates, apply, validate)


def build_rules(config: EditConfig) -> tuple[Rule, ...]:
    return (build_rule(config),)
