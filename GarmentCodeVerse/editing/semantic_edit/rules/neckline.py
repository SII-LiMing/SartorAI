"""Self-contained front-neckline semantic rule."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from editing.canonical_io import (
    canonical_hash,
    canonical_json_bytes,
    design_dotted_get,
    plain_copy,
)
from editing.config import ConfigError, EditConfig, ParameterSpec, RuleConfig
from editing.models import (
    EffectResult,
    Operation,
    OperationContractError,
    PlannedOperation,
    ProtectedStateSchemaError,
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
    specification_metadata,
    stitch_records,
)
from ..prompt_state import (
    ActivePathError,
    COMPONENT_STYLE_PATH,
    FRONT_NECKLINE_PATH,
    neckline_inputs,
    protected_neckline_phrase,
    resolve_active_paths,
)

NECKLINE_TYPES = (
    "CircleNeckHalf",
    "CurvyNeckHalf",
    "VNeckHalf",
    "SquareNeckHalf",
    "CircleArcNeckHalf",
    "Bezier2NeckHalf",
)

BACK_NECKLINE_PATH = "design.collar.b_collar.v"
FRONT_NECKLINE_READS = tuple(
    sorted(
        (
            "design.meta.upper.v",
            "design.left.enable_asym.v",
            "design.shirt.strapless.v",
            COMPONENT_STYLE_PATH,
            FRONT_NECKLINE_PATH,
            BACK_NECKLINE_PATH,
            "design.collar.width.v",
            "design.collar.fc_depth.v",
            "design.collar.bc_depth.v",
            "design.collar.fc_angle.v",
            "design.collar.bc_angle.v",
            "design.collar.f_bezier_x.v",
            "design.collar.f_bezier_y.v",
            "design.collar.b_bezier_x.v",
            "design.collar.b_bezier_y.v",
            "design.collar.f_flip_curve.v",
            "design.collar.b_flip_curve.v",
        )
    )
)

_WRITE_PATHS = (FRONT_NECKLINE_PATH,)
_CONFLICTS = ("collar_presence", "hood_presence")
_ENUM_PARAMETER_TYPES = {
    COMPONENT_STYLE_PATH: "select_null",
    FRONT_NECKLINE_PATH: "select",
    BACK_NECKLINE_PATH: "select",
}
_GEOMETRY_ATOL = 1.0e-7
_CHAIN_SAMPLES = 256


class EffectMeasurementError(ValueError):
    """A constructed specification cannot support the neckline metric."""


def _unavailable(code: str, message: str) -> StateUnavailableError:
    return StateUnavailableError(code, message)


def _number(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if positive and nonnegative:
        raise ValueError("positive and nonnegative are mutually exclusive")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EffectMeasurementError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise EffectMeasurementError(f"{field_name} must be finite") from exc
    if (
        not math.isfinite(result)
        or (positive and result <= 0.0)
        or (nonnegative and result < 0.0)
    ):
        suffix = (
            " finite and positive"
            if positive
            else " finite and non-negative"
            if nonnegative
            else " finite"
        )
        raise EffectMeasurementError(f"{field_name} must be{suffix}")
    return result


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EffectMeasurementError(f"{field_name} must be a mapping")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise EffectMeasurementError(f"{field_name} must be a sequence")
    return value


def _canonical_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _panels(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return _mapping(specification["pattern"]["panels"], "pattern.panels")
    except KeyError as exc:
        raise EffectMeasurementError("pattern.panels is required") from exc


def _units_scale_cm(specification: Mapping[str, Any]) -> float:
    try:
        units = specification["properties"]["units_in_meter"]
    except (KeyError, TypeError) as exc:
        raise EffectMeasurementError(
            "properties.units_in_meter is required"
        ) from exc
    scale = 100.0 / _number(
        units,
        "properties.units_in_meter",
        positive=True,
    )
    if not math.isfinite(scale):
        raise EffectMeasurementError("unit conversion scale must be finite")
    return scale


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


def _semantic_stitch_counter(
    specification: Mapping[str, Any],
    *,
    chain_edges: Mapping[str, frozenset[int]],
) -> Counter[Any]:
    result: Counter[Any] = Counter()
    for stitch in _stitch_records(specification):
        normalized = []
        for panel_name, edge_index in stitch[:2]:
            if edge_index in chain_edges.get(panel_name, frozenset()):
                relation: Any = "front_neckline_chain"
            else:
                edge = _edge(specification, panel_name, edge_index)
                relation = _edge_label(edge)
                if relation is None:
                    relation = (
                        edge_type(edge),
                        canonical_json_bytes(edge.get("curvature")),
                    )
            normalized.append((panel_name, relation))
        normalized.sort(key=repr)
        result[(normalized[0], normalized[1], stitch[2])] += 1
    return result


def _parameter_node(
    design: Mapping[str, Any],
    path: str,
) -> Mapping[str, Any] | None:
    parts = path.split(".")
    if not parts or parts[0] != "design" or parts[-1] != "v":
        raise OperationContractError(
            "enum parameter paths must be canonical design .v paths"
        )
    current: Any = design
    try:
        for part in parts[1:-1]:
            current = current[part]
    except (KeyError, TypeError):
        return None
    return current if isinstance(current, Mapping) else None


def _enum_metadata_is_exact(design: Mapping[str, Any]) -> bool:
    for path, expected_type in _ENUM_PARAMETER_TYPES.items():
        parameter = _parameter_node(design, path)
        if (
            parameter is None
            or "v" not in parameter
            or parameter.get("type") != expected_type
        ):
            return False
    return True


def _copy_enum_parameter(
    design: Mapping[str, Any],
    path: str,
    value: Any,
) -> dict[str, Any]:
    """Replace one whole enum parameter while preserving source provenance."""

    expected_type = _ENUM_PARAMETER_TYPES.get(path)
    parameter = _parameter_node(design, path)
    if expected_type is None:
        raise OperationContractError(f"undeclared enum write path: {path}")
    if parameter is None or "v" not in parameter:
        raise OperationContractError(
            f"candidate write path is absent from the design: {path}"
        )
    if parameter.get("type") != expected_type:
        raise OperationContractError(
            f"candidate write path has invalid enum metadata type: {path}"
        )
    updated = plain_copy(design)
    updated_parameter = _parameter_node(updated, path)
    if not isinstance(updated_parameter, dict):  # Already validated above.
        raise TypeError("enum parameter did not copy to a mapping")
    updated_parameter["v"] = value
    return updated


def _resolve_active(design: Mapping[str, Any]) -> Any:
    try:
        return resolve_active_paths(design)
    except (ActivePathError, KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "active_path_unavailable",
            f"cannot resolve active constructor paths: {exc}",
        ) from exc


def _visible_symmetric_upper(design: Mapping[str, Any]) -> None:
    active = _resolve_active(design)
    if active.dispatch.get("upper") is None:
        raise _unavailable(
            "neckline_inapplicable",
            "neckline state requires an active upper garment",
        )
    if active.dispatch.get("asymmetric") is True:
        raise _unavailable(
            "asymmetric_neckline_unrepresented",
            "asymmetric neckline is generic and its type is unrepresented",
        )
    if active.dispatch.get("upper") == "FittedShirt":
        try:
            strapless = design_dotted_get(
                design,
                "design.shirt.strapless.v",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _unavailable(
                "state_schema_unavailable",
                "active state path is missing or invalid: "
                "design.shirt.strapless.v",
            ) from exc
        if bool(strapless):
            raise _unavailable(
                "strapless_neckline_unrepresented",
                "strapless upper edge is not a neckline-type state",
            )


def _local_state(design: Mapping[str, Any]) -> SemanticState:
    _visible_symmetric_upper(design)
    inputs = neckline_inputs(design)
    if inputs["component"] is not None:
        raise _unavailable(
            "neckline_type_unrepresented",
            "a collar component suppresses the standalone neckline state",
        )
    constructor = inputs["constructor"]
    if type(constructor) is not str or constructor not in NECKLINE_TYPES:
        raise _unavailable(
            "neckline_type_unrepresented",
            "the front neckline constructor is outside the allowlist",
        )
    return SemanticState(
        rule_id="neckline_type",
        state_id=constructor,
        phrase=protected_neckline_phrase(inputs),
        attributes={"constructor": constructor},
    )


def _same_state(left: SemanticState, right: SemanticState) -> bool:
    return canonical_json_bytes(left.to_payload()) == canonical_json_bytes(
        right.to_payload()
    )


def _exact_protected(
    local: SemanticState,
    protected: Any,
    endpoint: str,
) -> SemanticState:
    if type(protected) is not SemanticState:
        raise TypeError(
            f"neckline_type protected {endpoint} state must be an exact "
            "SemanticState"
        )
    if not _same_state(local, protected):
        expected = canonical_json_bytes(local.to_payload()).decode("utf-8")
        observed = canonical_json_bytes(protected.to_payload()).decode("utf-8")
        raise ProtectedStateSchemaError(
            "protected_state_mismatch",
            "neckline_type protected "
            f"{endpoint} state mismatch: expected={expected}, observed={observed}",
        )
    return protected


def _all_reads_exist(design: Mapping[str, Any]) -> bool:
    try:
        for path in FRONT_NECKLINE_READS:
            design_dotted_get(design, path)
    except KeyError:
        return False
    return True


def _applicable(context: RuleContext) -> bool:
    if not _all_reads_exist(context.design) or not _enum_metadata_is_exact(
        context.design
    ):
        return False
    dispatch = context.active_paths.dispatch
    if dispatch.get("upper") not in {"Shirt", "FittedShirt"}:
        return False
    if (
        dispatch.get("asymmetric") is not False
        or dispatch.get("collar_component_suppressed") is not False
    ):
        return False
    return not (
        dispatch.get("upper") == "FittedShirt"
        and design_dotted_get(context.design, "design.shirt.strapless.v")
        is not False
    )


def _source_states(
    context: RuleContext,
) -> tuple[SemanticState, SemanticState] | None:
    protected = context.prompt_states.get("neckline_type")
    if protected is None:
        return None
    try:
        local = _local_state(context.design)
    except StateUnavailableError as exc:
        if exc.code != "neckline_type_unrepresented":
            raise
        return None
    return local, _exact_protected(local, protected, "source")


def _target_states(
    context: RuleContext,
    target_design: Mapping[str, Any],
) -> tuple[SemanticState, SemanticState] | None:
    try:
        protected_states = context.classify_prompt(
            plain_copy(target_design),
            context.raw_body_yaml,
        )
    except StateUnavailableError as exc:
        if exc.code != "neckline_type_unrepresented":
            raise
        return None
    if not isinstance(protected_states, Mapping):
        raise TypeError("protected state oracle must return a mapping")
    protected = protected_states.get("neckline_type")
    if protected is None:
        return None
    try:
        local = _local_state(target_design)
    except StateUnavailableError as exc:
        if exc.code != "neckline_type_unrepresented":
            raise
        return None
    return local, _exact_protected(local, protected, "target")


def _validate_rule_config(config: RuleConfig) -> ParameterSpec:
    if type(config) is not RuleConfig or config.enabled is not True:
        raise ConfigError("neckline_type rule_config must be enabled")
    if (
        config.semantic_group != "collar"
        or type(config.structural_level) is not int
        or config.structural_level != 1
        or config.effect_metric != "bilateral_front_neckline_cm"
        or config.effect_contract != "boundary_topology_change"
        or set(config.parameter_specs) != {"front_type"}
    ):
        raise ConfigError("neckline_type rule_config has an invalid operation contract")
    parameter = config.parameter_specs["front_type"]
    if (
        type(parameter) is not ParameterSpec
        or parameter.edit_mode != "enum"
        or parameter.value_type != "enum"
        or parameter.transitions != NECKLINE_TYPES
    ):
        raise ConfigError(
            "neckline_type front_type config has invalid transitions"
        )
    return parameter


def _planned_operation(
    context: RuleContext,
    rule_config: RuleConfig,
    parameter: ParameterSpec,
    target_design: Mapping[str, Any],
    source_local: SemanticState,
    source_protected: SemanticState,
    target_local: SemanticState,
    target_protected: SemanticState,
) -> PlannedOperation:
    source_constructor = source_local.state_id
    target_constructor = target_local.state_id
    operation = Operation(
        rule_id="neckline_type",
        semantic_group=rule_config.semantic_group,
        scope="front",
        direction="change",
        magnitude="structural",
        structural_level=rule_config.structural_level,
        reads=FRONT_NECKLINE_READS,
        writes=_WRITE_PATHS,
        conflicts=_CONFLICTS,
        prerequisites=("component_style_is_null", "symmetric_upper"),
        edit_mode=parameter.edit_mode,
        effect_metric_id=rule_config.effect_metric,
        effect_contract_id=rule_config.effect_contract,
        before={FRONT_NECKLINE_PATH: source_constructor},
        after={FRONT_NECKLINE_PATH: target_constructor},
        relative_change=None,
        requested_magnitude="structural",
        realized_magnitude="structural",
        clamped=False,
        instruction_tokens={
            "feature": "the neckline",
            "source_state_id": source_constructor,
            "target_state_id": target_constructor,
        },
    )
    transition = SemanticTransition(
        rule_id="neckline_type",
        scope="front",
        channels=("front",),
        feature_phrase="the neckline",
        source_state=source_local,
        target_state=target_local,
        direction="change",
    )
    evidence = {
        "rule_id": operation.rule_id,
        "scope": operation.scope,
        "source_design_hash": canonical_hash(context.design),
        "target_design_hash": canonical_hash(target_design),
        "body_hash": canonical_hash(context.body),
        "active_paths_design_hash": context.active_paths.design_hash,
        "upper_constructor": context.active_paths.dispatch.get("upper"),
        "symmetric_upper": context.active_paths.dispatch.get("asymmetric") is False,
        "component_style": design_dotted_get(
            context.design, COMPONENT_STYLE_PATH
        ),
        "stored_front_neckline": design_dotted_get(
            context.design, FRONT_NECKLINE_PATH
        ),
        "stored_back_neckline": design_dotted_get(
            context.design, BACK_NECKLINE_PATH
        ),
        "read_contract_hash": canonical_hash(
            {
                path: design_dotted_get(context.design, path)
                for path in operation.reads
            }
        ),
        "source_state": source_local.to_payload(),
        "target_state": target_local.to_payload(),
        "source_local_state": source_local.to_payload(),
        "source_protected_state": source_protected.to_payload(),
        "target_local_state": target_local.to_payload(),
        "target_protected_state": target_protected.to_payload(),
        "source_state_match": True,
        "target_state_match": True,
    }
    return PlannedOperation(
        operation=operation,
        transitions=(transition,),
        topology_kind=TopologyKind.BOUNDARY,
        applicability_evidence=evidence,
        conflict_evidence=empty_conflict_evidence(
            operation.rule_id,
            operation.conflicts,
        ),
    )


def _candidates(
    context: RuleContext,
    rule_config: RuleConfig,
    parameter: ParameterSpec,
) -> tuple[PlannedOperation, ...]:
    if not _applicable(context):
        return ()
    try:
        style = design_dotted_get(context.design, COMPONENT_STYLE_PATH)
        source_constructor = design_dotted_get(
            context.design, FRONT_NECKLINE_PATH
        )
    except KeyError:
        return ()
    if style is not None or source_constructor not in NECKLINE_TYPES:
        return ()
    source = _source_states(context)
    if source is None or source[0].state_id != source_constructor:
        return ()
    source_local, source_protected = source

    result = []
    for target_constructor in parameter.transitions:
        if target_constructor == source_constructor:
            continue
        target_design = _copy_enum_parameter(
            context.design,
            FRONT_NECKLINE_PATH,
            target_constructor,
        )
        target = _target_states(context, target_design)
        if target is None:
            return ()
        target_local, target_protected = target
        if target_local.state_id != target_constructor:
            return ()
        if target_local.phrase == source_local.phrase:
            continue
        result.append(
            _planned_operation(
                context,
                rule_config,
                parameter,
                target_design,
                source_local,
                source_protected,
                target_local,
                target_protected,
            )
        )
    return tuple(result)


def apply(
    design: dict[str, Any],
    planned: PlannedOperation,
) -> dict[str, Any]:
    """Apply the configured constructor without consulting source range."""

    return _copy_enum_parameter(
        design,
        FRONT_NECKLINE_PATH,
        planned.operation.after[FRONT_NECKLINE_PATH],
    )


@dataclass(frozen=True)
class _NecklineChain:
    panel_name: str
    edge_indices: tuple[int, ...]
    directed_edges: tuple[tuple[int, bool], ...]
    center_vertex: int
    shoulder_vertex: int
    center_adjacency: tuple[Any, ...]
    shoulder_adjacency: tuple[Any, ...]


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


def _adjacency_signature(
    panel: Mapping[str, Any],
    vertex: int,
    chain_edges: frozenset[int],
) -> tuple[Any, ...]:
    records = []
    for index, raw in enumerate(_sequence(panel.get("edges"), "panel.edges")):
        if index in chain_edges:
            continue
        endpoints = _edge_endpoints(panel, index)
        if vertex in endpoints:
            edge = _mapping(raw, "panel edge")
            records.append((_edge_label(edge), edge_type(edge)))
    return tuple(sorted(records, key=repr))


def _anchor_vertices_from_stitches(
    specification: Mapping[str, Any],
    *,
    panel_name: str,
    peer_panel_name: str,
    chain_endpoints: frozenset[int],
    chain_edges: frozenset[int],
) -> tuple[int, ...]:
    result: list[int] = []
    panel = _mapping(_panels(specification)[panel_name], panel_name)
    for stitch in _stitch_records(specification):
        endpoints = stitch[:2]
        for index, (name, edge_index) in enumerate(endpoints):
            if name != panel_name or edge_index in chain_edges:
                continue
            peer_name, peer_edge_index = endpoints[1 - index]
            if peer_name != peer_panel_name:
                continue
            _edge(specification, peer_name, peer_edge_index)
            candidates = chain_endpoints.intersection(
                _edge_endpoints(panel, edge_index)
            )
            if len(candidates) == 1:
                result.append(next(iter(candidates)))
    return tuple(result)


def _extract_neckline_chain(
    specification: Mapping[str, Any],
    side: str,
) -> _NecklineChain:
    panel_name = f"{side}_ftorso"
    panel = _mapping(_panels(specification).get(panel_name), panel_name)
    label = f"{side}_collar"
    raw_edges = _sequence(panel.get("edges"), f"{panel_name}.edges")
    labelled: list[int] = []
    endpoint_pairs: set[frozenset[int]] = set()
    adjacency: dict[int, list[int]] = defaultdict(list)
    for index, raw in enumerate(raw_edges):
        edge = _mapping(raw, f"{panel_name}.edges[{index}]")
        edge_label = _edge_label(edge)
        if edge_label != label:
            continue
        first, second = _edge_endpoints(panel, index)
        pair = frozenset((first, second))
        if pair in endpoint_pairs:
            raise EffectMeasurementError("duplicate neckline edge use")
        endpoint_pairs.add(pair)
        labelled.append(index)
        adjacency[first].append(index)
        adjacency[second].append(index)
    if not labelled:
        raise EffectMeasurementError(f"missing {label} edge label")
    if any(len(edges) > 2 for edges in adjacency.values()):
        raise EffectMeasurementError("front neckline edge chain is branched")
    endpoints = frozenset(
        vertex for vertex, edges in adjacency.items() if len(edges) == 1
    )
    if len(endpoints) != 2:
        raise EffectMeasurementError(
            "front neckline chain must have exactly two endpoints"
        )
    chain_edges = frozenset(labelled)

    # Prove connectedness without trusting serialized edge order.
    pending = [labelled[0]]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        first, second = _edge_endpoints(panel, current)
        pending.extend(adjacency[first])
        pending.extend(adjacency[second])
    if visited != chain_edges:
        raise EffectMeasurementError("front neckline chain is disconnected")

    shoulder_candidates = _anchor_vertices_from_stitches(
        specification,
        panel_name=panel_name,
        peer_panel_name=f"{side}_btorso",
        chain_endpoints=endpoints,
        chain_edges=chain_edges,
    )
    opposite = "right" if side == "left" else "left"
    center_candidates = _anchor_vertices_from_stitches(
        specification,
        panel_name=panel_name,
        peer_panel_name=f"{opposite}_ftorso",
        chain_endpoints=endpoints,
        chain_edges=chain_edges,
    )
    if len(shoulder_candidates) != 1:
        raise EffectMeasurementError(
            "neckline requires exactly one canonical shoulder stitch anchor"
        )
    if len(center_candidates) != 1:
        raise EffectMeasurementError(
            "neckline requires exactly one canonical center-front stitch anchor"
        )
    shoulder = shoulder_candidates[0]
    center = center_candidates[0]
    if shoulder == center or frozenset((shoulder, center)) != endpoints:
        raise EffectMeasurementError(
            "shoulder and center-front relations must anchor the full neckline chain"
        )

    directed: list[tuple[int, bool]] = []
    current_vertex = center
    unused = set(labelled)
    while unused:
        candidates = [
            index
            for index in adjacency[current_vertex]
            if index in unused
        ]
        if len(candidates) != 1:
            raise EffectMeasurementError(
                "neckline chain cannot be ordered uniquely"
            )
        edge_index = candidates[0]
        first, second = _edge_endpoints(panel, edge_index)
        forward = first == current_vertex
        directed.append((edge_index, forward))
        unused.remove(edge_index)
        current_vertex = second if forward else first
    if current_vertex != shoulder:
        raise EffectMeasurementError(
            "neckline chain does not terminate at the shoulder"
        )
    return _NecklineChain(
        panel_name=panel_name,
        edge_indices=tuple(sorted(labelled)),
        directed_edges=tuple(directed),
        center_vertex=center,
        shoulder_vertex=shoulder,
        center_adjacency=_adjacency_signature(panel, center, chain_edges),
        shoulder_adjacency=_adjacency_signature(panel, shoulder, chain_edges),
    )


def _validate_serialized_curvature(edge: Mapping[str, Any]) -> None:
    if "curvature" not in edge:
        return
    curvature = edge["curvature"]
    if isinstance(curvature, Mapping):
        curve_type = curvature.get("type")
        parameters = _sequence(curvature.get("params"), "curvature.params")
    else:
        curve_type = "quadratic"
        parameters = (curvature,)

    if curve_type == "circle" and len(parameters) == 3:
        _number(parameters[0], "circle.radius", positive=True)
        if any(
            type(flag) is not int or flag not in {0, 1}
            for flag in parameters[1:]
        ):
            raise EffectMeasurementError(
                "circle arc flags must be exact integer 0 or 1 values"
            )
        return

    if curve_type == "quadratic":
        expected_controls = 1
    elif curve_type == "cubic":
        expected_controls = 2
    else:
        raise EffectMeasurementError(
            f"cannot reconstruct {curve_type!r} neckline curvature"
        )
    if len(parameters) != expected_controls:
        raise EffectMeasurementError(
            f"cannot reconstruct {curve_type!r} neckline curvature"
        )
    for control in parameters:
        point = _sequence(control, "relative curve control point")
        if len(point) != 2:
            raise EffectMeasurementError(
                "relative curve control points must be 2-D"
            )
        _number(point[0], "relative_control.x")
        _number(point[1], "relative_control.y")


def _sample_edge(
    panel: Mapping[str, Any],
    edge_index: int,
    *,
    forward: bool,
) -> tuple[tuple[float, float], ...]:
    vertices = _panel_vertices(panel)
    raw = _sequence(panel.get("edges"), "panel.edges")[edge_index]
    edge = _mapping(raw, "panel edge")
    _edge_endpoints(panel, edge_index)
    _validate_serialized_curvature(edge)
    sample_count = 65 if "curvature" in edge else 33

    # Keep the curve stack out of lightweight command imports.
    from editing.pattern_geometry import edge_as_curve

    try:
        curve = edge_as_curve(vertices, edge)
        sampled = []
        for index in range(sample_count):
            point = complex(curve.point(index / (sample_count - 1)))
            sampled.append(
                (
                    _number(point.real, "curve_sample.x"),
                    _number(point.imag, "curve_sample.y"),
                )
            )
        points = tuple(sampled)
    except EffectMeasurementError:
        raise
    except (KeyError, TypeError, ValueError, NotImplementedError) as error:
        raise EffectMeasurementError(
            "cannot reconstruct serialized neckline curvature"
        ) from error
    return points if forward else tuple(reversed(points))


def _polyline_resample(
    points: Sequence[tuple[float, float]],
    count: int,
) -> tuple[tuple[float, float], ...]:
    if len(points) < 2:
        raise EffectMeasurementError("neckline chain has too few points")
    distances = [0.0]
    for first, second in zip(points, points[1:]):
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        distances.append(distances[-1] + length)
    total = distances[-1]
    if not math.isfinite(total) or total <= 0.0:
        raise EffectMeasurementError("neckline chain has zero or invalid length")
    result = []
    segment = 0
    for index in range(count):
        target = total * index / (count - 1)
        while segment + 1 < len(distances) and distances[segment + 1] < target:
            segment += 1
        if segment + 1 >= len(points):
            result.append(points[-1])
            continue
        span = distances[segment + 1] - distances[segment]
        fraction = 0.0 if span == 0.0 else (target - distances[segment]) / span
        first = points[segment]
        second = points[segment + 1]
        result.append(
            (
                first[0] + (second[0] - first[0]) * fraction,
                first[1] + (second[1] - first[1]) * fraction,
            )
        )
    return tuple(result)


def _chain_points_cm(
    specification: Mapping[str, Any],
    chain: _NecklineChain,
) -> tuple[tuple[float, float], ...]:
    panel = _mapping(_panels(specification)[chain.panel_name], chain.panel_name)
    points: list[tuple[float, float]] = []
    for edge_index, forward in chain.directed_edges:
        sampled = _sample_edge(panel, edge_index, forward=forward)
        points.extend(sampled if not points else sampled[1:])
    scale = _units_scale_cm(specification)
    return _polyline_resample(
        tuple((x * scale, y * scale) for x, y in points),
        _CHAIN_SAMPLES,
    )


def _aligned_max_displacement(
    source: Sequence[tuple[float, float]],
    target: Sequence[tuple[float, float]],
) -> float:
    if len(source) != _CHAIN_SAMPLES or len(target) != _CHAIN_SAMPLES:
        raise EffectMeasurementError("neckline resampling count drifted")
    source_vector = (
        source[-1][0] - source[0][0],
        source[-1][1] - source[0][1],
    )
    target_vector = (
        target[-1][0] - target[0][0],
        target[-1][1] - target[0][1],
    )
    source_length = math.hypot(*source_vector)
    target_length = math.hypot(*target_vector)
    if (
        not math.isfinite(source_length)
        or not math.isfinite(target_length)
        or source_length <= _GEOMETRY_ATOL
        or target_length <= _GEOMETRY_ATOL
    ):
        raise EffectMeasurementError(
            "neckline endpoints do not define a rigid alignment"
        )
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1],
        source_vector[0],
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = 0.0
    for source_point, target_point in zip(source, target):
        relative_x = source_point[0] - source[0][0]
        relative_y = source_point[1] - source[0][1]
        aligned = (
            target[0][0] + cosine * relative_x - sine * relative_y,
            target[0][1] + sine * relative_x + cosine * relative_y,
        )
        displacement = math.hypot(
            aligned[0] - target_point[0],
            aligned[1] - target_point[1],
        )
        if not math.isfinite(displacement):
            raise EffectMeasurementError(
                "neckline displacement must be finite"
            )
        result = max(result, displacement)
    if not math.isfinite(result):
        raise EffectMeasurementError("neckline displacement must be finite")
    return result


def _non_chain_edge_records(
    panel: Mapping[str, Any],
    chain: _NecklineChain,
) -> tuple[Any, ...]:
    vertices = _sequence(panel.get("vertices"), "panel.vertices")
    _panel_vertices(panel)
    anchor_tokens = {
        chain.center_vertex: "NECK_CENTER",
        chain.shoulder_vertex: "NECK_SHOULDER",
    }

    def coordinate_record(vertex_index: int) -> tuple[str, Any]:
        token = anchor_tokens.get(vertex_index)
        if token is not None:
            return ("anchor", token)
        return ("coordinate", canonical_json_bytes(vertices[vertex_index]))

    records = []
    for index, raw in enumerate(_sequence(panel.get("edges"), "panel.edges")):
        if index in chain.edge_indices:
            continue
        edge = _mapping(raw, "panel edge")
        first, second = _edge_endpoints(panel, index)
        endpoint_coordinates = (
            coordinate_record(first),
            coordinate_record(second),
        )
        records.append(
            (
                endpoint_coordinates,
                canonical_json_bytes(
                    {
                        key: value
                        for key, value in edge.items()
                        if key != "endpoints"
                    }
                ),
            )
        )
    return tuple(sorted(records, key=repr))


def _neckline_anchor_coordinate_drift(
    source_panel: Mapping[str, Any],
    target_panel: Mapping[str, Any],
    source_chain: _NecklineChain,
    target_chain: _NecklineChain,
) -> tuple[bool, Mapping[str, float]]:
    source_vertices = _panel_vertices(source_panel)
    target_vertices = _panel_vertices(target_panel)
    drift: dict[str, float] = {}
    for name, source_index, target_index in (
        (
            "center",
            source_chain.center_vertex,
            target_chain.center_vertex,
        ),
        (
            "shoulder",
            source_chain.shoulder_vertex,
            target_chain.shoulder_vertex,
        ),
    ):
        source_point = source_vertices[source_index]
        target_point = target_vertices[target_index]
        distance = math.hypot(
            target_point[0] - source_point[0],
            target_point[1] - source_point[1],
        )
        if not math.isfinite(distance):
            raise EffectMeasurementError(
                "neckline anchor coordinate drift must be finite"
            )
        drift[name] = distance
    return (
        all(value <= _GEOMETRY_ATOL for value in drift.values()),
        drift,
    )


def _neckline_topology_unchanged(
    source_specification: Mapping[str, Any],
    target_specification: Mapping[str, Any],
    source_chains: Mapping[str, _NecklineChain],
    target_chains: Mapping[str, _NecklineChain],
) -> tuple[bool, Mapping[str, Any]]:
    source_panels = _panels(source_specification)
    target_panels = _panels(target_specification)
    if set(source_panels) != set(target_panels):
        return False, {"complete_panel_membership_match": False}
    if not _canonical_equal(
        _spec_metadata(source_specification),
        _spec_metadata(target_specification),
    ):
        return False, {"metadata_match": False}
    front_names = {"left_ftorso", "right_ftorso"}
    unrelated_mismatch = []
    for name in sorted(set(source_panels) - front_names):
        if not _canonical_equal(source_panels[name], target_panels[name]):
            unrelated_mismatch.append(name)
    front_mismatch = []
    anchor_mismatch = []
    anchor_coordinate_mismatch = []
    anchor_coordinate_drift: dict[str, Mapping[str, float]] = {}
    for side in ("left", "right"):
        name = f"{side}_ftorso"
        source_panel = _mapping(source_panels.get(name), name)
        target_panel = _mapping(target_panels.get(name), name)
        source_static = {
            key: value
            for key, value in source_panel.items()
            if key not in {"vertices", "edges"}
        }
        target_static = {
            key: value
            for key, value in target_panel.items()
            if key not in {"vertices", "edges"}
        }
        if not _canonical_equal(source_static, target_static):
            front_mismatch.append(f"{name}:panel_metadata")
        if _non_chain_edge_records(
            source_panel,
            source_chains[side],
        ) != _non_chain_edge_records(
            target_panel,
            target_chains[side],
        ):
            front_mismatch.append(f"{name}:non_chain_edges")
        if (
            source_chains[side].center_adjacency
            != target_chains[side].center_adjacency
            or source_chains[side].shoulder_adjacency
            != target_chains[side].shoulder_adjacency
        ):
            anchor_mismatch.append(side)
        coordinates_match, coordinate_drift = (
            _neckline_anchor_coordinate_drift(
                source_panel,
                target_panel,
                source_chains[side],
                target_chains[side],
            )
        )
        anchor_coordinate_drift[side] = coordinate_drift
        if not coordinates_match:
            anchor_coordinate_mismatch.append(side)
    source_chain_edges = {
        chain.panel_name: frozenset(chain.edge_indices)
        for chain in source_chains.values()
    }
    target_chain_edges = {
        chain.panel_name: frozenset(chain.edge_indices)
        for chain in target_chains.values()
    }
    stitch_match = _semantic_stitch_counter(
        source_specification,
        chain_edges=source_chain_edges,
    ) == _semantic_stitch_counter(
        target_specification,
        chain_edges=target_chain_edges,
    )
    details = {
        "complete_panel_membership_match": True,
        "unrelated_panel_mismatches": unrelated_mismatch,
        "front_non_chain_mismatches": front_mismatch,
        "chain_anchor_mismatches": anchor_mismatch,
        "chain_anchor_coordinate_mismatches": anchor_coordinate_mismatch,
        "chain_anchor_coordinate_drift": anchor_coordinate_drift,
        "chain_anchor_coordinate_atol": _GEOMETRY_ATOL,
        "normalized_stitch_relations_match": stitch_match,
    }
    return (
        not unrelated_mismatch
        and not front_mismatch
        and not anchor_mismatch
        and not anchor_coordinate_mismatch
        and stitch_match,
        details,
    )


def bilateral_front_neckline_metric_cm(
    source_specification: Mapping[str, Any],
    target_specification: Mapping[str, Any],
) -> tuple[
    float,
    Mapping[str, float],
    Mapping[str, _NecklineChain],
    Mapping[str, _NecklineChain],
]:
    source_chains = {
        side: _extract_neckline_chain(source_specification, side)
        for side in ("left", "right")
    }
    target_chains = {
        side: _extract_neckline_chain(target_specification, side)
        for side in ("left", "right")
    }
    maxima = {}
    for side in ("left", "right"):
        maxima[side] = _aligned_max_displacement(
            _chain_points_cm(source_specification, source_chains[side]),
            _chain_points_cm(target_specification, target_chains[side]),
        )
    metric = min(maxima.values())
    if (
        not math.isfinite(metric)
        or any(not math.isfinite(value) for value in maxima.values())
    ):
        raise EffectMeasurementError(
            "bilateral neckline metric must be finite"
        )
    return (
        metric,
        maxima,
        source_chains,
        target_chains,
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
    """Validate one bilateral front-neckline constructor change."""

    del source_design, target_design, body
    if topology_policy not in {"strict", "local"}:
        raise ValueError("topology_policy must be exactly 'strict' or 'local'")
    operation = planned_operation.operation
    if operation.rule_id != "neckline_type":
        raise ValueError("operation names a foreign rule")
    try:
        rule_config = config.rules["neckline_type"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("config must contain neckline_type") from exc
    if operation.effect_metric_id != rule_config.effect_metric:
        raise ValueError("operation names the wrong neckline effect metric")
    if operation.effect_contract_id != rule_config.effect_contract:
        raise ValueError("operation names the wrong neckline effect contract")
    if operation.scope != "front":
        raise ValueError("neckline operation scope must be 'front'")
    if operation.direction != "change":
        raise ValueError("neckline operation direction must be 'change'")

    try:
        metric, maxima, source_chains, target_chains = (
            bilateral_front_neckline_metric_cm(
                source_specification,
                target_specification,
            )
        )
        topology_ok, topology_details = _neckline_topology_unchanged(
            source_specification,
            target_specification,
            source_chains,
            target_chains,
        )
        threshold = _number(
            rule_config.effective_metric_threshold_cm,
            "config.rules.neckline_type.effective_metric_threshold_cm",
            nonnegative=True,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        ArithmeticError,
        EffectMeasurementError,
    ) as exc:
        return rule_effect_result(
            "neckline_type",
            passed=False,
            code="effect/measurement_error",
            details={"message": str(exc)},
        )

    details = {
        "left_max_cm": maxima["left"],
        "right_max_cm": maxima["right"],
        "bilateral_front_neckline_cm": metric,
        "threshold_cm": threshold,
        "resample_points": _CHAIN_SAMPLES,
        "topology": topology_details,
        "topology_policy": topology_policy,
    }
    if topology_policy == "strict" and not topology_ok:
        return rule_effect_result(
            "neckline_type",
            passed=False,
            code="effect/topology_mismatch",
            details=details,
        )
    passed = metric >= threshold
    return rule_effect_result(
        "neckline_type",
        passed=passed,
        code="effect/passed" if passed else "effect/metric_threshold",
        details=details,
    )


def build_rule(config: EditConfig) -> Rule:
    """Bind the neckline rule to loaded configuration."""

    if not isinstance(config, EditConfig):
        raise TypeError("config must be an EditConfig")
    try:
        rule_config = config.rules["neckline_type"]
    except KeyError as exc:
        raise ValueError("config must contain neckline_type") from exc
    parameter = _validate_rule_config(rule_config)

    def candidates(context: RuleContext) -> tuple[PlannedOperation, ...]:
        return _candidates(context, rule_config, parameter)

    return Rule("neckline_type", candidates, apply, validate)


def build_rules(config: EditConfig) -> tuple[Rule, ...]:
    return (build_rule(config),)
