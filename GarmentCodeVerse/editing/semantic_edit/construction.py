"""Side-effect-isolated GarmentCode target construction.

Construct patterns from detached design/body copies, then restore source
panels that the edit leaves unchanged relative to a no-edit reconstruction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from editing.canonical_io import (
    CanonicalDataError,
    canonical_json_bytes,
    load_body_yaml_bytes,
    plain_copy,
    sha256_bytes,
)
from pattern_sampler import assert_param_combinations
from editing.pattern_compare import compare_patterns
from editing.pattern_geometry import edge_topology


_EDGE_LENGTH_TOLERANCE = 1e-9
_POLYGON_AREA_TOLERANCE = _EDGE_LENGTH_TOLERANCE**2
_CURVE_AREA_SAMPLE_COUNT = 16
PANEL_COMPARISON_ATOL = 1e-9


class InvalidTargetSpecification(ValueError):
    """The constructed target is not a usable canonical pattern."""


class InvalidStitchReference(InvalidTargetSpecification):
    """A stitch endpoint does not reference a serialized panel edge."""


def instantiate_body_parameters(body_parameters_cls: type) -> Any:
    """Instantiate the legacy body wrapper without triggering ``open('')``."""

    if not isinstance(body_parameters_cls, type):
        raise TypeError("body_parameters_cls must be a class")
    try:
        return body_parameters_cls()
    except FileNotFoundError as error:
        if error.filename not in {"", b""}:
            raise

    body = body_parameters_cls.__new__(body_parameters_cls)
    if not isinstance(body, body_parameters_cls):
        raise TypeError(
            "body_parameters_cls.__new__ returned an invalid instance"
        )
    body.params = {}
    return body


@dataclass(frozen=True, slots=True)
class ConstructionDependencies:
    """Lazy GarmentCode constructor boundary, injectable in unit tests."""

    body_parameters_cls: type
    meta_garment_cls: type
    domain_errors: tuple[type[BaseException], ...]
    validate_design: Callable[[Mapping[str, Any]], None] = (
        assert_param_combinations
    )

    def __post_init__(self) -> None:
        if not isinstance(self.body_parameters_cls, type):
            raise TypeError("body_parameters_cls must be a class")
        if not isinstance(self.meta_garment_cls, type):
            raise TypeError("meta_garment_cls must be a class")
        if not callable(self.validate_design):
            raise TypeError("validate_design must be callable")
        if not isinstance(self.domain_errors, tuple) or not self.domain_errors:
            raise TypeError("domain_errors must be a non-empty tuple")

        normalized: list[type[BaseException]] = []
        for error_type in self.domain_errors:
            if not isinstance(error_type, type) or not issubclass(
                error_type, BaseException
            ):
                raise TypeError(
                    "domain_errors must contain exception classes"
                )
            # A broad BaseException-like entry would also consume process
            # control exceptions. Reject it at dependency construction time.
            if (
                issubclass(KeyboardInterrupt, error_type)
                or issubclass(SystemExit, error_type)
                or issubclass(error_type, KeyboardInterrupt)
                or issubclass(error_type, SystemExit)
            ):
                raise TypeError(
                    "domain_errors cannot match KeyboardInterrupt or SystemExit"
                )
            if error_type not in normalized:
                normalized.append(error_type)
        object.__setattr__(self, "domain_errors", tuple(normalized))


@dataclass(frozen=True, slots=True)
class ConstructionResult:
    """Frozen record for one target-only construction attempt."""

    success: bool
    specification: Mapping[str, Any] | None
    specification_json_bytes: bytes | None
    specification_sha256: str | None
    failure_code: str | None
    failure_message: str | None

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise TypeError("success must be an exact bool")

        if self.success:
            if self.failure_code is not None or self.failure_message is not None:
                raise ValueError("successful result cannot contain a failure")
            if not isinstance(self.specification, Mapping):
                raise TypeError(
                    "successful specification must be a canonical mapping"
                )
            specification = plain_copy(
                self.specification,
                tuples_as_lists=True,
            )
            payload = self.specification_json_bytes
            if type(payload) is not bytes or not payload:
                raise TypeError(
                    "successful result requires immutable specification bytes"
                )
            payload = memoryview(payload).tobytes()
            expected = canonical_json_bytes(specification)
            if payload != expected:
                raise ValueError(
                    "specification_bytes are not the canonical specification"
                )
            expected_hash = sha256_bytes(payload)
            if self.specification_sha256 != expected_hash:
                raise ValueError(
                    "specification_sha256 does not match bytes"
                )
            object.__setattr__(self, "specification", specification)
            object.__setattr__(self, "specification_json_bytes", payload)
            return

        if any(
            value is not None
            for value in (
                self.specification,
                self.specification_json_bytes,
                self.specification_sha256,
            )
        ):
            raise ValueError(
                "failed result cannot retain specification artifacts"
            )
        for field_name in ("failure_code", "failure_message"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or not value
                or value != value.strip()
            ):
                raise ValueError(
                    f"failed result requires non-empty trimmed {field_name}"
                )


def _default_dependencies() -> ConstructionDependencies:
    """Import heavy GarmentCode classes only for a real construction."""

    from assets.bodies.body_params import BodyParameters
    from assets.garment_programs import meta_garment
    from pygarment import EmptyPatternError

    return ConstructionDependencies(
        body_parameters_cls=BodyParameters,
        meta_garment_cls=meta_garment.MetaGarment,
        domain_errors=(
            EmptyPatternError,
            meta_garment.IncorrectElementConfiguration,
            meta_garment.TotalLengthError,
        ),
    )


def _failure(
    code: str,
    message: str | BaseException,
) -> ConstructionResult:
    if isinstance(message, BaseException):
        detail = str(message)
        message = (
            f"{type(message).__name__}: {detail}"
            if detail
            else type(message).__name__
        )
    return ConstructionResult(
        success=False,
        specification=None,
        specification_json_bytes=None,
        specification_sha256=None,
        failure_code=code,
        failure_message=message,
    )


def _sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (list, tuple)
    ):
        raise InvalidTargetSpecification(
            f"{field_name} must be a sequence"
        )
    return tuple(value)


def _curve_aware_twice_area(
    panel_name: str,
    coordinates: list[tuple[int | float, int | float]],
    edges: tuple[Any, ...],
) -> float:
    """Approximate signed area along serialized GarmentCode curves."""

    # Keep the construction boundary light for straight panels.
    from editing.pattern_geometry import edge_as_curve

    twice_area = 0.0
    for edge_index, edge in enumerate(edges):
        try:
            curve = edge_as_curve(coordinates, edge)
            previous = curve.point(0.0)
            if not (
                math.isfinite(previous.real)
                and math.isfinite(previous.imag)
            ):
                raise ValueError("curve sample is not finite")
            for sample_index in range(1, _CURVE_AREA_SAMPLE_COUNT + 1):
                current = curve.point(
                    sample_index / _CURVE_AREA_SAMPLE_COUNT
                )
                if not (
                    math.isfinite(current.real)
                    and math.isfinite(current.imag)
                ):
                    raise ValueError("curve sample is not finite")
                twice_area += (
                    previous.real * current.imag
                    - current.real * previous.imag
                )
                previous = current
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            NotImplementedError,
        ) as error:
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} edge {edge_index} has invalid "
                "serialized curve geometry"
            ) from error
    if not math.isfinite(twice_area):
        raise InvalidTargetSpecification("curve-aware panel area is not finite")
    return twice_area


def _validated_panel_edge_count(
    panel_name: str,
    panel: Mapping[str, Any],
) -> int:
    """Validate one panel's finite, nondegenerate directed 2D boundary."""

    vertices = _sequence(
        panel.get("vertices"),
        f"panel {panel_name!r}.vertices",
    )
    if len(vertices) < 3:
        raise InvalidTargetSpecification(
            f"panel {panel_name!r} must have at least three vertices"
        )

    coordinates: list[tuple[int | float, int | float]] = []
    for vertex_index, raw_vertex in enumerate(vertices):
        vertex = _sequence(
            raw_vertex,
            f"panel {panel_name!r}.vertex {vertex_index}",
        )
        if len(vertex) != 2 or any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(coordinate)
            for coordinate in vertex
        ):
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} vertex {vertex_index} must contain "
                "exactly two finite numbers"
            )
        coordinates.append((vertex[0], vertex[1]))

    edges = _sequence(
        panel.get("edges"),
        f"panel {panel_name!r}.edges",
    )
    if not edges:
        raise InvalidTargetSpecification(
            f"panel {panel_name!r} must have at least one edge"
        )

    outgoing: dict[int, tuple[int, int]] = {}
    incoming: dict[int, int] = {}
    endpoints_by_edge: list[tuple[int, int]] = []
    length_tolerance_squared = _EDGE_LENGTH_TOLERANCE**2
    for edge_index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} edge {edge_index} must be a mapping"
            )
        endpoints = _sequence(
            edge.get("endpoints"),
            f"panel {panel_name!r} edge {edge_index}.endpoints",
        )
        if len(endpoints) != 2:
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} edge {edge_index} must have exactly "
                "two endpoints"
            )
        start, end = endpoints
        for endpoint_index, vertex_index in enumerate(endpoints):
            if (
                type(vertex_index) is not int
                or vertex_index < 0
                or vertex_index >= len(coordinates)
            ):
                raise InvalidTargetSpecification(
                    f"panel {panel_name!r} edge {edge_index} endpoint "
                    f"{endpoint_index} must be a valid integer vertex index"
                )

        start_vertex = coordinates[start]
        end_vertex = coordinates[end]
        delta_x = end_vertex[0] - start_vertex[0]
        delta_y = end_vertex[1] - start_vertex[1]
        if (
            delta_x * delta_x + delta_y * delta_y
            <= length_tolerance_squared
        ):
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} edge {edge_index} is zero-length "
                f"within tolerance {_EDGE_LENGTH_TOLERANCE:g}"
            )
        if start in outgoing or end in incoming:
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} edges must form one closed directed "
                "boundary loop"
            )
        outgoing[start] = (end, edge_index)
        incoming[end] = edge_index
        endpoints_by_edge.append((start, end))

    first_start = endpoints_by_edge[0][0]
    current = first_start
    used_edges: set[int] = set()
    for _ in edges:
        connection = outgoing.get(current)
        if connection is None:
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} edges must form one closed directed "
                "boundary loop"
            )
        current, edge_index = connection
        if edge_index in used_edges:
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} edges must form one closed directed "
                "boundary loop using all edges"
            )
        used_edges.add(edge_index)
    if current != first_start or len(used_edges) != len(edges):
        raise InvalidTargetSpecification(
            f"panel {panel_name!r} edges must form one closed directed "
            "boundary loop using all edges"
        )

    if any("curvature" in edge for edge in edges):
        twice_area = _curve_aware_twice_area(
            panel_name,
            coordinates,
            edges,
        )
    else:
        twice_area = sum(
            coordinates[start][0] * coordinates[end][1]
            - coordinates[end][0] * coordinates[start][1]
            for start, end in endpoints_by_edge
        )
    if abs(twice_area) <= _POLYGON_AREA_TOLERANCE:
        raise InvalidTargetSpecification(
            f"panel {panel_name!r} boundary must have nonzero polygon area"
        )
    return len(edges)


def _validated_specification(value: Any) -> dict[str, Any]:
    """Detach a specification and validate its simulation-critical graph."""

    if not isinstance(value, Mapping):
        raise InvalidTargetSpecification("specification must be a mapping")
    try:
        canonical_json_bytes(value)
    except CanonicalDataError as error:
        raise InvalidTargetSpecification(
            f"specification is not finite canonical data: {error}"
        ) from error
    specification = plain_copy(value, tuples_as_lists=True)

    pattern = specification.get("pattern")
    if not isinstance(pattern, Mapping):
        raise InvalidTargetSpecification(
            "specification.pattern must be a mapping"
        )
    panels = pattern.get("panels")
    if not isinstance(panels, Mapping) or not panels:
        raise InvalidTargetSpecification(
            "specification.pattern.panels must be a non-empty mapping"
        )
    stitches = _sequence(
        pattern.get("stitches"),
        "specification.pattern.stitches",
    )

    edge_counts: dict[str, int] = {}
    for panel_name, panel in panels.items():
        if type(panel_name) is not str or not panel_name:
            raise InvalidTargetSpecification(
                "panel names must be non-empty strings"
            )
        if not isinstance(panel, Mapping):
            raise InvalidTargetSpecification(
                f"panel {panel_name!r} must be a mapping"
            )
        edge_counts[panel_name] = _validated_panel_edge_count(
            panel_name,
            panel,
        )

    if "panel_order" in pattern:
        panel_order = _sequence(
            pattern["panel_order"],
            "specification.pattern.panel_order",
        )
        if any(
            type(panel_name) is not str or not panel_name
            for panel_name in panel_order
        ):
            raise InvalidTargetSpecification(
                "specification.pattern.panel_order must contain non-empty "
                "panel names"
            )
        if len(set(panel_order)) != len(panel_order) or set(panel_order) != set(
            panels
        ):
            raise InvalidTargetSpecification(
                "specification.pattern.panel_order must include every panel "
                "exactly once"
            )

    for stitch_index, raw_stitch in enumerate(stitches):
        stitch = _sequence(raw_stitch, f"stitch {stitch_index}")
        if len(stitch) not in {2, 3}:
            raise InvalidStitchReference(
                f"stitch {stitch_index} must have two endpoints and "
                "an optional marker"
            )
        if len(stitch) == 3 and stitch[2] != "right_wrong":
            raise InvalidStitchReference(
                f"stitch {stitch_index} has an unsupported marker"
            )
        for endpoint_index, endpoint in enumerate(stitch[:2]):
            if not isinstance(endpoint, Mapping) or set(endpoint) != {
                "panel",
                "edge",
            }:
                raise InvalidStitchReference(
                    f"stitch {stitch_index} endpoint {endpoint_index} must "
                    "contain exactly panel and edge"
                )
            panel_name = endpoint["panel"]
            edge_index = endpoint["edge"]
            if (
                type(panel_name) is not str
                or panel_name not in edge_counts
            ):
                raise InvalidStitchReference(
                    f"stitch {stitch_index} endpoint {endpoint_index} "
                    "references a missing panel"
                )
            if (
                type(edge_index) is not int
                or edge_index < 0
                or edge_index >= edge_counts[panel_name]
            ):
                raise InvalidStitchReference(
                    f"stitch {stitch_index} endpoint {endpoint_index} "
                    "references a missing edge"
                )
    return specification


def _extract_specification(assembled: Any) -> Any:
    panel_order = getattr(assembled, "panel_order", None)
    if callable(panel_order):
        panel_order()
    if isinstance(assembled, Mapping):
        return assembled
    return getattr(assembled, "spec", None)


def preserve_unchanged_panels(
    source: Mapping[str, Any],
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[ConstructionResult, tuple[str, ...]]:
    """Restore original panels unaffected by the edit, including their pose.

    Baseline and target must be constructed from the same saved body/design,
    differing only by the requested edit. Comparing these two isolates actual
    edit dependencies from source YAML rounding. Never infer scope from names.
    """
    original_panels = source["pattern"]["panels"]
    baseline_panels = baseline["pattern"]["panels"]
    if set(original_panels) != set(baseline_panels):
        raise InvalidTargetSpecification("source reconstruction changed panel membership")
    if source["properties"]["units_in_meter"] != baseline["properties"]["units_in_meter"]:
        raise InvalidTargetSpecification("source reconstruction changed geometry units")

    def single_panel(panel):
        return {
            "pattern": {"panels": {"panel": panel}, "stitches": [], "panel_order": ["panel"]},
            "parameters": {}, "parameter_order": [], "properties": {},
        }

    specification = plain_copy(target, tuples_as_lists=True)
    preserved = []
    for name, panel in specification["pattern"]["panels"].items():
        if name not in baseline_panels:
            continue  # An added panel belongs to the edit.
        comparison = compare_patterns(
            single_panel(baseline_panels[name]), single_panel(panel),
            atol=PANEL_COMPARISON_ATOL,
        )
        if not comparison.equal:
            continue  # Keep real shape, topology, or placement dependencies.
        original = original_panels[name]
        # Target stitches reference edge indices. Replacing a panel whose
        # original edge layout differs could silently connect the wrong edges.
        if (
            len(original["vertices"]) != len(panel["vertices"])
            or edge_topology(original) != edge_topology(panel)
        ):
            raise InvalidTargetSpecification(
                f"{name}: cannot preserve source with different edge topology"
            )
        specification["pattern"]["panels"][name] = plain_copy(original, tuples_as_lists=True)
        preserved.append(name)

    # Validate the final geometry and stitch references before hashing/rendering.
    specification = _validated_specification(specification)
    payload = canonical_json_bytes(specification)
    return ConstructionResult(
        success=True, specification=specification,
        specification_json_bytes=payload, specification_sha256=sha256_bytes(payload),
        failure_code=None, failure_message=None,
    ), tuple(sorted(preserved))


def build_target_pattern(
    design: Mapping[str, Any],
    body_yaml_bytes: bytes,
    name: str,
    *,
    dependencies: ConstructionDependencies | None = None,
) -> ConstructionResult:
    """Construct one target from detached canonical design and body copies."""

    if not isinstance(design, Mapping):
        raise TypeError("design must be a mapping")
    if type(body_yaml_bytes) is not bytes or not body_yaml_bytes:
        raise TypeError("body_yaml_bytes must be non-empty immutable bytes")
    if type(name) is not str or not name or name != name.strip():
        raise ValueError("name must be a non-empty trimmed string")
    if dependencies is None:
        dependencies = _default_dependencies()
    elif not isinstance(dependencies, ConstructionDependencies):
        raise TypeError(
            "dependencies must be ConstructionDependencies or None"
        )

    try:
        # GarmentCode receives mutable data, but never caller-owned mappings.
        canonical_json_bytes(design)
        working_design = plain_copy(design, tuples_as_lists=True)
        working_body = plain_copy(
            load_body_yaml_bytes(body_yaml_bytes),
            tuples_as_lists=True,
        )
    except (CanonicalDataError, UnicodeError) as error:
        return _failure("invalid_input", error)

    try:
        dependencies.validate_design(working_design)
        body_parameters = instantiate_body_parameters(
            dependencies.body_parameters_cls
        )
        body_parameters.load_from_dict(working_body)
        garment = dependencies.meta_garment_cls(
            name,
            body_parameters,
            working_design,
        )
        garment.assert_non_empty()
        garment.assert_total_length()
        if garment.is_self_intersecting():
            return _failure(
                "self_intersection",
                "constructed target contains a self-intersecting panel",
            )
        assembled = garment.assembly()
        raw_specification = _extract_specification(assembled)
        try:
            specification = _validated_specification(raw_specification)
        except InvalidTargetSpecification as error:
            return _failure("invalid_specification", error)
    except dependencies.domain_errors as error:
        return _failure("construction_failed", error)

    specification_bytes = canonical_json_bytes(specification)
    return ConstructionResult(
        success=True,
        specification=specification,
        specification_json_bytes=specification_bytes,
        specification_sha256=sha256_bytes(specification_bytes),
        failure_code=None,
        failure_message=None,
    )
