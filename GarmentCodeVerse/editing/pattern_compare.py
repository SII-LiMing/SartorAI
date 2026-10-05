"""Deterministic semantic comparison for GarmentCode pattern specifications.

The serialized JSON is not a stable equality representation: panel dictionary
order, stitch order, stitch endpoint order, and panel-order traversal can vary
without changing a garment.  This module compares the fields that carry
pattern semantics directly and returns the first deterministic mismatch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from .canonical_io import canonical_hash, canonical_json_bytes


DEFAULT_ATOL = 1e-5
PATTERN_SEMANTIC_VERSION = "pattern-semantic-v1"
_IGNORED_TOP_LEVEL_FIELDS = frozenset(
    {"name", "sample_name", "path", "source_path", "spec_file"}
)
_REQUIRED_TOP_LEVEL_FIELDS = (
    "pattern",
    "parameters",
    "parameter_order",
    "properties",
)
_TOP_LEVEL_EXTRA_EXCLUSIONS = (
    _IGNORED_TOP_LEVEL_FIELDS | frozenset(_REQUIRED_TOP_LEVEL_FIELDS)
)
_REQUIRED_PANEL_FIELDS = ("translation", "rotation", "vertices", "edges")
_PATTERN_FIELDS = ("panels", "stitches", "panel_order")
_EDGE_FIELDS = ("endpoints", "curvature")
_SUPPORTED_EDGE_TYPES = frozenset({"line", "quadratic", "circle", "cubic"})


class PatternFormatError(ValueError):
    """A pattern specification cannot be compared safely."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.detail = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class PatternMismatch:
    """The first deterministic semantic mismatch between two patterns."""

    code: str
    path: str
    expected: Any
    actual: Any
    message: str


@dataclass(frozen=True)
class PatternComparison:
    """Result of one tolerant semantic pattern comparison."""

    equal: bool
    mismatch: PatternMismatch | None = None

    def __post_init__(self) -> None:
        if self.equal == (self.mismatch is not None):
            raise ValueError("equal results cannot have a mismatch, and vice versa")

def _mismatch(
    code: str,
    path: str,
    expected: Any,
    actual: Any,
    message: str,
) -> PatternMismatch:
    return PatternMismatch(
        code=code,
        path=path,
        expected=deepcopy(expected),
        actual=deepcopy(actual),
        message=message,
    )


def _extract_specification(value: Any, side: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PatternFormatError(side, "expected a specification mapping")
    return value


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PatternFormatError(path, "expected a mapping")
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if not _is_sequence(value):
        raise PatternFormatError(path, "expected a sequence")
    return value


def _require_fields(
    mapping: Mapping[str, Any], fields: Sequence[str], path: str
) -> None:
    for field in fields:
        if field not in mapping:
            raise PatternFormatError(f"{path}.{field}", "required field is missing")


def _extra_fields(
    mapping: Mapping[str, Any],
    excluded: Collection[str],
) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in excluded}


def _validate_parameter_membership(spec: Mapping[str, Any], side: str) -> None:
    parameters = _require_mapping(spec["parameters"], f"{side}.parameters")
    if any(not isinstance(name, str) or not name for name in parameters):
        raise PatternFormatError(
            f"{side}.parameters", "parameter names must be non-empty strings"
        )
    order = _require_sequence(spec["parameter_order"], f"{side}.parameter_order")
    if any(not isinstance(name, str) or not name for name in order):
        raise PatternFormatError(
            f"{side}.parameter_order", "members must be non-empty strings"
        )
    if len(set(order)) != len(order):
        raise PatternFormatError(
            f"{side}.parameter_order", "duplicate parameter names are not allowed"
        )
    if set(order) != set(parameters):
        raise PatternFormatError(
            f"{side}.parameter_order",
            "membership must exactly match the parameters mapping",
        )


def _panel_data(
    spec: Mapping[str, Any], side: str
) -> tuple[Mapping[str, Any], Sequence[Any], Sequence[Any]]:
    _require_fields(spec, _REQUIRED_TOP_LEVEL_FIELDS, side)
    pattern = _require_mapping(spec["pattern"], f"{side}.pattern")
    _require_fields(pattern, _PATTERN_FIELDS, f"{side}.pattern")
    panels = _require_mapping(pattern["panels"], f"{side}.pattern.panels")
    if any(not isinstance(name, str) or not name for name in panels):
        raise PatternFormatError(
            f"{side}.pattern.panels", "panel names must be non-empty strings"
        )
    stitches = _require_sequence(
        pattern["stitches"], f"{side}.pattern.stitches"
    )
    panel_order = _require_sequence(
        pattern["panel_order"], f"{side}.pattern.panel_order"
    )
    if any(not isinstance(name, str) or not name for name in panel_order):
        raise PatternFormatError(
            f"{side}.pattern.panel_order",
            "members must be non-empty panel names",
        )
    if len(set(panel_order)) != len(panel_order) or set(panel_order) != set(panels):
        raise PatternFormatError(
            f"{side}.pattern.panel_order",
            "membership must include every panel exactly once",
        )
    _validate_parameter_membership(spec, side)
    _require_mapping(spec["properties"], f"{side}.properties")
    return panels, stitches, panel_order


def _freeze_exact(value: Any) -> Any:
    """Create a type-preserving, hashable value with sorted mapping keys."""

    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (key, _freeze_exact(value[key]))
                for key in sorted(value, key=lambda item: (type(item).__name__, repr(item)))
            ),
        )
    if _is_sequence(value):
        return ("sequence", tuple(_freeze_exact(item) for item in value))
    return (type(value).__name__, value)


def _exact_mismatch(left: Any, right: Any, path: str) -> PatternMismatch | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            missing = sorted(left_keys - right_keys, key=str)
            extra = sorted(right_keys - left_keys, key=str)
            field = missing[0] if missing else extra[0]
            return _mismatch(
                "field_presence_mismatch",
                f"{path}.{field}",
                left.get(field),
                right.get(field),
                "mapping field presence differs",
            )
        for key in sorted(left_keys, key=lambda item: (type(item).__name__, repr(item))):
            mismatch = _exact_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None

    if _is_sequence(left) and _is_sequence(right):
        if len(left) != len(right):
            return _mismatch(
                "sequence_length_mismatch",
                path,
                len(left),
                len(right),
                "sequence lengths differ",
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            mismatch = _exact_mismatch(
                left_item, right_item, f"{path}[{index}]"
            )
            if mismatch:
                return mismatch
        return None

    if type(left) is not type(right) or left != right:
        return _mismatch(
            "exact_value_mismatch",
            path,
            left,
            right,
            "values differ under exact comparison",
        )
    return None


def _numeric_mismatch(
    left: Any, right: Any, path: str, atol: float
) -> PatternMismatch | None:
    if _is_sequence(left) and _is_sequence(right):
        if len(left) != len(right):
            return _mismatch(
                "sequence_length_mismatch",
                path,
                len(left),
                len(right),
                "numeric sequence lengths differ",
            )
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            mismatch = _numeric_mismatch(
                left_item, right_item, f"{path}[{index}]", atol
            )
            if mismatch:
                return mismatch
        return None

    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, Real)
        or not isinstance(right, Real)
    ):
        return _mismatch(
            "numeric_type_mismatch",
            path,
            left,
            right,
            "expected finite numeric values",
        )
    left_float = float(left)
    right_float = float(right)
    if (
        not math.isfinite(left_float)
        or not math.isfinite(right_float)
        or not math.isclose(left_float, right_float, rel_tol=0.0, abs_tol=atol)
    ):
        return _mismatch(
            "numeric_mismatch",
            path,
            left,
            right,
            f"numeric values differ by more than atol={atol}",
        )
    return None


def _validate_finite_vector(value: Any, path: str, length: int) -> None:
    vector = _require_sequence(value, path)
    if len(vector) != length:
        raise PatternFormatError(path, f"expected exactly {length} numeric values")
    for index, item in enumerate(vector):
        if (
            isinstance(item, bool)
            or not isinstance(item, Real)
            or not math.isfinite(float(item))
        ):
            raise PatternFormatError(
                f"{path}[{index}]", "expected a finite numeric value"
            )


def edge_type(edge: Mapping[str, Any]) -> str:
    """Return the semantic curve type of one serialized edge."""

    if "curvature" not in edge:
        return "line"
    curvature = edge["curvature"]
    if isinstance(curvature, (list, tuple)):
        return "quadratic"
    if not isinstance(curvature, Mapping) or not isinstance(
        curvature.get("type"), str
    ):
        raise PatternFormatError("edge.curvature", "missing string curve type")
    curve_type = curvature["type"]
    if curve_type not in _SUPPORTED_EDGE_TYPES - {"line"}:
        raise PatternFormatError(
            "edge.curvature.type", f"unsupported curve type {curve_type!r}"
        )
    return curve_type


def _curvature_parts(
    edge: Mapping[str, Any], path: str
) -> tuple[str, Any, Mapping[str, Any]]:
    curve_type = edge_type(edge)
    if curve_type == "line":
        return curve_type, (), {}
    curvature = edge["curvature"]
    if isinstance(curvature, (list, tuple)):
        if len(curvature) != 2:
            raise PatternFormatError(path, "legacy quadratic curvature needs two values")
        _validate_finite_vector(curvature, path, 2)
        return curve_type, [curvature], {}
    if "params" not in curvature:
        raise PatternFormatError(f"{path}.params", "required field is missing")
    params = curvature["params"]
    extra = _extra_fields(curvature, ("type", "params"))
    if curve_type == "quadratic":
        if not _is_sequence(params) or len(params) != 1:
            raise PatternFormatError(path, "quadratic curvature needs one control point")
        _validate_finite_vector(params[0], f"{path}.params[0]", 2)
    elif curve_type == "cubic":
        if not _is_sequence(params) or len(params) != 2:
            raise PatternFormatError(path, "cubic curvature needs two control points")
        for index, point in enumerate(params):
            _validate_finite_vector(point, f"{path}.params[{index}]", 2)
    elif curve_type == "circle":
        if not _is_sequence(params) or len(params) != 3:
            raise PatternFormatError(path, "circle curvature needs radius and two flags")
        radius, large_arc, right = params
        if (
            isinstance(radius, bool)
            or not isinstance(radius, Real)
            or not math.isfinite(float(radius))
            or radius <= 0
        ):
            raise PatternFormatError(
                f"{path}.params[0]", "circle radius must be finite and positive"
            )
        def is_binary_flag(value: Any) -> bool:
            return isinstance(value, bool) or (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value in {0, 1}
            )

        if not is_binary_flag(large_arc) or not is_binary_flag(right):
            raise PatternFormatError(
                f"{path}.params[1:]", "circle flags must be booleans or 0/1"
            )
    return curve_type, params, extra


def _compare_curvature(
    left_edge: Mapping[str, Any],
    right_edge: Mapping[str, Any],
    path: str,
    atol: float,
) -> PatternMismatch | None:
    left_type, left_params, left_extra = _curvature_parts(
        left_edge, f"{path}.curvature"
    )
    right_type, right_params, right_extra = _curvature_parts(
        right_edge, f"{path}.curvature"
    )
    if left_type != right_type:
        return _mismatch(
            "edge_type_mismatch",
            f"{path}.type",
            left_type,
            right_type,
            "edge curve types differ",
        )
    mismatch = _exact_mismatch(
        left_extra, right_extra, f"{path}.curvature.metadata"
    )
    if mismatch:
        return mismatch
    if left_type == "circle":
        mismatch = _numeric_mismatch(
            left_params[0], right_params[0], f"{path}.curvature.params[0]", atol
        )
        if mismatch:
            return mismatch
        return _exact_mismatch(
            list(left_params[1:]),
            list(right_params[1:]),
            f"{path}.curvature.params[1:]",
        )
    return _numeric_mismatch(
        left_params, right_params, f"{path}.curvature.params", atol
    )


def _validate_endpoints(
    edge: Mapping[str, Any], vertex_count: int, path: str
) -> tuple[int, int]:
    if "endpoints" not in edge:
        raise PatternFormatError(f"{path}.endpoints", "required field is missing")
    endpoints = _require_sequence(edge["endpoints"], f"{path}.endpoints")
    if len(endpoints) != 2:
        raise PatternFormatError(f"{path}.endpoints", "exactly two indices required")
    if any(
        isinstance(index, bool) or not isinstance(index, int) for index in endpoints
    ):
        raise PatternFormatError(f"{path}.endpoints", "indices must be integers")
    if any(index < 0 or index >= vertex_count for index in endpoints):
        raise PatternFormatError(f"{path}.endpoints", "vertex index is out of range")
    return int(endpoints[0]), int(endpoints[1])


def _canonical_stitch(
    stitch: Any,
    panels: Mapping[str, Any],
    path: str,
) -> tuple[Any, ...]:
    stitch_sequence = _require_sequence(stitch, path)
    if len(stitch_sequence) not in {2, 3}:
        raise PatternFormatError(
            path, "a stitch requires two endpoints and one optional marker"
        )
    if len(stitch_sequence) == 3 and stitch_sequence[2] != "right_wrong":
        raise PatternFormatError(
            f"{path}[2]", "the only supported stitch marker is 'right_wrong'"
        )
    endpoints: list[tuple[str, int]] = []
    for side_index in (0, 1):
        endpoint_path = f"{path}[{side_index}]"
        endpoint = _require_mapping(stitch_sequence[side_index], endpoint_path)
        if set(endpoint) != {"panel", "edge"}:
            raise PatternFormatError(
                endpoint_path, "endpoint must contain exactly panel and edge"
            )
        panel_name = endpoint["panel"]
        edge_index = endpoint["edge"]
        if not isinstance(panel_name, str) or panel_name not in panels:
            raise PatternFormatError(endpoint_path, "panel reference does not exist")
        panel = _require_mapping(panels[panel_name], f"pattern.panels.{panel_name}")
        edges = _require_sequence(
            panel.get("edges"), f"pattern.panels.{panel_name}.edges"
        )
        if (
            isinstance(edge_index, bool)
            or not isinstance(edge_index, int)
            or edge_index < 0
            or edge_index >= len(edges)
        ):
            raise PatternFormatError(endpoint_path, "edge reference is out of range")
        endpoints.append((panel_name, edge_index))
    endpoints.sort()
    markers = tuple(_freeze_exact(marker) for marker in stitch_sequence[2:])
    return endpoints[0], endpoints[1], markers


def _stitch_counter(
    panels: Mapping[str, Any],
    stitches: Sequence[Any],
    path: str,
) -> Counter[tuple[Any, ...]]:
    return Counter(
        _canonical_stitch(stitch, panels, f"{path}.stitches[{index}]")
        for index, stitch in enumerate(stitches)
    )


def _validate_specification(
    specification: Mapping[str, Any], side: str
) -> tuple[
    Mapping[str, Any],
    Sequence[Any],
    Counter[tuple[Any, ...]],
]:
    """Validate every index-bearing structure before pairwise comparison."""

    panels, stitches, _ = _panel_data(specification, side)
    for panel_name in sorted(panels):
        panel_path = f"{side}.pattern.panels.{panel_name}"
        panel = _require_mapping(panels[panel_name], panel_path)
        _require_fields(panel, _REQUIRED_PANEL_FIELDS, panel_path)
        vertices = _require_sequence(panel["vertices"], f"{panel_path}.vertices")
        edges = _require_sequence(panel["edges"], f"{panel_path}.edges")
        _validate_finite_vector(
            panel["translation"], f"{panel_path}.translation", 3
        )
        _validate_finite_vector(panel["rotation"], f"{panel_path}.rotation", 3)
        if not vertices:
            raise PatternFormatError(
                f"{panel_path}.vertices", "a panel must contain at least one vertex"
            )
        for vertex_index, vertex in enumerate(vertices):
            _validate_finite_vector(
                vertex, f"{panel_path}.vertices[{vertex_index}]", 2
            )
        for edge_index, edge_value in enumerate(edges):
            edge_path = f"{panel_path}.edges[{edge_index}]"
            edge = _require_mapping(edge_value, edge_path)
            endpoints = _validate_endpoints(edge, len(vertices), edge_path)
            if endpoints[0] == endpoints[1]:
                raise PatternFormatError(
                    f"{edge_path}.endpoints",
                    "an edge cannot repeat the same vertex index",
                )
            _curvature_parts(edge, f"{edge_path}.curvature")

    stitch_counter = _stitch_counter(panels, stitches, f"{side}.pattern")
    return panels, stitches, stitch_counter


def _format_failure(side: str, error: PatternFormatError) -> PatternComparison:
    path = error.path
    if not path.startswith(f"{side}.") and path != side:
        path = f"{side}.{path}"
    code = (
        "invalid_panel_order"
        if path.endswith("pattern.panel_order")
        else "invalid_pattern"
    )
    return PatternComparison(
        equal=False,
        mismatch=_mismatch(
            code,
            path,
            "valid pattern specification",
            error.detail,
            f"{side} pattern is malformed: {error.detail}",
        ),
    )


def compare_patterns(
    expected: Any,
    actual: Any,
    *,
    atol: float = DEFAULT_ATOL,
) -> PatternComparison:
    """Compare two pattern specifications and return the first mismatch.

    Numeric geometry uses ``rtol=0`` and the supplied absolute tolerance.
    Design parameters and properties remain exact.  Dictionary order and the
    ordering of valid ``panel_order`` members do not affect equality.
    """

    if (
        isinstance(atol, bool)
        or not isinstance(atol, Real)
        or not math.isfinite(float(atol))
        or atol < 0
    ):
        raise ValueError("atol must be a finite non-negative number")
    atol = float(atol)

    try:
        left = _extract_specification(expected, "left")
        left_panels, _, left_counter = _validate_specification(
            left, "left"
        )
    except PatternFormatError as error:
        return _format_failure("left", error)
    try:
        right = _extract_specification(actual, "right")
        right_panels, _, right_counter = _validate_specification(
            right, "right"
        )
    except PatternFormatError as error:
        return _format_failure("right", error)

    left_names = set(left_panels)
    right_names = set(right_panels)
    if left_names != right_names:
        return PatternComparison(
            False,
            _mismatch(
                "panel_names_mismatch",
                "pattern.panels",
                sorted(left_names),
                sorted(right_names),
                "panel-name sets differ",
            ),
        )

    for field in ("parameters", "parameter_order", "properties"):
        mismatch = _exact_mismatch(left[field], right[field], field)
        if mismatch:
            return PatternComparison(False, mismatch)

    left_top_extra = _extra_fields(left, _TOP_LEVEL_EXTRA_EXCLUSIONS)
    right_top_extra = _extra_fields(right, _TOP_LEVEL_EXTRA_EXCLUSIONS)
    mismatch = _exact_mismatch(
        left_top_extra, right_top_extra, "top_level_metadata"
    )
    if mismatch:
        return PatternComparison(False, mismatch)

    left_pattern = left["pattern"]
    right_pattern = right["pattern"]
    left_pattern_extra = _extra_fields(left_pattern, _PATTERN_FIELDS)
    right_pattern_extra = _extra_fields(right_pattern, _PATTERN_FIELDS)
    mismatch = _exact_mismatch(
        left_pattern_extra, right_pattern_extra, "pattern.metadata"
    )
    if mismatch:
        return PatternComparison(False, mismatch)

    for panel_name in sorted(left_names):
        panel_path = f"pattern.panels.{panel_name}"
        left_panel = left_panels[panel_name]
        right_panel = right_panels[panel_name]
        left_edges = left_panel["edges"]
        right_edges = right_panel["edges"]

        for field in ("translation", "rotation", "vertices"):
            mismatch = _numeric_mismatch(
                left_panel[field],
                right_panel[field],
                f"{panel_path}.{field}",
                atol,
            )
            if mismatch:
                return PatternComparison(False, mismatch)

        left_panel_extra = _extra_fields(left_panel, _REQUIRED_PANEL_FIELDS)
        right_panel_extra = _extra_fields(right_panel, _REQUIRED_PANEL_FIELDS)
        mismatch = _exact_mismatch(
            left_panel_extra, right_panel_extra, panel_path
        )
        if mismatch:
            return PatternComparison(False, mismatch)

        if len(left_edges) != len(right_edges):
            return PatternComparison(
                False,
                _mismatch(
                    "edge_count_mismatch",
                    f"{panel_path}.edges",
                    len(left_edges),
                    len(right_edges),
                    "panel edge counts differ",
                ),
            )
        for edge_index, (left_edge_value, right_edge_value) in enumerate(
            zip(left_edges, right_edges)
        ):
            edge_path = f"{panel_path}.edges[{edge_index}]"
            left_edge = left_edge_value
            right_edge = right_edge_value
            left_endpoints = tuple(left_edge["endpoints"])
            right_endpoints = tuple(right_edge["endpoints"])
            if left_endpoints != right_endpoints:
                return PatternComparison(
                    False,
                    _mismatch(
                        "edge_endpoints_mismatch",
                        f"{edge_path}.endpoints",
                        left_endpoints,
                        right_endpoints,
                        "edge endpoint indices differ",
                    ),
                )
            mismatch = _compare_curvature(
                left_edge, right_edge, edge_path, atol
            )
            if mismatch:
                return PatternComparison(False, mismatch)
            left_edge_extra = _extra_fields(left_edge, _EDGE_FIELDS)
            right_edge_extra = _extra_fields(right_edge, _EDGE_FIELDS)
            mismatch = _exact_mismatch(
                left_edge_extra, right_edge_extra, edge_path
            )
            if mismatch:
                return PatternComparison(False, mismatch)

    if left_counter != right_counter:
        return PatternComparison(
            False,
            _mismatch(
                "stitch_multiset_mismatch",
                "pattern.stitches",
                sorted(left_counter.items(), key=repr),
                sorted(right_counter.items(), key=repr),
                "canonical stitch multisets differ",
            ),
        )
    return PatternComparison(True)


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonical_json_value(value[key])
            for key in sorted(value)
        }
    if _is_sequence(value):
        return [_canonical_json_value(item) for item in value]
    return deepcopy(value)


def _canonical_edge_payload(edge: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = {
        key: _canonical_json_value(value)
        for key, value in sorted(edge.items())
        if key != "curvature"
    }
    if "curvature" in edge:
        curvature = edge["curvature"]
        if isinstance(curvature, (list, tuple)):
            payload["curvature"] = {
                "type": "quadratic",
                "params": [_canonical_json_value(curvature)],
            }
        else:
            payload["curvature"] = _canonical_json_value(curvature)
    return payload


def canonical_pattern_payload(specification: Any) -> Mapping[str, Any]:
    """Normalize semantically unordered fields for provenance hashing.

    Numeric values are intentionally *not* tolerance-quantized.  Consequently,
    two tolerantly equal patterns can have different hashes; callers must still
    use :func:`compare_patterns` for acceptance decisions.
    """

    spec = _extract_specification(specification, "pattern")
    panels, stitches, _ = _validate_specification(spec, "pattern")

    canonical_panels = {}
    for panel_name in sorted(panels):
        panel = panels[panel_name]
        canonical_panel = {
            key: _canonical_json_value(value)
            for key, value in sorted(panel.items())
            if key != "edges"
        }
        canonical_panel["edges"] = [
            _canonical_edge_payload(edge) for edge in panel["edges"]
        ]
        canonical_panels[panel_name] = canonical_panel

    canonical_stitches = []
    for stitch in stitches:
        endpoints = sorted(
            (
                {"panel": stitch[0]["panel"], "edge": stitch[0]["edge"]},
                {"panel": stitch[1]["panel"], "edge": stitch[1]["edge"]},
            ),
            key=lambda endpoint: (endpoint["panel"], endpoint["edge"]),
        )
        canonical_stitches.append(
            endpoints
            + [_canonical_json_value(marker) for marker in stitch[2:]]
        )
    canonical_stitches.sort(key=canonical_json_bytes)

    pattern = spec["pattern"]
    canonical_pattern = {
        key: _canonical_json_value(value)
        for key, value in sorted(pattern.items())
        if key not in {"panels", "stitches", "panel_order"}
    }
    canonical_pattern.update(
        {
            "panels": canonical_panels,
            "stitches": canonical_stitches,
            "panel_order": sorted(pattern["panel_order"]),
        }
    )

    payload = {
        key: _canonical_json_value(value)
        for key, value in sorted(spec.items())
        if key not in _IGNORED_TOP_LEVEL_FIELDS
        and key not in {"pattern", "parameters", "parameter_order", "properties"}
    }
    payload.update(
        {
            "pattern": canonical_pattern,
            "parameters": _canonical_json_value(spec["parameters"]),
            "parameter_order": _canonical_json_value(spec["parameter_order"]),
            "properties": _canonical_json_value(spec["properties"]),
        }
    )
    return payload


def semantic_pattern_hash(specification: Any) -> str:
    """Hash the exact normalized semantic payload for provenance."""

    return canonical_hash(
        {
            "pattern_semantic_version": PATTERN_SEMANTIC_VERSION,
            "pattern": canonical_pattern_payload(specification),
        }
    )


def topology_signature(specification: Any) -> Mapping[str, Any]:
    """Return a geometry-free signature of panel, edge, and stitch topology."""

    spec = _extract_specification(specification, "pattern")
    panels, stitches, panel_order = _panel_data(spec, "pattern")
    panel_records = []
    for panel_name in sorted(panels):
        panel = _require_mapping(panels[panel_name], f"pattern.panels.{panel_name}")
        _require_fields(panel, _REQUIRED_PANEL_FIELDS, f"pattern.panels.{panel_name}")
        vertices = _require_sequence(
            panel["vertices"], f"pattern.panels.{panel_name}.vertices"
        )
        edges = _require_sequence(
            panel["edges"], f"pattern.panels.{panel_name}.edges"
        )
        edge_records = []
        for index, edge_value in enumerate(edges):
            path = f"pattern.panels.{panel_name}.edges[{index}]"
            edge = _require_mapping(edge_value, path)
            endpoints = _validate_endpoints(edge, len(vertices), path)
            curve_type, _, curvature_extra = _curvature_parts(
                edge, f"{path}.curvature"
            )
            edge_metadata = _extra_fields(edge, _EDGE_FIELDS)
            edge_records.append(
                {
                    "index": index,
                    "endpoints": endpoints,
                    "type": curve_type,
                    "metadata": _freeze_exact(edge_metadata),
                    "curvature_metadata": _freeze_exact(curvature_extra),
                }
            )
        panel_metadata = _extra_fields(panel, _REQUIRED_PANEL_FIELDS)
        panel_records.append(
            {
                "name": panel_name,
                "vertex_count": len(vertices),
                "metadata": _freeze_exact(panel_metadata),
                "edges": tuple(edge_records),
            }
        )

    stitch_counter = _stitch_counter(panels, stitches, "pattern")
    return {
        "panels": tuple(panel_records),
        "panel_order_members": tuple(sorted(panel_order)),
        "stitches": tuple(sorted(stitch_counter.items(), key=repr)),
    }
