"""Pattern-only geometric effect metrics.

This module intentionally measures panel boundary shape in local 2-D space.
Panel placement is not an edit effect, and anchoring every panel at its first
serialized vertex removes arbitrary local-coordinate translations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .canonical_io import canonical_json_bytes, plain_copy
from .pattern_compare import edge_type


DEFAULT_EDGE_SAMPLES = 101
_IGNORED_SPECIFICATION_FIELDS = frozenset(
    {"name", "sample_name", "path", "source_path", "spec_file", "pattern"}
)
_STRUCTURAL_PATTERN_FIELDS = frozenset(
    {"panels", "stitches", "panel_order"}
)


def edge_as_curve(
    vertices: Sequence[Sequence[float]],
    edge: Mapping[str, Any],
):
    """Return the serialized edge as an ``svgpathtools`` curve."""

    import numpy as np
    import svgpathtools as svgpath
    from pygarment.pattern import utils

    vertex_array = np.asarray(vertices)
    start = vertex_array[edge["endpoints"][0]]
    end = vertex_array[edge["endpoints"][1]]

    if "curvature" not in edge:
        return svgpath.Line(*utils.list_to_c([start, end]))

    curvature = edge["curvature"]
    if isinstance(curvature, (list, tuple)) or curvature["type"] == "quadratic":
        relative_control = (
            curvature
            if isinstance(curvature, (list, tuple))
            else curvature["params"][0]
        )
        control_point = utils.rel_to_abs_2d(start, end, relative_control)
        return svgpath.QuadraticBezier(
            *utils.list_to_c([start, control_point, end])
        )

    if curvature["type"] == "circle":
        radius, large_arc, right = curvature["params"]
        return svgpath.Arc(
            utils.list_to_c(start),
            radius + 1j * radius,
            rotation=0,
            large_arc=large_arc,
            sweep=right,
            end=utils.list_to_c(end),
        )

    if curvature["type"] == "cubic":
        control_points = [
            utils.rel_to_abs_2d(start, end, point)
            for point in curvature["params"]
        ]
        return svgpath.CubicBezier(
            *utils.list_to_c([start, *control_points, end])
        )

    raise NotImplementedError(f'Unknown curvature type {curvature["type"]}')


class PatternGeometryError(ValueError):
    """The boundary displacement metric cannot be evaluated."""


class TopologyMismatchError(PatternGeometryError):
    """The two patterns cannot be aligned panel-by-panel and edge-by-edge."""


def edge_topology(panel: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Return geometry-free edge records for one serialized panel."""

    vertices = _sequence(panel.get("vertices"), "panel.vertices")
    records = []
    for index, raw_edge in enumerate(
        _sequence(panel.get("edges"), "panel.edges")
    ):
        if not isinstance(raw_edge, Mapping):
            raise PatternGeometryError(f"panel.edges[{index}] must be a mapping")
        endpoints = _sequence(
            raw_edge.get("endpoints"),
            f"panel.edges[{index}].endpoints",
        )
        if (
            len(endpoints) != 2
            or any(
                isinstance(endpoint, bool)
                or not isinstance(endpoint, int)
                or endpoint < 0
                or endpoint >= len(vertices)
                for endpoint in endpoints
            )
        ):
            raise PatternGeometryError(
                f"panel.edges[{index}].endpoints must contain two in-range indices"
            )
        label = raw_edge.get("label")
        if label is not None and (type(label) is not str or not label):
            raise PatternGeometryError("edge labels must be non-empty strings")
        try:
            kind = edge_type(raw_edge)
        except ValueError as error:
            raise PatternGeometryError(str(error)) from error
        records.append((kind, tuple(endpoints), label))
    return tuple(records)


def _specification(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PatternGeometryError("expected a specification mapping")
    return value


def specification_metadata(specification: Mapping[str, Any]) -> dict[str, Any]:
    """Return non-geometric specification fields used by rule validators."""

    source = _specification(specification)
    pattern = source.get("pattern")
    if not isinstance(pattern, Mapping):
        raise PatternGeometryError("pattern must be a mapping")
    return plain_copy(
        {
            "top": {
                key: value
                for key, value in source.items()
                if key not in _IGNORED_SPECIFICATION_FIELDS
            },
            "pattern": {
                key: value
                for key, value in pattern.items()
                if key not in _STRUCTURAL_PATTERN_FIELDS
            },
        }
    )


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise PatternGeometryError(f"{path} must be a sequence")
    return value


def _panels(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        panels = specification["pattern"]["panels"]
    except (KeyError, TypeError) as error:
        raise PatternGeometryError("pattern.panels is required") from error
    if not isinstance(panels, Mapping):
        raise PatternGeometryError("pattern.panels must be a mapping")
    return panels


def stitch_records(
    specification: Mapping[str, Any],
) -> tuple[tuple[tuple[str, int], tuple[str, int], tuple[bytes, ...]], ...]:
    """Return undirected, marker-preserving stitch topology records."""

    panels = _panels(specification)
    try:
        stitches = _sequence(
            specification["pattern"]["stitches"],
            "pattern.stitches",
        )
    except KeyError as error:
        raise PatternGeometryError("pattern.stitches is required") from error

    def endpoint(raw: Any) -> tuple[str, int]:
        if not isinstance(raw, Mapping):
            raise PatternGeometryError("stitch endpoint must be a mapping")
        if set(raw) != {"panel", "edge"}:
            raise PatternGeometryError(
                "stitch endpoint must contain exactly panel and edge"
            )
        panel_name = raw["panel"]
        edge_index = raw["edge"]
        if type(panel_name) is not str:
            raise PatternGeometryError("stitch panel must be text")
        panel = panels.get(panel_name)
        if not isinstance(panel, Mapping):
            raise PatternGeometryError(
                f"pattern.panels.{panel_name} must be a mapping"
            )
        edges = _sequence(
            panel.get("edges"),
            f"pattern.panels.{panel_name}.edges",
        )
        if (
            isinstance(edge_index, bool)
            or not isinstance(edge_index, int)
            or edge_index < 0
            or edge_index >= len(edges)
        ):
            raise PatternGeometryError(
                f"invalid edge reference {panel_name}[{edge_index!r}]"
            )
        if not isinstance(edges[edge_index], Mapping):
            raise PatternGeometryError(
                f"pattern.panels.{panel_name}.edges[{edge_index}] "
                "must be a mapping"
            )
        return panel_name, edge_index

    records = []
    for raw in stitches:
        stitch = _sequence(raw, "stitch")
        if len(stitch) not in {2, 3}:
            raise PatternGeometryError(
                "stitches require two endpoints and an optional marker"
            )
        endpoints = sorted((endpoint(stitch[0]), endpoint(stitch[1])))
        marker = tuple(canonical_json_bytes(item) for item in stitch[2:])
        records.append((endpoints[0], endpoints[1], marker))
    return tuple(records)


def _units_scale_cm(specification: Mapping[str, Any]) -> float:
    try:
        units = specification["properties"]["units_in_meter"]
    except (KeyError, TypeError) as error:
        raise PatternGeometryError(
            "properties.units_in_meter is required"
        ) from error
    if (
        isinstance(units, bool)
        or not isinstance(units, Real)
        or not math.isfinite(float(units))
        or units <= 0
    ):
        raise PatternGeometryError(
            "properties.units_in_meter must be a finite positive number"
        )
    return 100.0 / float(units)


def _aligned_panels(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    before_panels = _panels(before)
    after_panels = _panels(after)
    if set(before_panels) != set(after_panels):
        raise TopologyMismatchError("panel-name sets differ")
    if not before_panels:
        raise PatternGeometryError("pattern must contain at least one panel")

    aligned = []
    for panel_name in sorted(before_panels):
        before_panel = before_panels[panel_name]
        after_panel = after_panels[panel_name]
        if not isinstance(before_panel, Mapping) or not isinstance(
            after_panel, Mapping
        ):
            raise PatternGeometryError(
                f"pattern.panels.{panel_name} must be a mapping"
            )
        try:
            before_vertices = _sequence(
                before_panel["vertices"],
                f"pattern.panels.{panel_name}.vertices",
            )
            after_vertices = _sequence(
                after_panel["vertices"],
                f"pattern.panels.{panel_name}.vertices",
            )
            before_edges = _sequence(
                before_panel["edges"], f"pattern.panels.{panel_name}.edges"
            )
            after_edges = _sequence(
                after_panel["edges"], f"pattern.panels.{panel_name}.edges"
            )
        except KeyError as error:
            raise PatternGeometryError(
                f"pattern.panels.{panel_name} is missing {error.args[0]}"
            ) from error
        if not before_vertices or not after_vertices:
            raise PatternGeometryError(
                f"pattern.panels.{panel_name} must contain vertices"
            )
        if len(before_vertices) != len(after_vertices):
            raise TopologyMismatchError(
                f"pattern.panels.{panel_name} vertex counts differ"
            )
        if len(before_edges) != len(after_edges):
            raise TopologyMismatchError(
                f"pattern.panels.{panel_name} edge counts differ"
            )
        if not before_edges:
            raise PatternGeometryError(
                f"pattern.panels.{panel_name} must contain at least one edge"
            )
        for index, (before_edge, after_edge) in enumerate(
            zip(before_edges, after_edges)
        ):
            if not isinstance(before_edge, Mapping) or not isinstance(
                after_edge, Mapping
            ):
                raise PatternGeometryError(
                    f"pattern.panels.{panel_name}.edges[{index}] must be a mapping"
                )
            try:
                before_endpoints = tuple(before_edge["endpoints"])
                after_endpoints = tuple(after_edge["endpoints"])
            except (KeyError, TypeError) as error:
                raise PatternGeometryError(
                    f"pattern.panels.{panel_name}.edges[{index}].endpoints "
                    "must contain two indices"
                ) from error
            for endpoint_set, vertex_count, side in (
                (before_endpoints, len(before_vertices), "before"),
                (after_endpoints, len(after_vertices), "after"),
            ):
                if (
                    len(endpoint_set) != 2
                    or endpoint_set[0] == endpoint_set[1]
                    or any(
                        isinstance(endpoint, bool)
                        or not isinstance(endpoint, int)
                        or endpoint < 0
                        or endpoint >= vertex_count
                        for endpoint in endpoint_set
                    )
                ):
                    raise PatternGeometryError(
                        f"{side} pattern.panels.{panel_name}.edges[{index}].endpoints "
                        "must contain two distinct in-range integer indices"
                    )
            if before_endpoints != after_endpoints:
                raise TopologyMismatchError(
                    f"pattern.panels.{panel_name}.edges[{index}] endpoint indices differ"
                )
            try:
                before_type = edge_type(before_edge)
                after_type = edge_type(after_edge)
            except ValueError as error:
                raise PatternGeometryError(str(error)) from error
            if before_type != after_type:
                raise TopologyMismatchError(
                    f"pattern.panels.{panel_name}.edges[{index}] curve types differ"
                )
        aligned.append((panel_name, before_panel, after_panel))
    return aligned


def _normalized_arc_points(curve: Any, sample_count: int) -> tuple[complex, ...]:
    """Sample a curve at equally spaced fractions of its arc length."""

    import numpy as np

    length = float(curve.length())
    if not math.isfinite(length):
        raise PatternGeometryError("curve length is not finite")
    if length <= 0.0:
        point = complex(curve.point(0.0))
        return (point,) * sample_count

    points = []
    for fraction in np.linspace(0.0, 1.0, sample_count):
        if fraction == 0.0:
            parameter = 0.0
        elif fraction == 1.0:
            parameter = 1.0
        else:
            parameter = curve.ilength(float(fraction) * length)
        points.append(complex(curve.point(parameter)))
    return tuple(points)


def boundary_displacement_cm(
    before: Any,
    after: Any,
    *,
    samples_per_edge: int = DEFAULT_EDGE_SAMPLES,
) -> float:
    """Return maximum aligned local-boundary displacement in centimeters.

    Same-name panels and same-index edges must have matching topology.  Each
    curve is sampled at ``samples_per_edge`` equally spaced normalized
    arc-length positions.  Panel translation/rotation are intentionally
    ignored, and each local boundary is anchored at the panel's first vertex.
    """

    import numpy as np

    if (
        isinstance(samples_per_edge, bool)
        or not isinstance(samples_per_edge, int)
        or samples_per_edge < 2
    ):
        raise ValueError("samples_per_edge must be an integer of at least two")

    before_spec = _specification(before)
    after_spec = _specification(after)
    before_scale = _units_scale_cm(before_spec)
    after_scale = _units_scale_cm(after_spec)
    maximum = 0.0

    for panel_name, before_panel, after_panel in _aligned_panels(
        before_spec, after_spec
    ):
        try:
            before_vertices = np.asarray(before_panel["vertices"], dtype=float)
            after_vertices = np.asarray(after_panel["vertices"], dtype=float)
        except (TypeError, ValueError) as error:
            raise PatternGeometryError(
                f"pattern.panels.{panel_name}.vertices must be finite 2-D coordinates"
            ) from error
        if (
            before_vertices.ndim != 2
            or after_vertices.ndim != 2
            or before_vertices.shape[1:] != (2,)
            or after_vertices.shape[1:] != (2,)
            or not np.isfinite(before_vertices).all()
            or not np.isfinite(after_vertices).all()
        ):
            raise PatternGeometryError(
                f"pattern.panels.{panel_name}.vertices must be finite 2-D coordinates"
            )
        before_anchor = complex(*before_vertices[0])
        after_anchor = complex(*after_vertices[0])

        for edge_index, (before_edge, after_edge) in enumerate(
            zip(before_panel["edges"], after_panel["edges"])
        ):
            try:
                before_curve = edge_as_curve(before_vertices, before_edge)
                after_curve = edge_as_curve(after_vertices, after_edge)
                before_points = _normalized_arc_points(
                    before_curve, samples_per_edge
                )
                after_points = _normalized_arc_points(
                    after_curve, samples_per_edge
                )
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise PatternGeometryError(
                    f"failed to sample pattern.panels.{panel_name}.edges[{edge_index}]: "
                    f"{error}"
                ) from error

            before_local_cm = (
                np.asarray(before_points, dtype=complex) - before_anchor
            ) * before_scale
            after_local_cm = (
                np.asarray(after_points, dtype=complex) - after_anchor
            ) * after_scale
            index = int(
                np.argmax(np.abs(after_local_cm - before_local_cm))
            )
            displacement = abs(
                (after_points[index] - after_anchor) * after_scale
                - (before_points[index] - before_anchor) * before_scale
            )
            maximum = max(maximum, float(displacement))
    return maximum


def _resampled_panel_perimeter(
    panel: Mapping[str, Any],
    *,
    panel_path: str,
    dense_samples_per_edge: int,
    boundary_samples: int,
) -> np.ndarray:
    import numpy as np

    try:
        vertices = np.asarray(panel["vertices"], dtype=float)
        edges = _sequence(panel["edges"], f"{panel_path}.edges")
    except (KeyError, TypeError, ValueError) as error:
        raise PatternGeometryError(
            f"{panel_path} must contain finite vertices and edges"
        ) from error
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (2,)
        or len(vertices) < 2
        or not np.isfinite(vertices).all()
        or not edges
    ):
        raise PatternGeometryError(
            f"{panel_path} must contain a finite nonempty 2-D boundary"
        )

    dense_points: list[tuple[float, float]] = []
    first_start: int | None = None
    previous_end: int | None = None
    for edge_index, raw_edge in enumerate(edges):
        if not isinstance(raw_edge, Mapping):
            raise PatternGeometryError(
                f"{panel_path}.edges[{edge_index}] must be a mapping"
            )
        try:
            endpoints = tuple(raw_edge["endpoints"])
        except (KeyError, TypeError) as error:
            raise PatternGeometryError(
                f"{panel_path}.edges[{edge_index}] has invalid endpoints"
            ) from error
        if (
            len(endpoints) != 2
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(vertices)
                for index in endpoints
            )
            or endpoints[0] == endpoints[1]
        ):
            raise PatternGeometryError(
                f"{panel_path}.edges[{edge_index}] has invalid endpoints"
            )
        if edge_index == 0:
            first_start = endpoints[0]
        elif endpoints[0] != previous_end:
            raise TopologyMismatchError(
                f"{panel_path} edges are not one directed boundary cycle"
            )
        previous_end = endpoints[1]
        try:
            curve = edge_as_curve(vertices, raw_edge)
            for sample_index in range(dense_samples_per_edge):
                point = complex(
                    curve.point(sample_index / dense_samples_per_edge)
                )
                if not math.isfinite(point.real) or not math.isfinite(point.imag):
                    raise PatternGeometryError(
                        f"{panel_path}.edges[{edge_index}] sampled non-finite"
                    )
                dense_points.append((point.real, point.imag))
        except PatternGeometryError:
            raise
        except Exception as error:
            raise PatternGeometryError(
                f"failed to sample {panel_path}.edges[{edge_index}]: {error}"
            ) from error
    if previous_end != first_start:
        raise TopologyMismatchError(
            f"{panel_path} edges do not close one directed boundary cycle"
        )

    dense = np.asarray(dense_points + [dense_points[0]], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    if not np.isfinite(segment_lengths).all():
        raise PatternGeometryError(f"{panel_path} perimeter is not finite")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1.0e-12))
    cumulative = cumulative[keep]
    dense = dense[keep]
    if len(cumulative) < 2 or cumulative[-1] <= 0.0:
        raise PatternGeometryError(f"{panel_path} perimeter must be positive")
    query = np.linspace(
        0.0,
        float(cumulative[-1]),
        boundary_samples,
        endpoint=False,
    )
    sampled = np.column_stack(
        (
            np.interp(query, cumulative, dense[:, 0]),
            np.interp(query, cumulative, dense[:, 1]),
        )
    )
    return sampled - sampled[0]


def resampled_boundary_displacement_cm(
    before: Any,
    after: Any,
    panel_names: Sequence[str],
    *,
    dense_samples_per_edge: int = 65,
    boundary_samples: int = 1024,
) -> float:
    """Compare same-name perimeters without requiring equal edge segmentation."""

    import numpy as np

    for name, value, minimum in (
        ("dense_samples_per_edge", dense_samples_per_edge, 2),
        ("boundary_samples", boundary_samples, 16),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    requested = tuple(panel_names)
    if not requested or len(set(requested)) != len(requested):
        raise PatternGeometryError("panel_names must be nonempty and unique")
    before_spec = _specification(before)
    after_spec = _specification(after)
    before_panels = _panels(before_spec)
    after_panels = _panels(after_spec)
    if any(
        name not in before_panels or name not in after_panels
        for name in requested
    ):
        raise TopologyMismatchError("a requested panel is missing")
    before_scale = _units_scale_cm(before_spec)
    after_scale = _units_scale_cm(after_spec)
    maximum = 0.0
    for panel_name in requested:
        before_points = _resampled_panel_perimeter(
            before_panels[panel_name],
            panel_path=f"before.pattern.panels.{panel_name}",
            dense_samples_per_edge=dense_samples_per_edge,
            boundary_samples=boundary_samples,
        )
        after_points = _resampled_panel_perimeter(
            after_panels[panel_name],
            panel_path=f"after.pattern.panels.{panel_name}",
            dense_samples_per_edge=dense_samples_per_edge,
            boundary_samples=boundary_samples,
        )
        displacement = np.linalg.norm(
            after_points * after_scale - before_points * before_scale,
            axis=1,
        )
        if not np.isfinite(displacement).all():
            raise PatternGeometryError("resampled displacement is not finite")
        maximum = max(maximum, float(np.max(displacement)))
    return maximum
