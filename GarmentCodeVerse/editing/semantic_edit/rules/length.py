"""The four body-aware semantic length rules.

This module owns length classification, candidate generation, design mutation,
semantic transitions, and constructed-pattern validation.  The four rule IDs
share one data-driven implementation and differ only in their active paths and
pattern metric.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any

from editing.semantic_edit.prompt_state import (
    ActivePathError,
    ActivePaths,
    resolve_active_paths,
)
from editing.canonical_io import (
    canonical_hash,
    canonical_json_bytes,
    design_dotted_get,
    design_dotted_set,
    plain_copy,
)
from editing.config import EditConfig, ParameterSpec, RuleConfig
from editing.models import (
    EffectResult,
    LengthCategoryChannel,
    LengthCategoryObservation,
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
)
from editing.pattern_compare import edge_type, semantic_pattern_hash, topology_signature


PANTS_CATEGORIES = (
    "shorts", "knee-length shorts", "cropped", "ankle-length", "full-length",
)
SLEEVE_CATEGORIES = (
    "short", "elbow-length", "three-quarter-length", "long",
)
SHIRT_CATEGORIES = (
    "underbust-length", "cropped", "waist-length", "high-hip-length",
    "hip-length", "tunic-length",
)
SKIRT_CATEGORIES = (
    "mini", "above-knee", "knee-length", "midi-length", "ankle-length",
    "floor-length",
)

LENGTH_RULE_IDS = (
    "shirt_length", "sleeve_length", "skirt_length", "pants_length",
)

_GARMENT_FEATURE = {
    "shirt_length": "the shirt",
    "skirt_length": "the skirt",
    "pants_length": "the pants",
}
_CHANNEL_FEATURE = {
    "left": "the left sleeve",
    "right": "the right sleeve",
    "front_hem": "the front hem of the skirt",
    "back_hem": "the back hem of the skirt",
}
_SLEEVE_CHANNELS = {
    "both": ("left", "right"), "left": ("left",), "right": ("right",),
}
_HIGH_LOW_CHANNELS = ("front_hem", "back_hem")


class LengthCategoryError(ValueError):
    """A design/body pair cannot support the requested length category."""


class RuleNotApplicable(ValueError):
    """A legal length edit cannot be expressed from the current endpoint."""


def _nullish(value: Any) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _body_float(body: Mapping[str, Any], key: str) -> float:
    if key not in body or _nullish(body[key]):
        raise LengthCategoryError(f"missing required body measurement: {key}")
    try:
        return float(body[key])
    except (TypeError, ValueError) as exc:
        raise LengthCategoryError(
            f"non-numeric required body measurement: {key}"
        ) from exc


def _design_value(design: Mapping[str, Any], path: str, default: Any = None) -> Any:
    try:
        value = design_dotted_get(design, path)
    except (KeyError, TypeError, ValueError):
        return default
    return default if _nullish(value) else value


def _active(design: Mapping[str, Any]) -> ActivePaths:
    if not isinstance(design, Mapping):
        raise LengthCategoryError("design must be a mapping")
    try:
        return resolve_active_paths(design)
    except (ActivePathError, KeyError, TypeError, ValueError) as exc:
        raise LengthCategoryError(f"invalid active design: {exc}") from exc


def _channel(
    name: str, category: str, categories: tuple[str, ...]
) -> LengthCategoryChannel:
    return LengthCategoryChannel(name, category, categories.index(category))


def _observation(
    rule_id: str,
    category: str,
    categories: tuple[str, ...],
) -> LengthCategoryObservation:
    return LengthCategoryObservation(
        rule_id, "garment", (_channel("garment", category, categories),)
    )


def _bottom_ratio(length: Any, rise: Any, body: Mapping[str, Any]) -> float:
    centimetres = (
        _float(rise, 0.75) * _body_float(body, "hips_line")
        + _float(length, 0.65) * _body_float(body, "_leg_length")
    )
    waist_level = _body_float(body, "_waist_level")
    if waist_level == 0:
        raise LengthCategoryError("required body measurement _waist_level must be non-zero")
    return centimetres / waist_level


def _pants_category(value: float) -> str:
    for limit, category in (
        (0.48, "shorts"), (0.58, "knee-length shorts"),
        (0.84, "cropped"), (0.87, "ankle-length"),
    ):
        if value < limit:
            return category
    return "full-length"


def _sleeve_category(length: Any, width: Any) -> str:
    value = _float(length, 0.5)
    index = 0 if value < 0.35 else 1 if value < 0.72 else 2 if value < 0.94 else 3
    if _float(width, 0.0) >= 1.0:
        index = min(index + 1, 3)
    return SLEEVE_CATEGORIES[index]


def _shirt_category(length: Any, body: Mapping[str, Any]) -> str:
    ratio = _float(length, 1.0)
    waist_line = _body_float(body, "waist_line")
    hips_line = _body_float(body, "hips_line")
    if ratio < 0.70:
        return "underbust-length"
    if ratio < 0.95:
        return "cropped"
    if ratio < 1.15:
        return "waist-length"
    below_waist = ratio * waist_line - waist_line
    if below_waist <= 0.35 * hips_line:
        return "high-hip-length"
    if below_waist <= 1.05 * hips_line:
        return "hip-length"
    return "tunic-length"


def _normalized_skirt_category(length: Any, rise: Any) -> str:
    effective = _float(length, 0.65) + 0.20 * _float(rise, 0.75)
    for limit, category in (
        (0.45, "mini"), (0.55, "above-knee"),
        (0.70, "knee-length"), (0.97, "midi-length"),
    ):
        if effective < limit:
            return category
    return "floor-length"


def _body_skirt_category(
    length_cm: float, body: Mapping[str, Any], ankle_threshold: float = 0.82
) -> str:
    waist_level = _body_float(body, "_waist_level")
    if waist_level == 0:
        raise LengthCategoryError("required body measurement _waist_level must be non-zero")
    relative = length_cm / waist_level
    for limit, category in (
        (0.40, "mini"), (0.46, "above-knee"), (0.60, "knee-length"),
        (ankle_threshold, "midi-length"), (0.88, "ankle-length"),
    ):
        if relative < limit:
            return category
    return "floor-length"


def _ordinary_skirt_category(length: Any, rise: Any, body: Mapping[str, Any]) -> str:
    centimetres = (
        _float(rise, 0.75) * _body_float(body, "hips_line")
        + _float(length, 0.65) * _body_float(body, "_leg_length")
    )
    return _body_skirt_category(centimetres, body)


def _paneled_skirt_category(length: Any, rise: Any, suns: Any) -> str:
    effective = (
        _float(length, 0.65) + 0.20 * _float(rise, 0.75)
        + 0.15 * _float(suns, 0.5)
    )
    for limit, category in (
        (0.45, "mini"), (0.55, "above-knee"),
        (0.66, "knee-length"), (0.97, "midi-length"),
    ):
        if effective < limit:
            return category
    return "floor-length"


def classify_target(
    rule_id: str,
    design: Mapping[str, Any],
    body: Mapping[str, Any],
    *,
    scope: str | None = None,
) -> LengthCategoryObservation:
    """Classify one constructor-active length target."""

    if not isinstance(body, Mapping):
        raise LengthCategoryError("body must be a mapping")
    active = _active(design)
    if rule_id == "shirt_length":
        if scope not in {None, "garment"}:
            raise LengthCategoryError("length classifier requires garment scope")
        paths = active.paths_for(rule_id)
        if active.dispatch.get("upper") != "Shirt" or len(paths) != 1:
            raise LengthCategoryError("shirt_length is inapplicable to this design")
        return _observation(rule_id, _shirt_category(_design_value(design, paths[0]), body), SHIRT_CATEGORIES)

    if rule_id == "pants_length":
        if scope not in {None, "garment"}:
            raise LengthCategoryError("length classifier requires garment scope")
        paths = active.paths_for(rule_id)
        if active.dispatch.get("bottom") != "Pants" or len(paths) != 1:
            raise LengthCategoryError("pants_length is inapplicable to this design")
        category = _pants_category(_bottom_ratio(
            _design_value(design, paths[0]),
            _design_value(design, "design.pants.rise.v"), body,
        ))
        return _observation(rule_id, category, PANTS_CATEGORIES)

    if rule_id == "sleeve_length":
        if scope is not None and scope not in _SLEEVE_CHANNELS:
            raise LengthCategoryError(f"unsupported sleeve scope: {scope!r}")
        asymmetric = active.dispatch.get("asymmetric") is True
        if asymmetric:
            paths = {
                side: values[0]
                for side in ("left", "right")
                if len(values := active.paths_for(f"sleeve_length:{side}")) == 1
            }
        else:
            shared = active.paths_for("sleeve_length:both")
            paths = {"left": shared[0], "right": shared[0]} if len(shared) == 1 else {}
        if scope is None:
            scope = "both" if set(paths) == {"left", "right"} else "left" if "left" in paths else "right" if "right" in paths else None
        if scope is None or any(side not in paths for side in _SLEEVE_CHANNELS[scope]):
            raise LengthCategoryError("sleeve_length is inapplicable to requested scope")
        if not asymmetric and scope != "both":
            raise LengthCategoryError("shared symmetric sleeves require scope 'both'")
        channels = []
        for side in _SLEEVE_CHANNELS[scope]:
            prefix = "design.left.sleeve" if asymmetric and side == "left" else "design.sleeve"
            category = _sleeve_category(
                _design_value(design, paths[side]),
                _design_value(design, f"{prefix}.connecting_width.v"),
            )
            channels.append(_channel(side, category, SLEEVE_CATEGORIES))
        return LengthCategoryObservation(rule_id, scope, tuple(channels))

    if rule_id != "skirt_length":
        raise LengthCategoryError(f"unsupported length rule: {rule_id!r}")
    if scope not in {None, "garment"}:
        raise LengthCategoryError("length classifier requires garment scope")
    paths = active.paths_for(rule_id)
    bottom = active.dispatch.get("bottom")
    supported = {"Skirt2", "SkirtCircle", "AsymmSkirtCircle", "SkirtManyPanels", "GodetSkirt", "PencilSkirt", "SkirtLevels"}
    if len(paths) != 1 or bottom not in supported:
        raise LengthCategoryError("skirt_length is inapplicable to this design")
    length = _design_value(design, paths[0])
    if bottom == "Skirt2":
        category = _ordinary_skirt_category(length, _design_value(design, "design.skirt.rise.v"), body)
    elif bottom == "SkirtCircle":
        category = _normalized_skirt_category(length, _design_value(design, "design.flare-skirt.rise.v"))
    elif bottom == "AsymmSkirtCircle":
        back_cm = max(
            _float(_design_value(design, "design.flare-skirt.rise.v"), 0.75) * _body_float(body, "hips_line")
            + _float(length, 0.65) * _body_float(body, "_leg_length"),
            5.0,
        )
        front_cm = _float(_design_value(design, "design.flare-skirt.asymm.front_length.v"), 1.0) * back_cm
        front = _body_skirt_category(front_cm, body, 0.78)
        back = _body_skirt_category(back_cm, body, 0.78)
        return LengthCategoryObservation(rule_id, "garment", (
            _channel("front_hem", front, SKIRT_CATEGORIES),
            _channel("back_hem", back, SKIRT_CATEGORIES),
        ))
    elif bottom == "SkirtManyPanels":
        category = _paneled_skirt_category(
            length, _design_value(design, "design.flare-skirt.rise.v"),
            _design_value(design, "design.flare-skirt.suns.v"),
        )
    elif bottom == "PencilSkirt":
        category = _ordinary_skirt_category(length, _design_value(design, "design.pencil-skirt.rise.v"), body)
    elif bottom == "GodetSkirt":
        rise = "design.pencil-skirt.rise.v" if active.dispatch.get("skirt_base") == "PencilSkirt" else "design.skirt.rise.v"
        category = _ordinary_skirt_category(length, _design_value(design, rise), body)
    else:
        category = _normalized_skirt_category(length, _design_value(design, "design.levels-skirt.rise.v"))
    return _observation(rule_id, category, SKIRT_CATEGORIES)


def _state(rule_id: str, category: str) -> SemanticState:
    return SemanticState(
        rule_id, re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-"),
        category, {"category": category},
    )


def _inventory(observation: LengthCategoryObservation) -> tuple[str, ...]:
    channels = tuple(channel.name for channel in observation.channels)
    valid = (
        observation.scope == "garment" and channels == ("garment",)
        if observation.rule_id in {"shirt_length", "pants_length"}
        else channels == _SLEEVE_CHANNELS.get(observation.scope)
        if observation.rule_id == "sleeve_length"
        else observation.scope == "garment" and channels in {("garment",), _HIGH_LOW_CHANNELS}
        if observation.rule_id == "skirt_length"
        else False
    )
    if not valid:
        raise ValueError(f"invalid {observation.rule_id} category channels {channels!r}")
    return channels


def _feature(rule_id: str, channels: tuple[str, ...]) -> str:
    if channels == ("left", "right"):
        return "both sleeves"
    if channels == _HIGH_LOW_CHANNELS:
        return "the skirt"
    return _GARMENT_FEATURE[rule_id] if channels == ("garment",) else _CHANNEL_FEATURE[channels[0]]


def _transitions(
    source: LengthCategoryObservation,
    target: LengthCategoryObservation,
    direction: str,
) -> tuple[SemanticTransition, ...]:
    if direction not in {"increase", "decrease"}:
        raise ValueError("length direction must be 'increase' or 'decrease'")
    source_channels, target_channels = _inventory(source), _inventory(target)
    if (source.rule_id, source.scope, source_channels) != (
        target.rule_id, target.scope, target_channels,
    ):
        raise ValueError("source and target length observations do not match")
    values = []
    failures = []
    for before, after in zip(source.channels, target.channels):
        moved = after.order_index - before.order_index
        if before.category == after.category:
            failures.append((0, f"channel {before.name!r} remains {before.category!r}"))
        elif (direction == "increase" and moved < 0) or (direction == "decrease" and moved > 0):
            failures.append((1, f"channel {before.name!r} does not move {direction!r}"))
        else:
            values.append((before.name, before.category, after.category))
    if failures:
        raise ValueError(min(failures)[1])
    groups = (
        ((tuple(item[0] for item in values), *values[0][1:]),)
        if len(values) == 2 and values[0][1:] == values[1][1:]
        else tuple(((name,), before, after) for name, before, after in values)
    )
    return tuple(
        SemanticTransition(
            source.rule_id, source.scope, channels, _feature(source.rule_id, channels),
            _state(source.rule_id, before), _state(source.rule_id, after), direction,
        )
        for channels, before, after in groups
    )


def transitions_for(
    operation: Operation,
    source_design: Mapping[str, Any],
    target_design: Mapping[str, Any],
    body: Mapping[str, Any],
) -> tuple[SemanticTransition, ...]:
    """Return direct body-aware semantic transitions for a length operation."""

    try:
        source = classify_target(operation.rule_id, source_design, body, scope=operation.scope)
        target = classify_target(operation.rule_id, target_design, body, scope=operation.scope)
        return _transitions(source, target, operation.direction)
    except (LengthCategoryError, TypeError, ValueError) as exc:
        raise StateUnavailableError(
            "length_semantic_state_unavailable",
            f"cannot build {operation.rule_id} transitions: {exc}",
        ) from exc


@dataclass(frozen=True, slots=True)
class _Target:
    scope: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    specs: tuple[str, ...]


_SHIRT_LENGTH = "design.shirt.length.v"
_RIGHT_LENGTH = "design.sleeve.length.v"
_LEFT_LENGTH = "design.left.sleeve.length.v"
_PANTS_LENGTH = "design.pants.length.v"
_PANTS_LENGTH_RANGE = "design.pants.length.range"
_PANTS_CUFF_TYPE = "design.pants.cuff.type.v"
_PANTS_CUFF_LENGTH = "design.pants.cuff.cuff_len.v"
_PANTS_CUFF_LENGTH_RANGE = "design.pants.cuff.cuff_len.range"
_SKIRT_SPEC = {
    "design.skirt.length.v": "skirt",
    "design.flare-skirt.length.v": "flare_skirt",
    "design.pencil-skirt.length.v": "pencil_skirt",
    "design.levels-skirt.length.v": "levels_skirt",
}


def _targets(rule_id: str, context: RuleContext) -> tuple[_Target, ...]:
    active = context.active_paths
    if rule_id == "shirt_length":
        return (
            (_Target("garment", ("design.meta.upper.v", _SHIRT_LENGTH), (_SHIRT_LENGTH,), ("length",)),)
            if active.paths_for(rule_id) == (_SHIRT_LENGTH,) else ()
        )
    if rule_id == "pants_length":
        return (
            (_Target(
                "garment",
                (
                    "design.meta.bottom.v",
                    _PANTS_LENGTH,
                    _PANTS_LENGTH_RANGE,
                    _PANTS_CUFF_TYPE,
                    _PANTS_CUFF_LENGTH,
                    _PANTS_CUFF_LENGTH_RANGE,
                ),
                (_PANTS_LENGTH,),
                ("length",),
            ),)
            if active.paths_for(rule_id) == (_PANTS_LENGTH,) and active.dispatch["bottom"] == "Pants" else ()
        )
    if rule_id == "sleeve_length":
        result = []
        asymmetric = bool(active.dispatch.get("asymmetric", False))
        for scope in ("both", "left", "right"):
            writes = active.paths_for(f"sleeve_length:{scope}")
            if not writes or (not asymmetric and scope != "both"):
                continue
            reads = ["design.meta.upper.v", "design.left.enable_asym.v"]
            if _RIGHT_LENGTH in writes:
                reads.append("design.sleeve.sleeveless.v")
            if _LEFT_LENGTH in writes:
                reads.append("design.left.sleeve.sleeveless.v")
            if active.dispatch.get("upper") == "FittedShirt":
                if _RIGHT_LENGTH in writes:
                    reads.append("design.shirt.strapless.v")
                if _LEFT_LENGTH in writes:
                    reads.append("design.left.shirt.strapless.v")
            reads.extend(writes)
            result.append(_Target(scope, tuple(dict.fromkeys(reads)), writes, ("length",) * len(writes)))
        return tuple(result)
    writes = active.paths_for("skirt_length")
    if len(writes) != 1:
        return ()
    path = writes[0]
    if path not in _SKIRT_SPEC:
        raise ValueError(f"unsupported active skirt length path: {path}")
    reads = ["design.meta.bottom.v"]
    bottom = active.dispatch["bottom"]
    if bottom == "GodetSkirt":
        reads.append("design.godet-skirt.base.v")
    elif bottom == "SkirtLevels":
        reads.extend(("design.levels-skirt.base.v", "design.levels-skirt.level.v"))
    reads.append(path)
    return (_Target("garment", tuple(reads), (path,), (_SKIRT_SPEC[path],)),)


_RULE_META = {
    "shirt_length": {
        "reads": {"design.meta.upper.v", _SHIRT_LENGTH},
        "writes": {_SHIRT_LENGTH},
        "conflicts": (),
        "prerequisites": ("upper_is_Shirt",),
        "feature": "shirt length",
    },
    "sleeve_length": {
        "reads": {
            "design.meta.upper.v", "design.left.enable_asym.v",
            "design.sleeve.sleeveless.v", "design.left.sleeve.sleeveless.v",
            "design.shirt.strapless.v", "design.left.shirt.strapless.v",
            _RIGHT_LENGTH, _LEFT_LENGTH,
        },
        "writes": {_RIGHT_LENGTH, _LEFT_LENGTH},
        "conflicts": ("sleeve_toggle",),
        "prerequisites": ("upper_exists", "target_sleeve_exists"),
        "feature": "sleeve length",
    },
    "skirt_length": {
        "reads": {
            "design.meta.bottom.v", "design.godet-skirt.base.v",
            "design.levels-skirt.base.v", "design.levels-skirt.level.v",
            *_SKIRT_SPEC,
        },
        "writes": set(_SKIRT_SPEC),
        "conflicts": (),
        "prerequisites": ("supported_skirt_exists",),
        "feature": "skirt length",
    },
    "pants_length": {
        "reads": {
            "design.meta.bottom.v",
            _PANTS_LENGTH,
            _PANTS_LENGTH_RANGE,
            _PANTS_CUFF_TYPE,
            _PANTS_CUFF_LENGTH,
            _PANTS_CUFF_LENGTH_RANGE,
        },
        "writes": {_PANTS_LENGTH},
        "conflicts": (),
        "prerequisites": ("bottom_is_Pants",),
        "feature": "pants length",
    },
}


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperationContractError(f"continuous parameter {path} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OperationContractError(f"continuous parameter {path} is non-finite")
    return result


def _range_lower(design: Mapping[str, Any], path: str) -> float:
    values = design_dotted_get(design, path)
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or len(values) != 2
    ):
        raise OperationContractError(f"parameter range {path} must have two values")
    lower = _number(values[0], f"{path}[0]")
    upper = _number(values[1], f"{path}[1]")
    if lower < 0 or lower >= upper:
        raise OperationContractError(f"parameter range {path} is invalid")
    return lower


def _pants_length_floor(
    design: Mapping[str, Any],
    configured_lower: float,
) -> float:
    """Minimum total pants length that keeps an enabled cuff non-degenerate."""

    pants_lower = max(
        configured_lower,
        _range_lower(design, _PANTS_LENGTH_RANGE),
    )
    if _nullish(design_dotted_get(design, _PANTS_CUFF_TYPE)):
        return pants_lower

    cuff_length = _number(
        design_dotted_get(design, _PANTS_CUFF_LENGTH),
        _PANTS_CUFF_LENGTH,
    )
    cuff_lower = _range_lower(design, _PANTS_CUFF_LENGTH_RANGE)
    if cuff_lower <= 0:
        raise OperationContractError("enabled pants cuff must have a positive lower bound")
    if cuff_length < cuff_lower:
        raise RuleNotApplicable(
            "enabled pants cuff length is below its configured lower bound"
        )
    return pants_lower + cuff_lower


def _continuous_value(
    current_value: Any,
    path: str,
    spec: ParameterSpec,
    direction: str,
    strength: float,
    minimum: float | None = None,
) -> tuple[float, float, bool]:
    current = _number(current_value, path)
    if spec.safe_bounds is None:
        raise OperationContractError(f"continuous parameter {path} has no safe bounds")
    configured_lower, upper = spec.safe_bounds
    lower = configured_lower if minimum is None else max(configured_lower, minimum)
    if lower >= upper:
        raise RuleNotApplicable(f"effective lower bound for {path} reaches its upper bound")
    if not lower <= current <= upper:
        raise RuleNotApplicable(f"current value for {path} is outside configured safe bounds")
    sign = 1.0 if direction == "increase" else -1.0
    if spec.edit_mode == "multiplicative":
        raw, denominator = current * (1.0 + sign * strength), abs(current)
    elif spec.edit_mode == "safe_span":
        span = upper - configured_lower
        raw = current + sign * strength * span
        denominator = spec.scale if spec.scale is not None else span
    else:
        raise OperationContractError(f"parameter {path} is not a continuous edit: {spec.edit_mode}")
    target = min(max(raw, lower), upper)
    moved = target - current
    if (direction == "increase" and moved <= 0) or (direction == "decrease" and moved >= 0):
        raise RuleNotApplicable(f"requested {direction} direction is erased by safe bounds for {path}")
    if denominator <= 0:
        raise OperationContractError(f"parameter {path} has a non-positive edit scale")
    return target, abs(moved) / denominator, raw != target


def _operation(
    rule_id: str,
    rule: RuleConfig,
    strengths: Mapping[str, float],
    context: RuleContext,
    target: _Target,
    direction: str,
    magnitude: str,
) -> Operation:
    before = {path: design_dotted_get(context.design, path) for path in target.writes}
    after = {}
    normalized = []
    clamped = False
    for path, spec_key in zip(target.writes, target.specs):
        spec = rule.parameter_specs[spec_key]
        minimum = None
        if rule_id == "pants_length" and path == _PANTS_LENGTH:
            if spec.safe_bounds is None:
                raise OperationContractError(
                    "pants_length requires configured safe bounds"
                )
            minimum = _pants_length_floor(context.design, spec.safe_bounds[0])
        value, change, was_clamped = _continuous_value(
            before[path], path, spec, direction,
            strengths[magnitude],
            minimum,
        )
        after[path] = value
        normalized.append(change)
        clamped |= was_clamped
    relative_change = sum(normalized) / len(normalized)
    tiers = sorted(strengths, key=lambda name: (strengths[name], name))
    realized = min(tiers, key=lambda name: (abs(strengths[name] - relative_change), strengths[name]))
    meta = _RULE_META[rule_id]
    return Operation(
        rule_id=rule_id,
        semantic_group=rule.semantic_group,
        scope=target.scope,
        direction=direction,
        magnitude=realized,
        structural_level=rule.structural_level,
        reads=tuple(sorted(target.reads)),
        writes=tuple(sorted(target.writes)),
        conflicts=tuple(sorted(meta["conflicts"])),
        prerequisites=tuple(sorted(meta["prerequisites"])),
        edit_mode=next(iter({spec.edit_mode for spec in rule.parameter_specs.values()})),
        effect_metric_id=rule.effect_metric,
        effect_contract_id=rule.effect_contract,
        before=before,
        after=after,
        relative_change=float(relative_change),
        requested_magnitude=magnitude,
        realized_magnitude=realized,
        clamped=clamped,
        instruction_tokens={
            "feature": meta["feature"],
            "increase_verb": "lengthen",
            "decrease_verb": "shorten",
            "scope": target.scope,
            "direction": direction,
            "requested_magnitude": magnitude,
            "realized_magnitude": realized,
        },
    )


def apply(design: dict[str, Any], planned: PlannedOperation) -> dict[str, Any]:
    """Apply one already planned length operation."""

    edited: Mapping[str, Any] = design
    for path in planned.operation.writes:
        edited = plain_copy(
            design_dotted_set(edited, path, planned.operation.after[path]),
            tuples_as_lists=True,
        )
    return plain_copy(edited, tuples_as_lists=True)


def _candidate(
    operation: Operation,
    context: RuleContext,
) -> PlannedOperation | None:
    edited: Mapping[str, Any] = context.design
    for path in operation.writes:
        edited = plain_copy(
            design_dotted_set(edited, path, operation.after[path]),
            tuples_as_lists=True,
        )
    try:
        transitions = transitions_for(operation, context.design, edited, context.body)
    except StateUnavailableError:
        return None
    evidence = {
        "rule_id": operation.rule_id,
        "scope": operation.scope,
        "source_design_hash": canonical_hash(context.design),
        "body_hash": canonical_hash(context.body),
        "active_paths_design_hash": context.active_paths.design_hash,
        "requested_magnitude": operation.requested_magnitude,
        "active_path_match": context.active_paths.design_hash == canonical_hash(context.design),
        "source_states": [item.source_state.to_payload() for item in transitions],
        "target_states": [item.target_state.to_payload() for item in transitions],
    }
    return PlannedOperation(
        operation=operation,
        transitions=transitions,
        topology_kind=TopologyKind.NONE,
        applicability_evidence=evidence,
        conflict_evidence=empty_conflict_evidence(operation.rule_id),
    )


def _candidates(
    rule_id: str,
    rule: RuleConfig,
    strengths: Mapping[str, float],
    order: tuple[str, ...],
    context: RuleContext,
) -> tuple[PlannedOperation, ...]:
    if canonical_hash(context.design) != context.active_paths.design_hash:
        raise OperationContractError("stale active-path context")
    meta = _RULE_META[rule_id]
    result = []
    seen = set()
    for magnitude in order:
        for target in _targets(rule_id, context):
            if not set(target.reads) <= meta["reads"] or not set(target.writes) <= meta["writes"]:
                raise OperationContractError(f"{rule_id} target resolved undeclared paths")
            for path in target.reads:
                design_dotted_get(context.design, path)
            for direction in ("increase", "decrease"):
                try:
                    operation = _operation(rule_id, rule, strengths, context, target, direction, magnitude)
                except RuleNotApplicable:
                    continue
                candidate = _candidate(operation, context)
                if candidate is None:
                    continue
                endpoint = canonical_hash({
                    "rule_id": rule_id,
                    "scope": operation.scope,
                    "direction": direction,
                    "reads": operation.reads,
                    "writes": operation.writes,
                    "before": operation.before,
                    "after": operation.after,
                    "transitions": [item.to_payload() for item in candidate.transitions],
                })
                if endpoint not in seen:
                    seen.add(endpoint)
                    result.append(candidate)
    return tuple(result)


_TORSO_PANEL = re.compile(r"^(?:left|right)_[fb]torso$")
_SLEEVE_PANEL = re.compile(r"^(left|right)_sleeve_([fb])$")
_PANTS_MAIN_PANEL = re.compile(r"^pant_[fb]_[lr]$")
_PANTS_COMPONENT_PANEL = re.compile(
    r"^(?:pant_[fb]_[lr]|pant_[lr]_cuff(?:_skirt)?_[fb])$"
)
_EPSILON = 1.0e-12
_CURVE_SAMPLES = 101
_IGNORED_TOP = {"name", "sample_name", "path", "source_path", "spec_file"}


@dataclass(slots=True)
class _Measurement:
    before: Any
    after: Any
    delta: Any
    changed: bool
    direction_ok: bool
    effect_size_cm: float
    panels: tuple[str, ...]
    details: dict[str, Any]


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return value


def _panels(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    panels = specification["pattern"]["panels"]
    if not isinstance(panels, Mapping):
        raise ValueError("pattern.panels must be a mapping")
    return panels


def _scale(specification: Mapping[str, Any]) -> float:
    value = specification["properties"]["units_in_meter"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("properties.units_in_meter must be a finite positive number")
    return 100.0 / float(value)


def _finite_vector(
    values: Sequence[Any], path: str, length: int
) -> tuple[float, ...]:
    if len(values) != length or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"{path} must contain {length} finite numbers")
    return tuple(float(value) for value in values)


def _vertices(panel: Mapping[str, Any], path: str) -> tuple[tuple[float, float], ...]:
    result = []
    for index, point in enumerate(_sequence(panel["vertices"], f"{path}.vertices")):
        point = _sequence(point, f"{path}.vertices[{index}]")
        if len(point) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) for value in point
        ):
            raise ValueError(f"{path}.vertices[{index}] must contain two finite numbers")
        result.append((float(point[0]), float(point[1])))
    if not result:
        raise ValueError(f"{path}.vertices must not be empty")
    return tuple(result)


def _vertical_extent(specification: Mapping[str, Any], name: str) -> float:
    points = _vertices(_panels(specification)[name], f"pattern.panels.{name}")
    ys = [point[1] for point in points]
    return (max(ys) - min(ys)) * _scale(specification)


def _numeric_measurement(
    before: Mapping[str, float],
    after: Mapping[str, float],
    direction: str,
) -> tuple[dict[str, float], bool, bool, float]:
    if set(before) != set(after) or not before:
        raise ValueError("metric channel membership changed")
    delta = {key: after[key] - before[key] for key in sorted(before)}
    changed = any(abs(value) > _EPSILON for value in delta.values())
    direction_ok = all(
        value > _EPSILON if direction == "increase" else value < -_EPSILON
        for value in delta.values()
    )
    return delta, changed, direction_ok, min(abs(value) for value in delta.values())


def _torso_measurement(
    before: Mapping[str, Any], after: Mapping[str, Any], direction: str, scope: str
) -> _Measurement:
    del scope
    names = tuple(sorted(name for name in _panels(before) if _TORSO_PANEL.fullmatch(name)))
    after_names = tuple(sorted(name for name in _panels(after) if _TORSO_PANEL.fullmatch(name)))
    if names != after_names or not names:
        raise ValueError("torso panel membership is missing or changed")
    before_values = {name: _vertical_extent(before, name) for name in names}
    after_values = {name: _vertical_extent(after, name) for name in names}
    delta, changed, direction_ok, size = _numeric_measurement(before_values, after_values, direction)
    return _Measurement(before_values, after_values, delta, changed, direction_ok, size, names, {"channel_kind": "torso_panel_vertical_extent_cm"})


def _edge_length(specification: Mapping[str, Any], name: str, index: int) -> float:
    panel = _panels(specification)[name]
    vertices = _vertices(panel, f"pattern.panels.{name}")
    edge = _sequence(panel["edges"], f"pattern.panels.{name}.edges")[index]
    endpoints = _sequence(edge.get("endpoints"), f"pattern.panels.{name}.edges[{index}].endpoints")
    if len(endpoints) != 2 or any(type(value) is not int or not 0 <= value < len(vertices) for value in endpoints):
        raise ValueError("sleeve seam endpoints must be valid vertex indices")
    if edge_type(edge) != "line":
        from editing.pattern_geometry import edge_as_curve

        try:
            length = float(edge_as_curve(vertices, edge).length())
        except Exception as exc:
            raise ValueError(
                f"pattern.panels.{name}.edges[{index}] curve length failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    else:
        first, second = vertices[endpoints[0]], vertices[endpoints[1]]
        length = math.hypot(second[0] - first[0], second[1] - first[1])
    if not math.isfinite(length) or length <= 0:
        raise ValueError("sleeve seam has invalid length")
    return length * _scale(specification)


def _sleeve_values(specification: Mapping[str, Any]) -> dict[str, float]:
    names: dict[str, dict[str, str]] = {}
    for name in _panels(specification):
        if match := _SLEEVE_PANEL.fullmatch(name):
            side, half = match.groups()
            names.setdefault(side, {})[half] = name
    result = {}
    for side, halves in sorted(names.items()):
        if set(halves) != {"f", "b"}:
            raise ValueError(f"{side} sleeve requires front and back panels")
        result[side] = sum(_edge_length(specification, halves[half], 1) for half in ("f", "b")) / 2.0
    return result


def _sleeve_measurement(
    before: Mapping[str, Any], after: Mapping[str, Any], direction: str, scope: str
) -> _Measurement:
    before_all, after_all = _sleeve_values(before), _sleeve_values(after)
    if set(before_all) != set(after_all):
        raise ValueError("sleeve-side membership changed")
    sides = ("left", "right") if scope == "both" else (scope,)
    if any(side not in before_all for side in sides):
        raise ValueError(f"scope {scope!r} does not identify existing sleeve panels")
    before_values = {side: before_all[side] for side in sides}
    after_values = {side: after_all[side] for side in sides}
    delta, changed, direction_ok, size = _numeric_measurement(before_values, after_values, direction)
    untargeted = {side: after_all[side] - before_all[side] for side in sorted(set(before_all) - set(sides))}
    direction_ok &= all(abs(value) <= _EPSILON for value in untargeted.values())
    panels = tuple(sorted(
        name for name in _panels(before)
        if (match := _SLEEVE_PANEL.fullmatch(name)) and match.group(1) in sides
    ))
    return _Measurement(before_values, after_values, delta, changed, direction_ok, size, panels, {
        "channel_kind": "mean_front_back_sleeve_side_seam_cm",
        "untargeted_side_delta_cm": untargeted,
    })


def _rotation_matrix(values: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
    x, y, z = (
        math.radians(value)
        for value in _finite_vector(values, "panel rotation", 3)
    )
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    matrices = (
        ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)),
        ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)),
        ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)),
    )
    def multiply(left, right):
        return tuple(tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)) for row in range(3))
    return multiply(multiply(matrices[0], matrices[1]), matrices[2])


def _assembled_extent(specification: Mapping[str, Any], names: tuple[str, ...]) -> float:
    world_y = []
    for name in names:
        panel = _panels(specification)[name]
        vertices = _vertices(panel, f"pattern.panels.{name}")
        points = list(vertices)
        curved = [
            (index, edge)
            for index, edge in enumerate(
                _sequence(panel["edges"], f"pattern.panels.{name}.edges")
            )
            if edge_type(edge) != "line"
        ]
        if curved:
            from editing.pattern_geometry import edge_as_curve

            for edge_index, edge in curved:
                path = f"pattern.panels.{name}.edges[{edge_index}]"
                try:
                    curve = edge_as_curve(vertices, edge)
                except Exception as exc:
                    raise ValueError(
                        f"{path} curve conversion failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                for index in range(_CURVE_SAMPLES):
                    try:
                        point = complex(
                            curve.point(index / (_CURVE_SAMPLES - 1))
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"{path} curve sampling failed: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    if not math.isfinite(point.real) or not math.isfinite(point.imag):
                        raise ValueError("curved boundary sample must be finite")
                    points.append((float(point.real), float(point.imag)))
        rotation = _rotation_matrix(_sequence(panel.get("rotation"), f"pattern.panels.{name}.rotation"))
        translation = _finite_vector(
            _sequence(panel.get("translation"), f"pattern.panels.{name}.translation"),
            "panel translation",
            3,
        )
        for x, y in points:
            value = rotation[1][0] * x + rotation[1][1] * y + translation[1]
            if not math.isfinite(value):
                raise ValueError("assembled panel position must be finite")
            world_y.append(value)
    return (max(world_y) - min(world_y)) * _scale(specification)


def _assembled_measurement(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    direction: str,
    names: Callable[[Mapping[str, Any]], tuple[str, ...]],
    membership_error: str,
    channel_kind: str,
) -> _Measurement:
    before_names, after_names = names(before), names(after)
    if before_names != after_names:
        raise ValueError(membership_error)
    before_value = _assembled_extent(before, before_names)
    after_value = _assembled_extent(after, after_names)
    delta = after_value - before_value
    return _Measurement(
        before_value, after_value, delta, abs(delta) > _EPSILON,
        delta > _EPSILON if direction == "increase" else delta < -_EPSILON,
        abs(delta), before_names,
        {"channel_kind": channel_kind, "after_target_panels": after_names},
    )


def _skirt_names(specification: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(sorted(name for name, panel in _panels(specification).items() if isinstance(panel, Mapping) and panel.get("label") == "leg"))
    if not names:
        raise ValueError("skirt metric requires at least one leg-labeled panel")
    return names


def _pants_names(specification: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(sorted(name for name in _panels(specification) if _PANTS_COMPONENT_PANEL.fullmatch(name)))
    if tuple(name for name in names if _PANTS_MAIN_PANEL.fullmatch(name)) != ("pant_b_l", "pant_b_r", "pant_f_l", "pant_f_r"):
        raise ValueError("pants metric requires exactly the four main panels")
    return names


def _measure(
    rule_id: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    direction: str,
    scope: str,
) -> _Measurement:
    if rule_id == "shirt_length":
        return _torso_measurement(before, after, direction, scope)
    if rule_id == "sleeve_length":
        return _sleeve_measurement(before, after, direction, scope)
    if rule_id == "skirt_length":
        return _assembled_measurement(
            before,
            after,
            direction,
            _skirt_names,
            "active skirt panel membership changed",
            "assembled_leg_panel_vertical_extent_cm",
        )
    if rule_id == "pants_length":
        return _assembled_measurement(
            before,
            after,
            direction,
            _pants_names,
            "pants component panel membership changed",
            "assembled_pants_component_vertical_extent_cm",
        )
    raise ValueError(f"unsupported length rule: {rule_id}")


def _validate_request(operation: Operation, config: EditConfig) -> RuleConfig:
    if operation.rule_id not in LENGTH_RULE_IDS:
        raise ValueError(f"unsupported length rule: {operation.rule_id}")
    rule = config.rules[operation.rule_id]
    if operation.effect_metric_id != rule.effect_metric:
        raise ValueError(
            f"{operation.rule_id} requires metric {rule.effect_metric!r}"
        )
    if operation.effect_contract_id != rule.effect_contract:
        raise ValueError(
            f"{operation.rule_id} requires contract {rule.effect_contract!r}"
        )
    if operation.direction not in {"increase", "decrease"}:
        raise ValueError("length direction must be 'increase' or 'decrease'")
    scopes = (
        {"left", "right", "both"}
        if operation.rule_id == "sleeve_length"
        else {"garment"}
    )
    if operation.scope not in scopes:
        raise ValueError(
            f"{operation.rule_id} does not support scope {operation.scope!r}"
        )
    return rule


def _metadata(specification: Mapping[str, Any]) -> dict[str, Any]:
    pattern = specification["pattern"]
    return {
        "top": {
            key: value for key, value in specification.items()
            if key not in _IGNORED_TOP and key != "pattern"
        },
        "pattern": {
            key: value for key, value in pattern.items()
            if key not in {"panels", "stitches", "panel_order"}
        },
    }


def _pattern_subset(
    specification: Mapping[str, Any], names: tuple[str, ...]
) -> dict[str, Any]:
    panels = _panels(specification)
    return {
        "pattern": {
            "panels": {name: panels[name] for name in names},
            "stitches": [],
            "panel_order": list(names),
        },
        "parameters": {},
        "parameter_order": [],
        "properties": specification["properties"],
    }


def _boundary_displacement(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    names: tuple[str, ...],
    topology_policy: str,
) -> float:
    from editing import pattern_geometry

    if topology_policy == "local":
        return pattern_geometry.resampled_boundary_displacement_cm(before, after, names)
    return pattern_geometry.boundary_displacement_cm(
        _pattern_subset(before, names), _pattern_subset(after, names)
    )


def _evidence_payload(
    rule: RuleConfig,
    operation: Operation,
    measurement: _Measurement,
    details: Mapping[str, Any],
    passed: bool,
) -> dict[str, Any]:
    return {
        "metric_id": operation.effect_metric_id,
        "before": measurement.before,
        "after": measurement.after,
        "delta": measurement.delta,
        "expected_direction": operation.direction,
        "threshold": rule.perceptibility_cm,
        "passed": passed,
        "details": details,
        "metric_threshold_cm": rule.effective_metric_threshold_cm,
        "boundary_threshold_cm": rule.effective_boundary_threshold_cm,
    }


def _failure_payload(
    operation: Operation,
    code: str,
    message: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "code": code,
        "metric_id": operation.effect_metric_id,
        "contract_id": operation.effect_contract_id,
        "message": message,
        "details": details,
    }


def _result(
    operation: Operation,
    *,
    evidence: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
) -> EffectResult:
    code = None if failure is None else str(failure["code"])
    stage = (
        "metric" if code == "metric_threshold_failed"
        else "boundary" if code == "boundary_threshold_failed"
        else "effect"
    )
    gate_failure = [] if failure is None else [{
        "code": code,
        "operation_index": 0,
        "subject": operation.effect_metric_id,
        "details": {
            "contract_id": operation.effect_contract_id,
            "message": failure["message"],
            "failure_details": failure["details"],
        },
    }]
    v4 = {
        "passed": failure is None,
        "evidence": evidence,
        "failure": failure,
    }
    gate = {
        "gate": stage,
        "passed": failure is None,
        "failure_code": code,
        "failures": gate_failure,
        "length_transitions": [],
        "parameter_evidence": [],
        "effect_evidence_count": 0 if evidence is None else 1,
    }
    return EffectResult(
        failure is None,
        "effect/passed" if failure is None else code,
        {"rule_id": operation.rule_id, "details": {
            "v4_effect_result": v4,
            "v4_effect_gate": gate,
        }},
        stage,
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
    """Validate one constructed length edit and preserve its evidence schema."""

    del source_design, target_design, body
    if topology_policy not in {"strict", "local"}:
        raise ValueError("topology_policy must be exactly 'strict' or 'local'")
    operation = planned_operation.operation
    try:
        rule = _validate_request(operation, config)
        before_topology = topology_signature(source_specification)
        after_topology = topology_signature(target_specification)
        measurement = _measure(
            operation.rule_id,
            source_specification,
            target_specification,
            operation.direction,
            operation.scope,
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        failure = _failure_payload(
            operation,
            "effect_metric_failed",
            f"{type(exc).__name__}: {exc}",
            {},
        )
        return _result(operation, evidence=None, failure=failure)

    details = {
        **measurement.details,
        "contract_id": operation.effect_contract_id,
        "scope": operation.scope,
        "target_panels": measurement.panels,
        "metric_threshold_cm": rule.effective_metric_threshold_cm,
        "boundary_threshold_cm": rule.effective_boundary_threshold_cm,
        "metric_effect_size_cm": measurement.effect_size_cm,
    }

    def fail(code: str, message: str) -> EffectResult:
        evidence = _evidence_payload(rule, operation, measurement, details, False)
        return _result(
            operation,
            evidence=evidence,
            failure=_failure_payload(operation, code, message, details),
        )

    try:
        if semantic_pattern_hash(source_specification) == semantic_pattern_hash(target_specification):
            return fail("operation_noop", "before and after patterns are semantically identical")
        strict_match = (
            canonical_json_bytes(_metadata(source_specification))
            == canonical_json_bytes(_metadata(target_specification))
            and before_topology == after_topology
        )
        if topology_policy == "strict" and not strict_match:
            details["strict_topology_match"] = False
            return fail(
                "undeclared_topology_change",
                "panel metadata, edge topology, or stitch topology changed unexpectedly",
            )
        displacement = _boundary_displacement(
            source_specification,
            target_specification,
            measurement.panels,
            topology_policy,
        )
        details.update({
            "target_boundary_displacement_cm": displacement,
            "topology_policy": topology_policy,
            "strict_topology_match": strict_match,
        })
    except (KeyError, TypeError, ValueError, ArithmeticError) as exc:
        return fail("effect_metric_failed", f"{type(exc).__name__}: {exc}")
    if not measurement.changed:
        return fail("operation_noop", "the requested metric did not change")
    if not measurement.direction_ok:
        return fail(
            "effect_direction_failed",
            "the observed metric moved outside the requested scope or direction",
        )
    if measurement.effect_size_cm < rule.effective_metric_threshold_cm:
        return fail("metric_threshold_failed", "the observed metric effect is below its threshold")
    if displacement < rule.effective_boundary_threshold_cm:
        return fail("boundary_threshold_failed", "the target boundary displacement is below its threshold")
    return _result(
        operation,
        evidence=_evidence_payload(rule, operation, measurement, details, True),
        failure=None,
    )


def build_rules(config: EditConfig) -> tuple[Rule, ...]:
    """Bind all four length rules to one loaded edit configuration."""

    missing = [rule_id for rule_id in LENGTH_RULE_IDS if rule_id not in config.rules]
    if missing:
        raise ValueError(f"edit config is missing selected rules: {', '.join(missing)}")
    strengths = {
        tier: float(config.magnitude[tier].strength) for tier in config.magnitude
    }

    def build(rule_id: str) -> Rule:
        rule = config.rules[rule_id]

        def candidates(context: RuleContext) -> tuple[PlannedOperation, ...]:
            return _candidates(
                rule_id,
                rule,
                strengths,
                config.magnitude_attempt_order,
                context,
            )

        return Rule(rule_id, candidates, apply, validate)

    return tuple(build(rule_id) for rule_id in LENGTH_RULE_IDS)
