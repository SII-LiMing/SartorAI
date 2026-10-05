"""Strict configuration loading for edit-dataset generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml


APPROVED_SEMANTIC_GROUPS = frozenset(
    {"upper", "sleeve", "collar", "pants", "skirt", "waistband", "symmetry"}
)
APPROVED_MAGNITUDE_TIERS = ("slightly", "moderately", "significantly")
EDIT_MODES = frozenset(
    {"multiplicative", "safe_span", "discrete_step", "toggle", "enum"}
)
VALUE_TYPES = frozenset({"float", "int", "bool", "enum"})
MODE_VALUE_TYPES = {
    "multiplicative": frozenset({"float"}),
    "safe_span": frozenset({"float"}),
    "discrete_step": frozenset({"float", "int"}),
    "toggle": frozenset({"bool"}),
    "enum": frozenset({"enum", "int"}),
}


class ConfigError(ValueError):
    """Raised when an edit-generation config violates its schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConfigError(f"unhashable YAML mapping key: {key!r}") from exc
        if duplicate:
            raise ConfigError(f"duplicate key in YAML config: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class MagnitudeTier:
    strength: float


@dataclass(frozen=True)
class ParameterSpec:
    edit_mode: str
    value_type: str
    safe_bounds: Optional[Tuple[float, float]] = None
    transitions: Tuple[Any, ...] = ()
    step: Optional[float] = None
    scale: Optional[float] = None


@dataclass(frozen=True)
class RuleConfig:
    enabled: bool
    semantic_group: str
    structural_level: int
    effect_metric: str
    effect_contract: str
    perceptibility_cm: float
    parameter_specs: Mapping[str, ParameterSpec]
    metric_threshold_cm: Optional[float] = None
    boundary_threshold_cm: Optional[float] = None
    parameter_delta_mode: Optional[str] = None
    min_parameter_delta: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_specs, Mapping):
            raise TypeError("parameter_specs must be a mapping")
        parameter_specs = dict(self.parameter_specs)
        if any(not isinstance(name, str) or not name for name in parameter_specs):
            raise TypeError("parameter_specs keys must be non-empty strings")
        if any(
            not isinstance(spec, ParameterSpec) for spec in parameter_specs.values()
        ):
            raise TypeError("parameter_specs values must be ParameterSpec values")
        object.__setattr__(
            self,
            "parameter_specs",
            deepcopy(parameter_specs),
        )

    @property
    def effective_metric_threshold_cm(self) -> float:
        return (
            self.perceptibility_cm
            if self.metric_threshold_cm is None
            else self.metric_threshold_cm
        )

    @property
    def effective_boundary_threshold_cm(self) -> float:
        return (
            self.perceptibility_cm
            if self.boundary_threshold_cm is None
            else self.boundary_threshold_cm
        )


@dataclass(frozen=True)
class EditConfig:
    version: int
    expected_source_count: int
    magnitude: Mapping[str, MagnitudeTier]
    rules: Mapping[str, RuleConfig]
    magnitude_attempt_order: Tuple[str, ...]


_REQUIRED_TOP_LEVEL_KEYS = {
    "version",
    "expected_source_count",
    "magnitude",
    "magnitude_attempt_order",
    "rules",
}

_REQUIRED_RULE_KEYS = {
    "enabled",
    "semantic_group",
    "structural_level",
    "effect_metric",
    "effect_contract",
    "perceptibility_cm",
    "parameter_specs",
}
_OPTIONAL_RULE_KEYS = {
    "metric_threshold_cm",
    "boundary_threshold_cm",
    "parameter_delta_mode",
    "min_parameter_delta",
}
_RULE_KEYS = _REQUIRED_RULE_KEYS | _OPTIONAL_RULE_KEYS
_PARAMETER_KEYS = {
    "edit_mode",
    "value_type",
    "safe_bounds",
    "transitions",
    "step",
    "scale",
}


def _mapping(value: Any, location: str) -> Mapping[Any, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be a mapping")
    return value


def _unknown_keys(mapping: Mapping[Any, Any], allowed, location: str) -> None:
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ConfigError(f"{location} has unknown keys: {sorted(unknown, key=str)}")


def _required(mapping: Mapping[Any, Any], keys, location: str) -> None:
    missing = set(keys) - set(mapping)
    if missing:
        raise ConfigError(f"{location} is missing required keys: {sorted(missing)}")


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{location} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ConfigError(f"{location} must be a finite number")
    return numeric


def _positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{location} must be a positive integer")
    return value


def _matches_value_type(value: Any, value_type: str) -> bool:
    if value_type == "float":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    if value_type == "int":
        return not isinstance(value, bool) and isinstance(value, int)
    if value_type == "bool":
        return isinstance(value, bool)
    if value_type == "enum":
        return (
            value is None
            or isinstance(value, (str, int))
            and not isinstance(value, bool)
        )
    return False


def _parameter_spec(value: Any, location: str) -> ParameterSpec:
    mapping = _mapping(value, location)
    _unknown_keys(mapping, _PARAMETER_KEYS, location)
    _required(mapping, {"edit_mode", "value_type"}, location)

    edit_mode = mapping["edit_mode"]
    value_type = mapping["value_type"]
    if edit_mode not in EDIT_MODES:
        raise ConfigError(f"{location}.edit_mode is unsupported: {edit_mode!r}")
    if value_type not in VALUE_TYPES:
        raise ConfigError(f"{location}.value_type is unsupported: {value_type!r}")
    if value_type not in MODE_VALUE_TYPES[edit_mode]:
        raise ConfigError(
            f"{location}.{edit_mode} is incompatible with value_type {value_type!r}"
        )

    safe_bounds = None
    if "safe_bounds" in mapping:
        if value_type not in {"float", "int"}:
            raise ConfigError(f"{location}.safe_bounds require a numeric value_type")
        raw_bounds = mapping["safe_bounds"]
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 2:
            raise ConfigError(f"{location}.safe_bounds must contain two values")
        low = _finite_number(raw_bounds[0], f"{location}.safe_bounds[0]")
        high = _finite_number(raw_bounds[1], f"{location}.safe_bounds[1]")
        if low >= high:
            raise ConfigError(f"{location}.safe_bounds must be strictly increasing")
        if value_type == "int" and not all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw_bounds
        ):
            raise ConfigError(f"{location}.safe_bounds must be integers")
        safe_bounds = (
            (raw_bounds[0], raw_bounds[1]) if value_type == "int" else (low, high)
        )

    transitions_raw = mapping.get("transitions", [])
    if not isinstance(transitions_raw, list):
        raise ConfigError(f"{location}.transitions must be a list")
    transitions = tuple(transitions_raw)
    if transitions and edit_mode not in {"discrete_step", "toggle", "enum"}:
        raise ConfigError(f"{location}.{edit_mode} does not accept transitions")
    if len(set(transitions)) != len(transitions):
        raise ConfigError(f"{location}.transitions must not contain duplicates")
    for transition in transitions:
        if not _matches_value_type(transition, value_type):
            raise ConfigError(
                f"{location}.transition {transition!r} does not match {value_type}"
            )

    step = None
    if "step" in mapping:
        if edit_mode != "discrete_step":
            raise ConfigError(f"{location}.step is valid only for discrete_step")
        step = _finite_number(mapping["step"], f"{location}.step")
        if step <= 0:
            raise ConfigError(f"{location}.step must be positive")
        if value_type == "int" and not float(step).is_integer():
            raise ConfigError(f"{location}.step must be an integer")
        if value_type == "int":
            step = int(step)

    scale = None
    if "scale" in mapping:
        if edit_mode not in {"multiplicative", "safe_span"}:
            raise ConfigError(
                f"{location}.scale is valid only for multiplicative or safe_span"
            )
        scale = _finite_number(mapping["scale"], f"{location}.scale")
        if scale <= 0:
            raise ConfigError(f"{location}.scale must be positive")

    if edit_mode in {"multiplicative", "safe_span"} and safe_bounds is None:
        raise ConfigError(f"{location}.{edit_mode} requires safe_bounds")
    if edit_mode == "discrete_step" and safe_bounds is None and not transitions:
        raise ConfigError(
            f"{location}.discrete_step requires safe_bounds or transitions"
        )
    if edit_mode in {"toggle", "enum"} and len(transitions) < 2:
        raise ConfigError(f"{location}.{edit_mode} requires at least two transitions")

    return ParameterSpec(
        edit_mode=edit_mode,
        value_type=value_type,
        safe_bounds=safe_bounds,
        transitions=transitions,
        step=step,
        scale=scale,
    )


def _rule_config(rule_id: str, value: Any) -> RuleConfig:
    location = f"rules.{rule_id}"
    mapping = _mapping(value, location)
    _unknown_keys(mapping, _RULE_KEYS, location)
    _required(mapping, _REQUIRED_RULE_KEYS, location)

    enabled = mapping["enabled"]
    if not isinstance(enabled, bool):
        raise ConfigError(f"{location}.enabled must be a boolean")
    semantic_group = mapping["semantic_group"]
    if semantic_group not in APPROVED_SEMANTIC_GROUPS:
        raise ConfigError(f"{location}.semantic_group is unsupported")
    structural_level = mapping["structural_level"]
    if (
        isinstance(structural_level, bool)
        or not isinstance(structural_level, int)
        or structural_level not in {0, 1, 2}
    ):
        raise ConfigError(f"{location}.structural_level must be 0, 1, or 2")
    effect_metric = mapping["effect_metric"]
    effect_contract = mapping["effect_contract"]
    if not isinstance(effect_metric, str) or not effect_metric.strip():
        raise ConfigError(f"{location}.effect_metric must be a non-empty string")
    if not isinstance(effect_contract, str) or not effect_contract.strip():
        raise ConfigError(f"{location}.effect_contract must be a non-empty string")
    perceptibility_cm = _finite_number(
        mapping["perceptibility_cm"], f"{location}.perceptibility_cm"
    )
    if perceptibility_cm < 0:
        raise ConfigError(f"{location}.perceptibility_cm must be non-negative")

    metric_threshold_cm = None
    if "metric_threshold_cm" in mapping:
        metric_threshold_cm = _finite_number(
            mapping["metric_threshold_cm"], f"{location}.metric_threshold_cm"
        )
        if metric_threshold_cm < 0:
            raise ConfigError(f"{location}.metric_threshold_cm must be non-negative")

    boundary_threshold_cm = None
    if "boundary_threshold_cm" in mapping:
        boundary_threshold_cm = _finite_number(
            mapping["boundary_threshold_cm"], f"{location}.boundary_threshold_cm"
        )
        if boundary_threshold_cm < 0:
            raise ConfigError(f"{location}.boundary_threshold_cm must be non-negative")

    has_delta_mode = "parameter_delta_mode" in mapping
    has_delta_floor = "min_parameter_delta" in mapping
    if has_delta_mode != has_delta_floor:
        raise ConfigError(
            f"{location}.parameter_delta_mode and "
            "min_parameter_delta must either both be present or both be absent"
        )
    parameter_delta_mode = None
    min_parameter_delta = None
    if has_delta_mode:
        parameter_delta_mode = mapping["parameter_delta_mode"]
        if parameter_delta_mode not in {"absolute", "relative"}:
            raise ConfigError(
                f"{location}.parameter_delta_mode must be 'absolute' or 'relative'"
            )
        min_parameter_delta = _finite_number(
            mapping["min_parameter_delta"], f"{location}.min_parameter_delta"
        )
        if min_parameter_delta < 0:
            raise ConfigError(f"{location}.min_parameter_delta must be non-negative")

    parameter_mapping = _mapping(
        mapping["parameter_specs"], f"{location}.parameter_specs"
    )
    if not parameter_mapping:
        raise ConfigError(f"{location}.parameter_specs must not be empty")
    parameters = {
        name: _parameter_spec(spec, f"{location}.parameter_specs.{name}")
        for name, spec in parameter_mapping.items()
        if isinstance(name, str) and name
    }
    if len(parameters) != len(parameter_mapping):
        raise ConfigError(f"{location}.parameter_specs keys must be non-empty strings")

    return RuleConfig(
        enabled=enabled,
        semantic_group=semantic_group,
        structural_level=structural_level,
        effect_metric=effect_metric,
        effect_contract=effect_contract,
        perceptibility_cm=perceptibility_cm,
        parameter_specs=deepcopy(parameters),
        metric_threshold_cm=metric_threshold_cm,
        boundary_threshold_cm=boundary_threshold_cm,
        parameter_delta_mode=parameter_delta_mode,
        min_parameter_delta=min_parameter_delta,
    )


def load_config(path: Path | str) -> EditConfig:
    """Load and validate an edit-generation YAML config."""

    path = Path(path)
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"failed to load config {path}: {exc}") from exc

    mapping = _mapping(loaded, "config")
    _unknown_keys(mapping, _REQUIRED_TOP_LEVEL_KEYS, "config")
    _required(mapping, _REQUIRED_TOP_LEVEL_KEYS, "config")

    version = mapping["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ConfigError("version must be integer 2")
    expected_source_count = _positive_int(
        mapping["expected_source_count"], "expected_source_count"
    )

    magnitude_mapping = _mapping(mapping["magnitude"], "magnitude")
    if set(magnitude_mapping) != set(APPROVED_MAGNITUDE_TIERS):
        raise ConfigError(
            f"magnitude must contain exactly {list(APPROVED_MAGNITUDE_TIERS)}"
        )
    magnitudes = {}
    for tier in APPROVED_MAGNITUDE_TIERS:
        tier_mapping = _mapping(magnitude_mapping[tier], f"magnitude.{tier}")
        _unknown_keys(tier_mapping, {"strength"}, f"magnitude.{tier}")
        _required(tier_mapping, {"strength"}, f"magnitude.{tier}")
        strength = _finite_number(
            tier_mapping["strength"], f"magnitude.{tier}.strength"
        )
        if strength <= 0:
            raise ConfigError(f"magnitude.{tier}.strength must be positive")
        magnitudes[tier] = MagnitudeTier(strength=strength)
    strengths = [magnitudes[tier].strength for tier in APPROVED_MAGNITUDE_TIERS]
    if not all(left < right for left, right in zip(strengths, strengths[1:])):
        raise ConfigError("magnitude strengths must be strictly increasing")

    raw_attempt_order = mapping["magnitude_attempt_order"]
    if not isinstance(raw_attempt_order, list):
        raise ConfigError("magnitude_attempt_order must be a list")
    if not raw_attempt_order:
        raise ConfigError("magnitude_attempt_order must not be empty")
    if any(
        not isinstance(tier, str) or tier not in APPROVED_MAGNITUDE_TIERS
        for tier in raw_attempt_order
    ):
        raise ConfigError(
            "magnitude_attempt_order contains an unsupported magnitude tier"
        )
    if len(raw_attempt_order) != len(set(raw_attempt_order)):
        raise ConfigError("magnitude_attempt_order must not contain duplicates")
    if "slightly" in raw_attempt_order:
        raise ConfigError("magnitude_attempt_order must not contain slightly")
    magnitude_attempt_order = tuple(raw_attempt_order)

    rule_mapping = _mapping(mapping["rules"], "rules")
    if not rule_mapping:
        raise ConfigError("rules must not be empty")
    rules = {}
    for rule_id, rule in rule_mapping.items():
        if not isinstance(rule_id, str) or not rule_id:
            raise ConfigError("rule IDs must be non-empty strings")
        rules[rule_id] = _rule_config(rule_id, rule)

    return EditConfig(
        version=version,
        expected_source_count=expected_source_count,
        magnitude=deepcopy(magnitudes),
        rules=deepcopy(rules),
        magnitude_attempt_order=magnitude_attempt_order,
    )


def effective_config_payload(config: EditConfig) -> dict[str, Any]:
    """Return the complete JSON-like semantic payload used for run identity."""

    if not isinstance(config, EditConfig):
        raise TypeError("config must be an EditConfig")

    def parameter_payload(spec: ParameterSpec) -> dict[str, Any]:
        return {
            "edit_mode": spec.edit_mode,
            "value_type": spec.value_type,
            "safe_bounds": spec.safe_bounds,
            "transitions": spec.transitions,
            "step": spec.step,
            "scale": spec.scale,
        }

    def rule_payload(rule: RuleConfig) -> dict[str, Any]:
        payload = {
            "enabled": rule.enabled,
            "semantic_group": rule.semantic_group,
            "structural_level": rule.structural_level,
            "effect_metric": rule.effect_metric,
            "effect_contract": rule.effect_contract,
            "perceptibility_cm": rule.perceptibility_cm,
            "parameter_specs": {
                name: parameter_payload(rule.parameter_specs[name])
                for name in sorted(rule.parameter_specs)
            },
        }
        if rule.metric_threshold_cm is not None:
            payload["metric_threshold_cm"] = rule.metric_threshold_cm
        if rule.boundary_threshold_cm is not None:
            payload["boundary_threshold_cm"] = rule.boundary_threshold_cm
        if rule.parameter_delta_mode is not None:
            payload["parameter_delta_mode"] = rule.parameter_delta_mode
        if rule.min_parameter_delta is not None:
            payload["min_parameter_delta"] = rule.min_parameter_delta
        return payload

    payload = {
        "version": config.version,
        "expected_source_count": config.expected_source_count,
        "magnitude": {
            name: {"strength": config.magnitude[name].strength}
            for name in sorted(config.magnitude)
        },
        "magnitude_attempt_order": config.magnitude_attempt_order,
        "rules": {
            rule_id: rule_payload(config.rules[rule_id])
            for rule_id in sorted(config.rules)
        },
    }
    return payload


def config_digest(config: EditConfig) -> str:
    """Hash every effective config value through canonical JSON."""

    from .canonical_io import canonical_hash

    return canonical_hash(effective_config_payload(config))
