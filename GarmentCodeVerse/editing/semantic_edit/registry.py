"""Semantic-rule registry and configuration boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from editing.config import ConfigError, EditConfig, load_config
from editing.models import Rule
from editing.semantic_edit.rules.length import build_rules as build_length_rule_set

RULE_ORDER = (
    "shirt_fit", "shirt_length", "neckline_type", "hood_presence",
    "collar_presence", "sleeve_presence", "sleeve_length", "skirt_length",
    "godet_insert_count", "pants_length",
)


def load_semantic_config(path: Path | str) -> EditConfig:
    """Load the only supported semantic-edit configuration."""

    config = load_config(path)
    _validate_semantic_config(config)
    return config


def _validate_semantic_config(config: EditConfig) -> None:
    if not isinstance(config, EditConfig):
        raise TypeError("config must be an EditConfig")
    if config.expected_source_count != 100:
        raise ConfigError("semantic edit expected_source_count must be exactly 100")
    if tuple(rule_id for rule_id in RULE_ORDER if rule_id in config.rules) != (
        RULE_ORDER
    ) or set(config.rules) != set(RULE_ORDER):
        raise ConfigError(
            "semantic edit rules must contain exactly the canonical ten-rule set"
        )
    disabled = tuple(
        rule_id for rule_id in RULE_ORDER if not config.rules[rule_id].enabled
    )
    if disabled:
        raise ConfigError(
            f"semantic edit rules must all be enabled: {list(disabled)}"
        )
    order = config.magnitude_attempt_order
    if (
        order is None
        or not isinstance(order, tuple)
        or not order
        or len(set(order)) != len(order)
        or any(tier not in config.magnitude for tier in order)
    ):
        raise ConfigError(
            "semantic edit magnitude_attempt_order must be a non-empty "
            "unique tuple of configured tiers"
        )


def build_semantic_registry(config: EditConfig) -> Mapping[str, Rule]:
    """Build the ten complete callable rules in canonical order."""

    _validate_semantic_config(config)

    from .rules.collar_hood import build_rule as build_collar_hood_rule
    from .rules.fit import build_rules as build_fit_rules
    from .rules.godet import build_rule as build_godet_rule
    from .rules.neckline import build_rule as build_neckline_rule
    from .rules.sleeve import build_rules as build_sleeve_rules

    try:
        length_rule_map = {
            rule.id: rule for rule in build_length_rule_set(config)
        }
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"length rule construction failed: {exc}") from exc

    factories = (
        ("shirt_fit", lambda: build_fit_rules(config)[0]),
        (
            "shirt_length",
            lambda: length_rule_map["shirt_length"],
        ),
        (
            "neckline_type",
            lambda: build_neckline_rule(config),
        ),
        (
            "hood_presence",
            lambda: build_collar_hood_rule(config, "hood_presence"),
        ),
        (
            "collar_presence",
            lambda: build_collar_hood_rule(config, "collar_presence"),
        ),
        (
            "sleeve_presence",
            lambda: build_sleeve_rules(config)[0],
        ),
        (
            "sleeve_length",
            lambda: length_rule_map["sleeve_length"],
        ),
        (
            "skirt_length",
            lambda: length_rule_map["skirt_length"],
        ),
        (
            "godet_insert_count",
            lambda: build_godet_rule(config),
        ),
        (
            "pants_length",
            lambda: length_rule_map["pants_length"],
        ),
    )
    if tuple(rule_id for rule_id, _factory in factories) != RULE_ORDER:
        raise AssertionError("semantic rule factories must use canonical order")

    ordered: dict[str, Rule] = {}
    for rule_id, factory in factories:
        try:
            implementation = factory()
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{rule_id} rule construction failed: {exc}"
            ) from exc
        if implementation.id != rule_id:
            raise ConfigError(
                "semantic rule key/id mismatch: "
                f"{rule_id!r} != {implementation.id!r}"
            )
        rule = implementation
        if rule_id in ordered:  # pragma: no cover - invariant.
            raise ConfigError(f"duplicate semantic rule id: {rule_id}")
        ordered[rule_id] = rule
    if tuple(ordered) != RULE_ORDER:  # pragma: no cover - loop invariant.
        raise ValueError("semantic registry rules must use the canonical order")
    return dict(ordered)
