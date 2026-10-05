"""Self-contained ordinary-shirt fit rule."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import math
from typing import Any

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
from editing.pattern_geometry import (
    PatternGeometryError,
    edge_topology,
    specification_metadata,
    stitch_records,
)

from ..prompt import PromptError


FIT_WIDTH_PATH = "design.shirt.width.v"
FIT_READ_PATHS = (
    "design.meta.upper.v",
    "design.left.enable_asym.v",
    "design.shirt.width.v",
    "design.shirt.flare.v",
    "design.shirt.length.v",
    "design.sleeve.sleeveless.v",
)
CLOSE_FIT_TARGET_WIDTH = 0.84
LOOSE_WIDTH_MARGIN = 0.04

_FIT_CONFLICTS = ("shirt_length",)
_FIT_PREREQUISITES = (
    "named_fit_agreement",
    "ordinary_symmetric_shirt",
)
_NAMED_STATES = frozenset({"close-fitting", "loose"})
_TORSO_PANELS = (
    "left_ftorso",
    "right_ftorso",
    "left_btorso",
    "right_btorso",
)
class FitMeasurementError(ValueError):
    """A constructed specification cannot support the fit metric."""


def _unavailable(code: str, message: str) -> StateUnavailableError:
    return StateUnavailableError(code, message)


def _state_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings")
    return value


def _state_design_value(design: Mapping[str, Any], path: str) -> Any:
    _state_mapping(design, "design")
    try:
        return design_dotted_get(design, path)
    except (KeyError, TypeError, ValueError) as exc:
        raise _unavailable(
            "state_schema_unavailable",
            f"active state path is missing or invalid: {path}",
        ) from exc


def _state_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise _unavailable(
            "state_schema_unavailable",
            f"{field_name} must be numeric",
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _unavailable(
            "state_schema_unavailable",
            f"{field_name} must be numeric",
        ) from exc
    if not math.isfinite(result):
        raise _unavailable(
            "state_schema_unavailable",
            f"{field_name} must be finite",
        )
    return result


def _body_number(body: Mapping[str, Any], key: str) -> float:
    source = _state_mapping(body, "body")
    if key not in source:
        raise _unavailable(
            "body_measurement_unavailable",
            f"missing required body measurement: {key}",
        )
    return _state_number(source[key], f"body.{key}")


def classify_design(
    design: Mapping[str, Any],
    body: Mapping[str, Any],
) -> SemanticState:
    """Classify fit directly from one design/body pair."""

    width = _state_number(
        _state_design_value(design, "design.shirt.width.v"),
        "design.shirt.width.v",
    )
    flare = _state_number(
        _state_design_value(design, "design.shirt.flare.v"),
        "design.shirt.flare.v",
    )
    length = _state_number(
        _state_design_value(design, "design.shirt.length.v"),
        "design.shirt.length.v",
    )
    bust = _body_number(body, "bust")
    waist = _body_number(body, "waist")
    hips = _body_number(body, "hips")
    waist_line = _body_number(body, "waist_line")
    hips_line = _body_number(body, "hips_line")
    if hips_line == 0.0:
        raise _unavailable(
            "body_measurement_unavailable",
            "body.hips_line must be non-zero",
        )

    garment_bust = width * bust
    garment_hem = flare * garment_bust
    below_waist = max(0.0, length * waist_line - waist_line)
    hip_progress = min(1.0, below_waist / hips_line)
    body_at_hem = waist + hip_progress * (hips - waist)
    if body_at_hem == 0.0:
        raise _unavailable(
            "body_measurement_unavailable",
            "body circumference at the garment hem must be non-zero",
        )
    hem_ease = garment_hem / body_at_hem

    if length < 1.0:
        state_id = (
            "close-fitting"
            if width < 0.88 and flare < 1.20
            else "unnamed"
        )
    elif flare >= 1.45:
        state_id = "loose"
    elif width < 0.88 and flare < 1.25:
        state_id = "close-fitting"
    elif width > 1.15 and hem_ease > 1.10:
        state_id = "loose"
    else:
        state_id = "unnamed"
    return SemanticState(
        rule_id="shirt_fit",
        state_id=state_id,
        phrase=state_id,
        attributes={"hem_ease": hem_ease},
    )


def classify(context: RuleContext) -> SemanticState:
    """Classify fit from the shared rule context."""

    return classify_design(context.design, context.body)


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _same_value(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _prompt_fit_agrees(local: SemanticState, prompt: Any) -> bool:
    """Match semantic fit identity while checking the raw prompt-state shape."""

    return (
        isinstance(prompt, SemanticState)
        and prompt.rule_id == "shirt_fit"
        and prompt.state_id in _NAMED_STATES
        and prompt.phrase == prompt.state_id
        and prompt.attributes.get("protected_phrase_present") is True
        and local.rule_id == "shirt_fit"
        and local.state_id == prompt.state_id
    )


def _cold_shoulder(design: Mapping[str, Any]) -> bool:
    return (
        design_dotted_get(design, "design.meta.upper.v") == "Shirt"
        and design_dotted_get(design, "design.left.enable_asym.v") is False
        and design_dotted_get(design, "design.sleeve.sleeveless.v") is True
        and _number(
            design_dotted_get(design, "design.shirt.width.v"),
            "design.shirt.width.v",
        )
        >= 1.20
        and _number(
            design_dotted_get(design, "design.shirt.flare.v"),
            "design.shirt.flare.v",
        )
        >= 1.45
    )


def _body_at_hem(
    *,
    length: float,
    body: Mapping[str, Any],
) -> tuple[float, float]:
    bust = _number(body["bust"], "body.bust")
    waist = _number(body["waist"], "body.waist")
    hips = _number(body["hips"], "body.hips")
    waist_line = _number(body["waist_line"], "body.waist_line")
    hips_line = _number(body["hips_line"], "body.hips_line")
    measurements = {
        "body.bust": bust,
        "body.waist": waist,
        "body.hips": hips,
        "body.waist_line": waist_line,
        "body.hips_line": hips_line,
    }
    nonpositive = tuple(name for name, value in measurements.items() if value <= 0.0)
    if nonpositive:
        raise ValueError(
            "fit body measurements must be physically positive: "
            f"{list(nonpositive)!r}"
        )
    below_waist = max(0.0, length * waist_line - waist_line)
    hip_progress = min(1.0, below_waist / hips_line)
    result = waist + hip_progress * (hips - waist)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            "body circumference at the shirt hem must be finite and positive"
        )
    return bust, result


def _validate_rule_config(config: RuleConfig) -> None:
    if not isinstance(config, RuleConfig):
        raise TypeError("rule_config must be a RuleConfig")
    if type(config.enabled) is not bool or config.enabled is not True:
        raise ValueError("shirt_fit rule_config must be enabled")
    if (
        config.semantic_group != "upper"
        or type(config.structural_level) is not int
        or config.structural_level != 0
    ):
        raise ValueError("shirt_fit rule_config has an invalid semantic contract")
    if (
        config.effect_metric != "full_torso_width_cm"
        or config.effect_contract != "topology_preserving"
    ):
        raise ValueError("shirt_fit rule_config has an invalid effect contract")
    if set(config.parameter_specs) != {"width"}:
        raise ValueError("shirt_fit rule_config must contain only width")
    spec = config.parameter_specs["width"]
    if (
        spec.edit_mode != "safe_span"
        or spec.value_type != "float"
        or spec.safe_bounds != (0.8, 1.3)
    ):
        raise ValueError("shirt_fit width config must use safe_span [0.8, 1.3]")
    threshold = config.effective_metric_threshold_cm
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
    ):
        raise ValueError("shirt_fit metric threshold must be finite and non-negative")


def _candidate(
    context: RuleContext,
    rule_config: RuleConfig,
) -> PlannedOperation | None:
    if context.active_paths.design_hash != canonical_hash(context.design):
        return None
    if context.active_paths.dispatch.get("upper") != "Shirt":
        return None
    if context.active_paths.dispatch.get("asymmetric") is not False:
        return None
    if FIT_WIDTH_PATH not in context.active_paths.paths:
        return None
    if not set(FIT_READ_PATHS).issubset(context.active_paths.paths):
        return None

    read_values = {
        path: design_dotted_get(context.design, path) for path in FIT_READ_PATHS
    }
    if read_values["design.meta.upper.v"] != "Shirt":
        return None
    if read_values["design.left.enable_asym.v"] is not False:
        return None
    if not isinstance(read_values["design.sleeve.sleeveless.v"], bool):
        return None
    source_width = _number(read_values[FIT_WIDTH_PATH], FIT_WIDTH_PATH)
    flare = _number(
        read_values["design.shirt.flare.v"],
        "design.shirt.flare.v",
    )
    length = _number(
        read_values["design.shirt.length.v"],
        "design.shirt.length.v",
    )
    lower, upper = rule_config.parameter_specs["width"].safe_bounds  # type: ignore[misc]
    if source_width < lower or source_width > upper:
        return None
    if length < 1.0 or flare <= 0.0 or flare >= 1.45:
        return None
    if _cold_shoulder(context.design):
        return None

    source_local = classify(context)
    if source_local.state_id not in _NAMED_STATES:
        return None
    source_protected = context.prompt_states.get("shirt_fit")
    if not _prompt_fit_agrees(source_local, source_protected):
        return None

    bust, body_at_hem = _body_at_hem(length=length, body=context.body)
    if source_local.state_id == "close-fitting":
        required_denominator = flare * bust
        if not math.isfinite(required_denominator) or required_denominator <= 0.0:
            return None
        required = 1.10 * body_at_hem / required_denominator
        if not math.isfinite(required):
            return None
        raw_target = max(1.20, required + LOOSE_WIDTH_MARGIN)
        if not math.isfinite(raw_target):
            return None
        target_width = min(upper, raw_target)
        direction = "increase"
        expected_target_state = "loose"
        clamped = target_width != raw_target
    else:
        target_width = CLOSE_FIT_TARGET_WIDTH
        direction = "decrease"
        expected_target_state = "close-fitting"
        clamped = False
    if target_width < lower or target_width > upper:
        return None
    if _same_value(source_width, target_width):
        return None

    target_design = plain_copy(
        design_dotted_set(
            context.design,
            FIT_WIDTH_PATH,
            target_width,
        ),
        tuples_as_lists=True,
    )
    if _cold_shoulder(target_design):
        return None
    target_local = classify_design(target_design, context.body)
    if target_local.state_id != expected_target_state:
        return None
    target_protected_states = context.classify_prompt(
        target_design,
        context.raw_body_yaml,
    )
    if not isinstance(target_protected_states, Mapping):
        raise TypeError("protected fit oracle result must be a mapping")
    target_protected = target_protected_states.get("shirt_fit")
    if target_protected is None:
        return None
    if not isinstance(target_protected, SemanticState):
        raise TypeError("protected fit oracle value must be a SemanticState")
    if not _prompt_fit_agrees(target_local, target_protected):
        return None

    metric = abs(target_width - source_width) * bust
    if not math.isfinite(metric):
        return None
    threshold = float(rule_config.effective_metric_threshold_cm)
    if metric < threshold:
        return None
    relative_change = abs(target_width - source_width) / (upper - lower)
    if not math.isfinite(relative_change):
        return None
    reads = tuple(sorted(FIT_READ_PATHS))
    writes = (FIT_WIDTH_PATH,)
    transition = SemanticTransition(
        rule_id="shirt_fit",
        scope="garment",
        channels=("garment",),
        feature_phrase="the shirt",
        source_state=source_local,
        target_state=target_local,
        direction=direction,
    )
    operation = Operation(
        rule_id="shirt_fit",
        semantic_group=rule_config.semantic_group,
        scope="garment",
        direction=direction,
        magnitude="semantic",
        structural_level=rule_config.structural_level,
        reads=reads,
        writes=writes,
        conflicts=tuple(sorted(_FIT_CONFLICTS)),
        prerequisites=tuple(sorted(_FIT_PREREQUISITES)),
        edit_mode=rule_config.parameter_specs["width"].edit_mode,
        effect_metric_id=rule_config.effect_metric,
        effect_contract_id=rule_config.effect_contract,
        before={FIT_WIDTH_PATH: read_values[FIT_WIDTH_PATH]},
        after={FIT_WIDTH_PATH: target_width},
        relative_change=float(relative_change),
        requested_magnitude="semantic",
        realized_magnitude="semantic",
        clamped=clamped,
        instruction_tokens={
            "feature": "the shirt",
            "source_state_id": source_local.state_id,
            "target_state_id": target_local.state_id,
        },
    )
    return PlannedOperation(
        operation=operation,
        transitions=(transition,),
        topology_kind=TopologyKind.NONE,
        applicability_evidence={
            "rule_id": "shirt_fit",
            "source_design_hash": canonical_hash(context.design),
            "target_design_hash": canonical_hash(target_design),
            "body_hash": canonical_hash(context.body),
            "active_paths_design_hash": context.active_paths.design_hash,
            "read_contract_hash": canonical_hash(read_values),
            "source_local_state": source_local.to_payload(),
            "source_protected_state": source_protected.to_payload(),
            "source_state_match": True,
            "target_local_state": target_local.to_payload(),
            "target_protected_state": target_protected.to_payload(),
            "target_state_match": True,
            "metric_id": rule_config.effect_metric,
            "metric_value_cm": metric,
            "metric_threshold_cm": threshold,
        },
        conflict_evidence=empty_conflict_evidence(
            "shirt_fit",
            _FIT_CONFLICTS,
        ),
    )


def _candidates(
    context: RuleContext,
    rule_config: RuleConfig,
) -> tuple[PlannedOperation, ...]:
    try:
        candidate = _candidate(context, rule_config)
    except (
        ProtectedStateSchemaError,
        PromptError,
        TypeError,
    ):
        raise
    except (
        KeyError,
        ValueError,
        StateUnavailableError,
    ):
        return ()
    return () if candidate is None else (candidate,)


def apply(
    design: dict[str, Any],
    operation: PlannedOperation,
) -> dict[str, Any]:
    """Apply one planned fit width mutation."""

    return plain_copy(
        design_dotted_set(
            design,
            FIT_WIDTH_PATH,
            operation.operation.after[FIT_WIDTH_PATH],
        ),
        tuples_as_lists=True,
    )


def _effect_number(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if positive and nonnegative:
        raise ValueError("positive and nonnegative are mutually exclusive")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FitMeasurementError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise FitMeasurementError(f"{field_name} must be finite") from exc
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
        raise FitMeasurementError(f"{field_name} must be{suffix}")
    return result


def fit_width_metric_cm(
    source_width: Any,
    target_width: Any,
    *,
    bust_cm: Any,
) -> float:
    """Measure the full-torso width delta in centimetres."""

    source = _effect_number(source_width, "source_width")
    target = _effect_number(target_width, "target_width")
    bust = _effect_number(bust_cm, "bust_cm", positive=True)
    try:
        metric = float(
            abs(Decimal(str(target)) - Decimal(str(source)))
            * Decimal(str(bust))
        )
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise FitMeasurementError(
            "full torso width metric must be finite"
        ) from exc
    if not math.isfinite(metric):
        raise FitMeasurementError("full torso width metric must be finite")
    return metric


def _transition_named_fit(planned: PlannedOperation) -> bool:
    operation = planned.operation
    if len(planned.transitions) != 1:
        return False
    transition = planned.transitions[0]
    pair = (
        transition.source_state.state_id,
        transition.target_state.state_id,
        transition.direction,
    )
    expected = {
        ("close-fitting", "loose", "increase"),
        ("loose", "close-fitting", "decrease"),
    }
    return (
        pair in expected
        and transition.rule_id == "shirt_fit"
        and transition.scope == operation.scope == "garment"
        and transition.channels == ("garment",)
        and transition.direction == operation.direction
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FitMeasurementError(f"{field_name} must be a mapping")
    return value


def _panels(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return _mapping(
            specification["pattern"]["panels"],
            "pattern.panels",
        )
    except KeyError as exc:
        raise FitMeasurementError("pattern.panels is required") from exc


def _stitch_counter(
    specification: Mapping[str, Any],
) -> Counter[Any]:
    try:
        return Counter(stitch_records(specification))
    except PatternGeometryError as exc:
        raise FitMeasurementError(str(exc)) from exc


def _spec_metadata(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return specification_metadata(specification)
    except PatternGeometryError as exc:
        raise FitMeasurementError(str(exc)) from exc


def _fit_edge_topology(panel: Mapping[str, Any]) -> tuple[Any, ...]:
    try:
        return edge_topology(panel)
    except PatternGeometryError as exc:
        raise FitMeasurementError(str(exc)) from exc


def _torso_topology_unchanged(
    source_specification: Mapping[str, Any],
    target_specification: Mapping[str, Any],
) -> tuple[bool, Mapping[str, Any]]:
    source_panels = _panels(source_specification)
    target_panels = _panels(target_specification)
    source_names = set(source_panels)
    target_names = set(target_panels)
    if source_names != target_names:
        return False, {
            "source_panel_names": sorted(source_names),
            "target_panel_names": sorted(target_names),
        }
    if not set(_TORSO_PANELS).issubset(source_names):
        return False, {
            "missing_torso_panels": sorted(set(_TORSO_PANELS) - source_names)
        }
    if canonical_json_bytes(_spec_metadata(source_specification)) != (
        canonical_json_bytes(_spec_metadata(target_specification))
    ):
        return False, {"metadata_match": False}
    if _stitch_counter(source_specification) != _stitch_counter(target_specification):
        return False, {"canonical_stitch_multiset_match": False}
    mismatches = []
    for name in _TORSO_PANELS:
        source_panel = _mapping(source_panels[name], name)
        target_panel = _mapping(target_panels[name], name)
        if _fit_edge_topology(source_panel) != _fit_edge_topology(target_panel):
            mismatches.append(name)
    return (
        not mismatches,
        {
            "complete_panel_membership_match": True,
            "canonical_stitch_multiset_match": True,
            "torso_topology_mismatches": mismatches,
        },
    )


def _configured_threshold(config: EditConfig) -> float:
    try:
        value = config.rules["shirt_fit"].effective_metric_threshold_cm
    except (KeyError, AttributeError) as exc:
        raise FitMeasurementError(
            "missing configured threshold "
            "shirt_fit.effective_metric_threshold_cm"
        ) from exc
    return _effect_number(
        value,
        "config.rules.shirt_fit.effective_metric_threshold_cm",
        nonnegative=True,
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
    """Validate the named fit transition, metric, topology, and evidence."""

    if topology_policy not in {"strict", "local"}:
        raise ValueError(
            "topology_policy must be exactly 'strict' or 'local'"
        )
    if not _transition_named_fit(planned_operation):
        return rule_effect_result(
            "shirt_fit",
            passed=False,
            code="effect/state_mismatch",
            details={"named_fit_transition": False},
        )
    try:
        source_width = design_dotted_get(source_design, FIT_WIDTH_PATH)
        target_width = design_dotted_get(target_design, FIT_WIDTH_PATH)
        metric = fit_width_metric_cm(
            source_width,
            target_width,
            bust_cm=body["bust"],
        )
        source_width_number = _effect_number(source_width, "source_width")
        target_width_number = _effect_number(target_width, "target_width")
        actual_direction_ok = (
            target_width_number > source_width_number
            if planned_operation.operation.direction == "increase"
            else target_width_number < source_width_number
        )
        topology_ok, topology_details = _torso_topology_unchanged(
            source_specification,
            target_specification,
        )
        threshold = _configured_threshold(config)
    except (
        KeyError,
        TypeError,
        ValueError,
        ArithmeticError,
        FitMeasurementError,
    ) as exc:
        return rule_effect_result(
            "shirt_fit",
            passed=False,
            code="effect/measurement_error",
            details={"message": str(exc)},
        )
    details = {
        "full_torso_width_cm": metric,
        "threshold_cm": threshold,
        "torso_topology_unchanged": topology_ok,
        "topology": topology_details,
        "topology_policy": topology_policy,
        "actual_direction_ok": actual_direction_ok,
    }
    if topology_policy == "strict" and not topology_ok:
        return rule_effect_result(
            "shirt_fit",
            passed=False,
            code="effect/topology_mismatch",
            details=details,
        )
    if topology_policy == "local" and not actual_direction_ok:
        return rule_effect_result(
            "shirt_fit",
            passed=False,
            code="effect/direction_mismatch",
            details=details,
        )
    passed = metric >= threshold
    return rule_effect_result(
        "shirt_fit",
        passed=passed,
        code="effect/passed" if passed else "effect/metric_threshold",
        details=details,
    )


def build_rules(config: EditConfig) -> tuple[Rule, ...]:
    """Bind the fit candidate callable to one immutable loaded config."""

    if not isinstance(config, EditConfig):
        raise TypeError("config must be an EditConfig")
    try:
        rule_config = config.rules["shirt_fit"]
    except KeyError as exc:
        raise ValueError("config must contain shirt_fit") from exc
    _validate_rule_config(rule_config)

    def candidates(context: RuleContext) -> tuple[PlannedOperation, ...]:
        return _candidates(context, rule_config)

    return (
        Rule(
            id="shirt_fit",
            candidates=candidates,
            apply=apply,
            validate=validate,
        ),
    )
