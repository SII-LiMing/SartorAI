"""Self-contained collar- and hood-presence rules."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable, NamedTuple

from editing.canonical_io import (
    canonical_hash,
    canonical_json_bytes,
    design_dotted_get,
    plain_copy,
)
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
)
from editing.config import ConfigError, EditConfig, ParameterSpec, RuleConfig
from editing.pattern_compare import (
    compare_patterns,
    edge_type,
    semantic_pattern_hash,
    topology_signature,
)
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

BACK_NECKLINE_PATH = "design.collar.b_collar.v"

HOOD_LENGTH_PATH = "design.collar.component.hood_length.v"
HOOD_ADD_MAX_HOOD_LENGTH = 1.25
HOOD_ADD_GUARDRAIL_ID = "hood_presence/add/hood_length_max/v1"
HOOD_ADD_REJECTION_REASON = "latent_hood_length_exceeds_pilot_safe_max"
HOOD_ADD_ACCEPT_REASON = "latent_hood_length_within_pilot_safe_max"
HOOD_REMOVE_GUARDRAIL_REASON = "remove_direction_not_restricted"

# Operation identity tuples are canonical-sorted at their definition site.
# The read sets deliberately include dormant constructor parameters so that a
# probe and a later generation run cannot acquire different contracts merely
# because a different enum endpoint activates one of them.
COLLAR_APPLICABILITY_READS = tuple(
    sorted(
        (
            "design.meta.upper.v",
            "design.left.enable_asym.v",
            "design.shirt.strapless.v",
            COMPONENT_STYLE_PATH,
            FRONT_NECKLINE_PATH,
            BACK_NECKLINE_PATH,
        )
    )
)

HOOD_PRESENCE_READS = tuple(
    sorted(
        COLLAR_APPLICABILITY_READS
        + (
            "design.collar.width.v",
            "design.collar.fc_depth.v",
            "design.collar.bc_depth.v",
            "design.collar.component.hood_depth.v",
            HOOD_LENGTH_PATH,
        )
    )
)

COLLAR_PRESENCE_READS = tuple(
    sorted(
        COLLAR_APPLICABILITY_READS
        + (
            "design.collar.width.v",
            "design.collar.fc_depth.v",
            "design.collar.bc_depth.v",
            "design.collar.fc_angle.v",
            "design.collar.f_flip_curve.v",
            "design.collar.component.depth.v",
            "design.collar.component.lapel_standing.v",
        )
    )
)

_WRITE_COMPONENT = (COMPONENT_STYLE_PATH,)
_ENUM_PARAMETER_TYPES = {
    COMPONENT_STYLE_PATH: "select_null",
    FRONT_NECKLINE_PATH: "select",
    BACK_NECKLINE_PATH: "select",
}
_LEGITIMATE_ABSENCE_CODES = {
    "hood_presence": frozenset({"hood_presence_unrepresented"}),
    "collar_presence": frozenset({"collar_presence_unrepresented"}),
}

COLLAR_PANEL = re.compile(r"^(left|right)_collar_(front|back)$")
_HOOD_PANEL = re.compile(r"^(?:left|right)_hood$")
_TORSO_PANEL = re.compile(r"^(?:left|right)_[fb]torso$")
_PATTERN_ATOL = 1.0e-5


def _validate_rule_config(
    config: RuleConfig,
    *,
    rule_id: str,
    parameter_name: str,
    semantic_group: str,
    structural_level: int,
    effect_metric: str,
    effect_contract: str,
    transitions: tuple[Any, ...],
) -> ParameterSpec:
    """Reject config drift from one structural rule's frozen contract."""

    if type(config) is not RuleConfig:
        raise ConfigError(f"{rule_id} rule_config must be an exact RuleConfig")
    if type(config.enabled) is not bool or config.enabled is not True:
        raise ConfigError(f"{rule_id} rule_config must be enabled")
    if (
        config.semantic_group != semantic_group
        or type(config.structural_level) is not int
        or config.structural_level != structural_level
        or config.effect_metric != effect_metric
        or config.effect_contract != effect_contract
    ):
        raise ConfigError(f"{rule_id} rule_config has an invalid operation contract")
    if set(config.parameter_specs) != {parameter_name}:
        raise ConfigError(
            f"{rule_id} rule_config must contain only {parameter_name}"
        )
    spec = config.parameter_specs[parameter_name]
    if (
        type(spec) is not ParameterSpec
        or spec.edit_mode != "enum"
        or spec.value_type != "enum"
        or spec.transitions != transitions
    ):
        raise ConfigError(f"{rule_id} {parameter_name} config has invalid transitions")
    return spec


def _is_legitimate_unrepresented(
    error: StateUnavailableError,
    *,
    rule_id: str,
) -> bool:
    """Distinguish semantic absence from malformed state input.

    ``StateUnavailableError`` also carries hard schema codes such as
    ``state_schema_unavailable``.  Those must propagate rather than silently
    shrinking source capability.
    """

    return error.code in _LEGITIMATE_ABSENCE_CODES[rule_id]


def _unrepresented_or_raise(
    error: StateUnavailableError,
    *,
    rule_id: str,
) -> None:
    if not _is_legitimate_unrepresented(error, rule_id=rule_id):
        raise error


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


def _enum_metadata_is_exact(
    design: Mapping[str, Any],
) -> bool:
    for path, expected_type in _ENUM_PARAMETER_TYPES.items():
        parameter = _parameter_node(design, path)
        if parameter is None or "v" not in parameter:
            return False
        schema = parameter.get("type")
        if type(schema) is not str or schema != expected_type:
            return False
    return True


def _prompt_hood_agrees(local: SemanticState, prompt: Any) -> bool:
    present = local.state_id == "hooded"
    return (
        isinstance(prompt, SemanticState)
        and prompt.rule_id == "hood_presence"
        and prompt.state_id == local.state_id
        and prompt.phrase == local.phrase
        and prompt.attributes.get("present") is present
        and (
            prompt.attributes.get("protected_phrase") == "hood"
            if present
            else "protected_phrase" not in prompt.attributes
        )
    )


def _prompt_collar_agrees(local: SemanticState, prompt: Any) -> bool:
    return (
        isinstance(prompt, SemanticState)
        and prompt.rule_id == "collar_presence"
        and prompt.state_id == "collared" == local.state_id
        and prompt.phrase == local.phrase
        and prompt.attributes.get("present") is True
        and prompt.attributes.get("component_style") == "SimpleLapel"
        and prompt.attributes.get("subtype_phrase")
        == local.attributes.get("subtype_phrase")
    )


def _state_unavailable(code: str, message: str) -> StateUnavailableError:
    return StateUnavailableError(code, message)


def _active_collar_paths(design: Mapping[str, Any]) -> Any:
    try:
        return resolve_active_paths(design)
    except (ActivePathError, KeyError, TypeError, ValueError) as exc:
        raise _state_unavailable(
            "active_path_unavailable",
            f"cannot resolve active constructor paths: {exc}",
        ) from exc


def _component_style(design: Mapping[str, Any]) -> str | None:
    try:
        value = design_dotted_get(design, COMPONENT_STYLE_PATH)
    except (KeyError, TypeError, ValueError) as exc:
        raise _state_unavailable(
            "state_schema_unavailable",
            f"active state path is missing or invalid: {COMPONENT_STYLE_PATH}",
        ) from exc
    if value is None:
        return None
    if type(value) is not str or value not in {
        "SimpleLapel",
        "Hood2Panels",
        "Turtle",
    }:
        raise _state_unavailable(
            "collar_component_style_schema_unavailable",
            "collar component style schema requires exactly None, "
            "SimpleLapel, Hood2Panels, or Turtle",
        )
    return value




def _visible_symmetric_upper(design: Mapping[str, Any]) -> None:
    active = _active_collar_paths(design)
    if active.dispatch.get("upper") is None:
        raise _state_unavailable(
            "neckline_inapplicable",
            "neckline state requires an active upper garment",
        )
    if active.dispatch.get("asymmetric") is True:
        raise _state_unavailable(
            "asymmetric_neckline_unrepresented",
            "asymmetric neckline is generic and its type is unrepresented",
        )
    if active.dispatch.get("upper") == "FittedShirt":
        try:
            strapless = design_dotted_get(
                design, "design.shirt.strapless.v"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _state_unavailable(
                "state_schema_unavailable",
                "active state path is missing or invalid: "
                "design.shirt.strapless.v",
            ) from exc
        if bool(strapless):
            raise _state_unavailable(
                "strapless_neckline_unrepresented",
                "strapless upper edge is not a neckline-type state",
            )


def classify_hood_state(design: Mapping[str, Any]) -> SemanticState:
    """Classify constructor-active hood presence."""

    active = _active_collar_paths(design)
    if active.dispatch.get("upper") is None:
        raise _state_unavailable(
            "hood_presence_inapplicable",
            "hood presence requires an active upper garment",
        )
    if active.dispatch.get("asymmetric") is True:
        raise _state_unavailable(
            "asymmetric_hood_path_suppressed",
            "asymmetric construction suppresses the shared hood component",
        )
    if active.dispatch.get("upper") == "FittedShirt":
        try:
            strapless = design_dotted_get(
                design, "design.shirt.strapless.v"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _state_unavailable(
                "state_schema_unavailable",
                "active state path is missing or invalid: "
                "design.shirt.strapless.v",
            ) from exc
        if bool(strapless):
            raise _state_unavailable(
                "strapless_hood_path_suppressed",
                "strapless construction suppresses the hood component",
            )
    if active.dispatch.get("collar_component_suppressed") is True:
        raise _state_unavailable(
            "hood_component_path_suppressed",
            "the collar component path is constructor-suppressed",
        )
    style = _component_style(design)
    hooded = style == "Hood2Panels"
    return SemanticState(
        rule_id="hood_presence",
        state_id="hooded" if hooded else "hoodless",
        phrase="hooded" if hooded else "hoodless",
        attributes={"present": hooded, "component_style": style},
    )


def _local_collar_state(
    design: Mapping[str, Any],
) -> SemanticState:
    inputs = neckline_inputs(design)
    component = inputs["component"]
    if component is None:
        return SemanticState(
            rule_id="collar_presence",
            state_id="collarless",
            phrase="collarless",
            attributes={"present": False, "component_style": None},
        )
    if component != "SimpleLapel":
        raise StateUnavailableError(
            "collar_presence_unrepresented",
            f"component style {component!r} has no approved collar state",
        )
    subtype = protected_neckline_phrase(inputs)
    if subtype == "V-neckline":
        raise StateUnavailableError(
            "collar_presence_unrepresented",
            "SimpleLapel's V-only phrase leaves collar presence unrepresented",
        )
    return SemanticState(
        rule_id="collar_presence",
        state_id="collared",
        phrase=f"having a {subtype}",
        attributes={
            "present": True,
            "component_style": "SimpleLapel",
            "subtype_phrase": subtype,
        },
    )


def _set_enum_value(
    design: Mapping[str, Any],
    path: str,
    value: Any,
) -> dict[str, Any]:
    """Write one declared ``.v`` leaf without consulting source YAML ranges.

    The config and frozen constructor allowlists are the target authority.
    Per-source ``range`` metadata is sampling provenance and can legitimately
    omit a constructor supported by the execution code.
    """

    expected_type = _ENUM_PARAMETER_TYPES.get(path)
    if expected_type is None:
        raise OperationContractError(f"undeclared enum write path: {path}")
    parameter = _parameter_node(design, path)
    if parameter is None:
        raise OperationContractError(
            f"candidate write path is absent from the design: {path}"
        )
    schema = parameter.get("type")
    if type(schema) is not str or schema != expected_type:
        raise OperationContractError(
            f"candidate write path has invalid enum metadata type: {path}"
        )

    if "v" not in parameter:
        raise OperationContractError(
            f"enum write path is missing from the design: {path}"
        )
    updated = plain_copy(design)
    updated_parameter = _parameter_node(updated, path)
    if not isinstance(updated_parameter, dict):  # Already validated above.
        raise TypeError("enum parameter did not copy to a mapping")
    updated_parameter["v"] = value
    return updated


def _all_reads_exist(
    design: Mapping[str, Any],
    reads: tuple[str, ...],
) -> bool:
    try:
        for path in reads:
            design_dotted_get(design, path)
    except KeyError:
        return False
    return True


def _symmetric_upper_is_applicable(
    context: RuleContext,
    reads: tuple[str, ...],
) -> bool:
    if not _all_reads_exist(context.design, reads):
        return False
    if not _enum_metadata_is_exact(context.design):
        return False
    dispatch = context.active_paths.dispatch
    upper = dispatch.get("upper")
    if upper not in {"Shirt", "FittedShirt"}:
        return False
    if dispatch.get("asymmetric") is not False:
        return False
    if dispatch.get("collar_component_suppressed") is not False:
        return False
    if upper == "FittedShirt":
        try:
            strapless = design_dotted_get(
                context.design,
                "design.shirt.strapless.v",
            )
        except KeyError:
            return False
        if strapless is not False:
            return False
    return True


def _hood_presence_pilot_guardrail(
    design: Mapping[str, Any],
    *,
    direction: str,
) -> tuple[bool, dict[str, Any]]:
    """Return the add-only pilot guardrail decision and stable evidence.

    A failed decision is ordinary inapplicability: callers omit the candidate
    and do not invent a second rejection-report API. Removal is deliberately
    unrestricted so existing hoods can always be taken off.
    """

    if direction not in {"add", "remove"}:
        raise ValueError(f"unsupported hood_presence direction: {direction!r}")
    hood_length = design_dotted_get(design, HOOD_LENGTH_PATH)
    evidence: dict[str, Any] = {
        "id": HOOD_ADD_GUARDRAIL_ID,
        "applies_to_direction": "add",
        "direction": direction,
        "parameter_path": HOOD_LENGTH_PATH,
        "operator": "<=",
        "threshold": HOOD_ADD_MAX_HOOD_LENGTH,
        "threshold_inclusive": True,
        "observed_value": hood_length,
    }
    if direction == "remove":
        evidence.update(
            {
                "applicable": False,
                "passed": None,
                "status": "not_applicable",
                "reason_code": HOOD_REMOVE_GUARDRAIL_REASON,
            }
        )
        return True, evidence

    if (
        isinstance(hood_length, bool)
        or not isinstance(hood_length, (int, float))
        or not math.isfinite(float(hood_length))
    ):
        evidence.update(
            {
                "applicable": True,
                "passed": False,
                "status": "rejected",
                "reason_code": "latent_hood_length_is_not_finite_numeric",
            }
        )
        return False, evidence

    passed = float(hood_length) <= HOOD_ADD_MAX_HOOD_LENGTH
    evidence.update(
        {
            "applicable": True,
            "passed": passed,
            "status": "passed" if passed else "rejected",
            "reason_code": (
                HOOD_ADD_ACCEPT_REASON if passed else HOOD_ADD_REJECTION_REASON
            ),
        }
    )
    return passed, evidence


def _oracle_target_state(
    context: RuleContext,
    *,
    rule_id: str,
    target_design: Mapping[str, Any],
    local_classifier: Callable[[Mapping[str, Any]], SemanticState],
) -> tuple[SemanticState, SemanticState] | None:
    """Return independently derived local/protected target states."""

    try:
        protected_states = context.classify_prompt(
            plain_copy(target_design),
            context.raw_body_yaml,
        )
    except StateUnavailableError as exc:
        _unrepresented_or_raise(exc, rule_id=rule_id)
        return None
    if not isinstance(protected_states, Mapping):
        raise TypeError("protected state oracle must return a mapping")
    try:
        local_state = local_classifier(target_design)
    except StateUnavailableError as exc:
        _unrepresented_or_raise(exc, rule_id=rule_id)
        return None
    protected_state = protected_states.get(rule_id)
    if rule_id == "collar_presence":
        if local_state.state_id == "collarless":
            if protected_state is not None:
                raise ProtectedStateSchemaError(
                    "unexpected_collar_state",
                    "collarless target has a protected collar state",
                )
            return local_state, local_state
        if protected_state is None:
            return None
        if not _prompt_collar_agrees(local_state, protected_state):
            raise ProtectedStateSchemaError(
                "protected_state_mismatch",
                "collar_presence protected target state mismatch",
            )
    else:
        if protected_state is None:
            return None
        if not _prompt_hood_agrees(local_state, protected_state):
            raise ProtectedStateSchemaError(
                "protected_state_mismatch",
                "hood_presence protected target state mismatch",
            )
    return local_state, protected_state


def _source_state(
    context: RuleContext,
    *,
    rule_id: str,
    local_classifier: Callable[[Mapping[str, Any]], SemanticState],
) -> tuple[SemanticState, SemanticState] | None:
    try:
        local = local_classifier(context.design)
    except StateUnavailableError as exc:
        _unrepresented_or_raise(exc, rule_id=rule_id)
        return None
    protected_state = context.prompt_states.get(rule_id)
    if rule_id == "collar_presence":
        if local.state_id == "collarless":
            if protected_state is not None:
                raise ProtectedStateSchemaError(
                    "unexpected_collar_state",
                    "collarless source has a protected collar state",
                )
            return local, local
        if protected_state is None:
            return None
        if not _prompt_collar_agrees(local, protected_state):
            raise ProtectedStateSchemaError(
                "protected_state_mismatch",
                "collar_presence protected source state mismatch",
            )
    else:
        if protected_state is None:
            return None
        if not _prompt_hood_agrees(local, protected_state):
            raise ProtectedStateSchemaError(
                "protected_state_mismatch",
                "hood_presence protected source state mismatch",
            )
    return local, protected_state


def _applicability_evidence(
    context: RuleContext,
    *,
    operation: Operation,
    target_design: Mapping[str, Any],
    source_local_state: SemanticState,
    source_protected_state: SemanticState,
    target_local_state: SemanticState,
    target_protected_state: SemanticState,
) -> dict[str, Any]:
    """Stable content evidence without invocation-local paths or identifiers."""

    return {
        "rule_id": operation.rule_id,
        "scope": operation.scope,
        "source_design_hash": canonical_hash(context.design),
        "target_design_hash": canonical_hash(target_design),
        "body_hash": canonical_hash(context.body),
        "active_paths_design_hash": context.active_paths.design_hash,
        "upper_constructor": context.active_paths.dispatch.get("upper"),
        "symmetric_upper": (context.active_paths.dispatch.get("asymmetric") is False),
        "component_style": design_dotted_get(
            context.design,
            COMPONENT_STYLE_PATH,
        ),
        "stored_front_neckline": design_dotted_get(
            context.design,
            FRONT_NECKLINE_PATH,
        ),
        "stored_back_neckline": design_dotted_get(
            context.design,
            BACK_NECKLINE_PATH,
        ),
        "read_contract_hash": canonical_hash(
            {path: design_dotted_get(context.design, path) for path in operation.reads}
        ),
        "source_state": source_local_state.to_payload(),
        "target_state": target_local_state.to_payload(),
        "source_local_state": source_local_state.to_payload(),
        "source_protected_state": source_protected_state.to_payload(),
        "target_local_state": target_local_state.to_payload(),
        "target_protected_state": target_protected_state.to_payload(),
        "source_state_match": True,
        "target_state_match": True,
    }


def _planned_operation(
    context: RuleContext,
    *,
    rule_config: RuleConfig,
    parameter_spec: ParameterSpec,
    rule_id: str,
    scope: str,
    channels: tuple[str, ...],
    feature_phrase: str,
    direction: str,
    topology_kind: TopologyKind,
    reads: tuple[str, ...],
    writes: tuple[str, ...],
    conflicts: tuple[str, ...],
    prerequisites: tuple[str, ...],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    target_design: Mapping[str, Any],
    source_local_state: SemanticState,
    source_protected_state: SemanticState,
    target_local_state: SemanticState,
    target_protected_state: SemanticState,
    extra_applicability_evidence: Mapping[str, Any] | None = None,
) -> PlannedOperation:
    operation = Operation(
        rule_id=rule_id,
        semantic_group=rule_config.semantic_group,
        scope=scope,
        direction=direction,
        magnitude="structural",
        structural_level=rule_config.structural_level,
        reads=tuple(sorted(reads)),
        writes=tuple(sorted(writes)),
        conflicts=tuple(sorted(conflicts)),
        prerequisites=tuple(sorted(prerequisites)),
        edit_mode=parameter_spec.edit_mode,
        effect_metric_id=rule_config.effect_metric,
        effect_contract_id=rule_config.effect_contract,
        before=before,
        after=after,
        relative_change=None,
        requested_magnitude="structural",
        realized_magnitude="structural",
        clamped=False,
        instruction_tokens={
            "feature": feature_phrase,
            "source_state_id": source_local_state.state_id,
            "target_state_id": target_local_state.state_id,
        },
    )
    transition = SemanticTransition(
        rule_id=rule_id,
        scope=scope,
        channels=channels,
        feature_phrase=feature_phrase,
        source_state=source_local_state,
        target_state=target_local_state,
        direction=direction,
    )
    applicability_evidence = _applicability_evidence(
        context,
        operation=operation,
        target_design=target_design,
        source_local_state=source_local_state,
        source_protected_state=source_protected_state,
        target_local_state=target_local_state,
        target_protected_state=target_protected_state,
    )
    if extra_applicability_evidence is not None:
        applicability_evidence.update(extra_applicability_evidence)
    return PlannedOperation(
        operation=operation,
        transitions=(transition,),
        topology_kind=topology_kind,
        applicability_evidence=applicability_evidence,
        conflict_evidence=empty_conflict_evidence(
            operation.rule_id,
            operation.conflicts,
        ),
    )


def _presence_candidates(
    context: RuleContext,
    rule_config: RuleConfig,
    parameter_spec: ParameterSpec,
    *,
    rule_id: str,
) -> tuple[PlannedOperation, ...]:
    hood = rule_id == "hood_presence"
    reads = HOOD_PRESENCE_READS if hood else COLLAR_PRESENCE_READS
    if not _symmetric_upper_is_applicable(context, reads):
        return ()
    try:
        style = design_dotted_get(context.design, COMPONENT_STYLE_PATH)
        back = design_dotted_get(context.design, BACK_NECKLINE_PATH)
        front = design_dotted_get(context.design, FRONT_NECKLINE_PATH)
    except KeyError:
        return ()
    target_component = "Hood2Panels" if hood else "SimpleLapel"
    if style is not None and (type(style) is not str or style != target_component):
        return ()
    if back != "CircleNeckHalf" or (hood and front != "CircleNeckHalf"):
        return ()
    classifier = classify_hood_state if hood else _local_collar_state
    source_states = _source_state(
        context, rule_id=rule_id, local_classifier=classifier
    )
    if source_states is None:
        return ()
    source_local, source_protected = source_states
    target_style = (
        parameter_spec.transitions[1]
        if style == parameter_spec.transitions[0]
        else parameter_spec.transitions[0]
    )
    direction = "add" if style is None else "remove"
    extra_evidence = None
    if hood:
        passed, guardrail = _hood_presence_pilot_guardrail(
            context.design, direction=direction
        )
        if not passed:
            return ()
        extra_evidence = {"pilot_guardrail": guardrail}
    target_design = _set_enum_value(
        context.design, COMPONENT_STYLE_PATH, target_style
    )
    target_states = _oracle_target_state(
        context,
        rule_id=rule_id,
        target_design=target_design,
        local_classifier=classifier,
    )
    if target_states is None:
        return ()
    target_local, target_protected = target_states
    if source_local.state_id == target_local.state_id:
        return ()
    return (
        _planned_operation(
            context,
            rule_config=rule_config,
            parameter_spec=parameter_spec,
            rule_id=rule_id,
            scope="garment",
            channels=("garment",),
            feature_phrase="the top",
            direction=direction,
            topology_kind=TopologyKind.MEMBERSHIP,
            reads=reads,
            writes=_WRITE_COMPONENT,
            conflicts=(
                ("collar_presence", "neckline_type")
                if hood
                else ("hood_presence", "neckline_type")
            ),
            prerequisites=(
                ("round_front_and_back_necklines", "symmetric_upper")
                if hood
                else (
                    "explicit_protected_collar_state",
                    "round_back_neckline",
                    "symmetric_upper",
                )
            ),
            before={COMPONENT_STYLE_PATH: style},
            after={COMPONENT_STYLE_PATH: target_style},
            target_design=target_design,
            source_local_state=source_local,
            source_protected_state=source_protected,
            target_local_state=target_local,
            target_protected_state=target_protected,
            extra_applicability_evidence=extra_evidence,
        ),
    )


def apply(design: dict[str, Any], planned: PlannedOperation) -> dict[str, Any]:
    """Apply one collar/hood component toggle."""

    return _set_enum_value(
        design,
        COMPONENT_STYLE_PATH,
        planned.operation.after[COMPONENT_STYLE_PATH],
    )


def _result(
    rule_id: str,
    passed: bool,
    code: str,
    details: Mapping[str, Any],
    *,
    stage: str = "effect",
) -> EffectResult:
    return EffectResult(
        passed=passed,
        code=code,
        stage=stage,
        evidence={
            "rule_id": rule_id,
            "details": plain_copy(details),
        },
    )


class _PatternError(ValueError):
    pass


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _PatternError(f"{path} must be a sequence")
    return value


def _panels(specification: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        panels = specification["pattern"]["panels"]
    except (KeyError, TypeError) as exc:
        raise _PatternError("pattern.panels is required") from exc
    if not isinstance(panels, Mapping):
        raise _PatternError("pattern.panels must be a mapping")
    return panels


def _edge(
    specification: Mapping[str, Any], panel_name: str, edge_index: int
) -> Mapping[str, Any]:
    panel = _panels(specification).get(panel_name)
    if not isinstance(panel, Mapping):
        raise _PatternError(f"missing panel {panel_name!r}")
    edges = _sequence(panel.get("edges"), f"{panel_name}.edges")
    if (
        isinstance(edge_index, bool)
        or not isinstance(edge_index, int)
        or edge_index < 0
        or edge_index >= len(edges)
        or not isinstance(edges[edge_index], Mapping)
    ):
        raise _PatternError(f"invalid edge reference {panel_name}[{edge_index!r}]")
    return edges[edge_index]


def _edge_label(edge: Mapping[str, Any]) -> str | None:
    label = edge.get("label")
    return label if isinstance(label, str) else None


def _stitches(
    specification: Mapping[str, Any],
) -> tuple[tuple[tuple[str, int], tuple[str, int], tuple[bytes, ...]], ...]:
    try:
        return stitch_records(specification)
    except PatternGeometryError as exc:
        raise _PatternError(str(exc)) from exc


def _metadata(specification: Mapping[str, Any]) -> Any:
    try:
        return specification_metadata(specification)
    except PatternGeometryError as exc:
        raise _PatternError(str(exc)) from exc


def _membership_effect(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    direction: str,
    topology_policy: str,
) -> EffectResult:
    try:
        source_panels = _panels(source)
        target_panels = _panels(target)
        source_component = {
            name for name in source_panels if COLLAR_PANEL.fullmatch(name)
        }
        target_component = {
            name for name in target_panels if COLLAR_PANEL.fullmatch(name)
        }
        expected = {
            "left_collar_front",
            "left_collar_back",
            "right_collar_front",
            "right_collar_back",
        }
        membership_ok = (
            not source_component and target_component == expected
            if direction == "add"
            else source_component == expected and not target_component
        )
        source_unrelated = set(source_panels) - source_component
        target_unrelated = set(target_panels) - target_component
        unrelated_membership_ok = source_unrelated == target_unrelated
        unrelated_panels_equal = all(
            canonical_json_bytes(source_panels[name])
            == canonical_json_bytes(target_panels[name])
            for name in source_unrelated.intersection(target_unrelated)
        )
        metadata_match = canonical_json_bytes(_metadata(source)) == canonical_json_bytes(
            _metadata(target)
        )
        source_stitches = Counter(_stitches(source))
        target_stitches = Counter(_stitches(target))
        removed = source_stitches - target_stitches
        added = target_stitches - source_stitches

        def local(stitch: tuple[Any, ...], specification: Mapping[str, Any]) -> bool:
            endpoints = stitch[:2]
            component = tuple(
                bool(COLLAR_PANEL.fullmatch(panel)) for panel, _edge_index in endpoints
            )
            if all(component):
                return True
            if sum(component) != 1:
                return False
            external = endpoints[0 if not component[0] else 1]
            return _edge_label(_edge(specification, *external)) in {
                "left_collar",
                "right_collar",
            }

        localized = all(local(stitch, source) for stitch in removed) and all(
            local(stitch, target) for stitch in added
        )
        stitch_direction_ok = (
            bool(added) and not removed
            if direction == "add"
            else bool(removed) and not added
        )
    except (KeyError, TypeError, ValueError, ArithmeticError, _PatternError) as exc:
        return _result(
            "collar_presence",
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
        "metadata_match": metadata_match,
        "localized_stitch_change": localized,
        "stitch_direction_ok": stitch_direction_ok,
        "topology_policy": topology_policy,
    }
    passed = all((membership_ok, unrelated_membership_ok, localized, stitch_direction_ok))
    if topology_policy == "strict":
        passed = passed and unrelated_panels_equal and metadata_match
    return _result(
        "collar_presence",
        passed,
        "effect/passed" if passed else "effect/topology_mismatch",
        details,
    )


def _collar_attachments(
    specification: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, bool], bool, tuple[str, ...]]:
    panels = {name for name in _panels(specification) if COLLAR_PANEL.fullmatch(name)}
    side_attached = {"left": False, "right": False}
    panel_attached = {name: False for name in panels}
    mismatches = []
    for stitch in _stitches(specification):
        endpoints = stitch[:2]
        component_indices = [
            index
            for index, (panel_name, _edge_index) in enumerate(endpoints)
            if COLLAR_PANEL.fullmatch(panel_name)
        ]
        if len(component_indices) != 1:
            continue
        component_index = component_indices[0]
        component_name = endpoints[component_index][0]
        match = COLLAR_PANEL.fullmatch(component_name)
        if match is None:  # pragma: no cover - selected above.
            raise _PatternError("component panel attribution drifted")
        side, half = match.groups()
        external_name, external_edge = endpoints[1 - component_index]
        expected_external = f"{side}_{'f' if half == 'front' else 'b'}torso"
        label = _edge_label(_edge(specification, external_name, external_edge))
        if label == f"{side}_collar":
            side_attached[side] = True
        if external_name == expected_external and label == f"{side}_collar":
            panel_attached[component_name] = True
        else:
            mismatches.append(f"{component_name}->{external_name}:{label}")
    return (
        side_attached,
        panel_attached,
        not mismatches,
        tuple(sorted(mismatches)),
    )


def _validate_collar(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    direction: str,
    topology_policy: str,
) -> EffectResult:
    result = _membership_effect(source, target, direction, topology_policy)
    if not result.passed:
        return result
    present = target if direction == "add" else source
    try:
        sides, panels, exact, mismatches = _collar_attachments(present)
    except (KeyError, TypeError, ValueError, ArithmeticError, _PatternError) as exc:
        return _result(
            "collar_presence",
            False,
            "effect/measurement_error",
            {"message": str(exc)},
        )
    details = dict(result.evidence["details"])
    details.update(
        collar_stitch_membership=sides,
        collar_panel_attachments=panels,
        exact_attachment_relations=exact,
        attachment_mismatches=mismatches,
    )
    passed = all(sides.values()) and all(panels.values()) and exact
    return _result(
        "collar_presence",
        passed,
        "effect/passed" if passed else "effect/topology_mismatch",
        details,
    )


def _stitch_payload(stitch: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "endpoints": (
            {"panel": stitch[0][0], "edge": stitch[0][1]},
            {"panel": stitch[1][0], "edge": stitch[1][1]},
        ),
        "markers": tuple(repr(marker) for marker in stitch[2]),
    }


def _counter_payload(counter: Counter[Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"stitch": _stitch_payload(stitch), "count": count}
        for stitch, count in sorted(counter.items(), key=repr)
    )


def _hood_signature(signature: Mapping[str, Any]) -> dict[str, Any]:
    names = tuple(
        record["name"]
        for record in signature["panels"]
        if _HOOD_PANEL.fullmatch(record["name"])
    )
    incident = tuple(
        {"stitch": _stitch_payload(stitch), "count": count}
        for stitch, count in signature["stitches"]
        if stitch[0][0] in names or stitch[1][0] in names
    )
    return {
        "panel_names": names,
        "panel_count": len(names),
        "panel_topology": {
            record["name"]: repr(record)
            for record in signature["panels"]
            if record["name"] in names
        },
        "incident_stitches": incident,
        "incident_stitch_count": sum(item["count"] for item in incident),
    }


def _non_collar_edge_signature(panel: Mapping[str, Any]) -> Any:
    records = []
    for raw in _sequence(panel.get("edges"), "panel.edges"):
        if not isinstance(raw, Mapping):
            raise _PatternError("panel edges must be mappings")
        label = _edge_label(raw)
        if label is not None and label.endswith("_collar"):
            continue
        records.append(
            {
                "type": edge_type(raw),
                "metadata": {
                    key: value
                    for key, value in raw.items()
                    if key not in {"endpoints", "curvature"}
                },
                "curvature_metadata": (
                    {
                        key: value
                        for key, value in raw["curvature"].items()
                        if key not in {"type", "params"}
                    }
                    if isinstance(raw.get("curvature"), Mapping)
                    else {}
                ),
            }
        )
    return plain_copy({"panel_label": panel.get("label"), "edges": records})


def _single_panel_spec(
    specification: Mapping[str, Any], panel_name: str
) -> dict[str, Any]:
    return {
        "pattern": {
            "panels": {panel_name: _panels(specification)[panel_name]},
            "stitches": [],
            "panel_order": [panel_name],
        },
        "parameters": {},
        "parameter_order": [],
        "properties": specification["properties"],
    }


def _hood_or_collar_local(
    stitch: tuple[Any, ...], specification: Mapping[str, Any]
) -> bool:
    if any(_HOOD_PANEL.fullmatch(endpoint[0]) for endpoint in stitch[:2]):
        return True
    for panel_name, edge_index in stitch[:2]:
        panel = _panels(specification).get(panel_name)
        if not isinstance(panel, Mapping):
            continue
        edges = panel.get("edges")
        if isinstance(edges, Sequence) and not isinstance(
            edges, (str, bytes, bytearray)
        ) and 0 <= edge_index < len(edges) and isinstance(edges[edge_index], Mapping):
            label = _edge_label(edges[edge_index])
            if label is not None and label.endswith("_collar"):
                return True
    return False


def _torso_relations(counter: Counter[Any]) -> Counter[Any]:
    result = Counter()
    for stitch, count in counter.items():
        panels = (stitch[0][0], stitch[1][0])
        if all(_TORSO_PANEL.fullmatch(panel) for panel in panels):
            relation = (
                tuple(sorted(panels)),
                tuple(repr(marker) for marker in stitch[2]),
            )
            result[relation] += count
    return result


class _ContractResult(NamedTuple):
    passed: bool
    code: str | None
    message: str | None
    details: dict[str, Any]


def _contract_fail(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> _ContractResult:
    return _ContractResult(False, code, message, details or {})


def _contract_pass(details: dict[str, Any]) -> _ContractResult:
    return _ContractResult(True, None, None, details)


def _hood_contract(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    direction: str,
    topology_policy: str,
    before_topology: Mapping[str, Any],
    after_topology: Mapping[str, Any],
) -> _ContractResult:
    metadata_match = _metadata(before) == _metadata(after)
    if topology_policy == "strict" and not metadata_match:
        return _contract_fail(
            "undeclared_topology_change",
            "pattern metadata changed during a hood toggle",
        )
    before_panels = _panels(before)
    after_panels = _panels(after)
    before_hood = {name for name in before_panels if _HOOD_PANEL.fullmatch(name)}
    after_hood = {name for name in after_panels if _HOOD_PANEL.fullmatch(name)}
    expected = {"left_hood", "right_hood"}
    transition_ok = (
        not before_hood and after_hood == expected
        if direction == "add"
        else before_hood == expected and not after_hood
    )
    if not transition_ok:
        return _contract_fail(
            "effect_contract_failed",
            "hood toggle must perform exactly null <-> two Hood2Panels panels",
            {
                "before_hood_panels": tuple(sorted(before_hood)),
                "after_hood_panels": tuple(sorted(after_hood)),
            },
        )
    before_nonhood = set(before_panels) - before_hood
    after_nonhood = set(after_panels) - after_hood
    if before_nonhood != after_nonhood:
        code = (
            "undeclared_topology_change"
            if topology_policy == "strict"
            else "effect_contract_failed"
        )
        return _contract_fail(code, "a non-hood panel was added or removed")

    strict_nonhood_match: bool | None = (
        True if topology_policy == "strict" else None
    )
    if topology_policy == "strict":
        before_records = {
            record["name"]: record for record in before_topology["panels"]
        }
        after_records = {
            record["name"]: record for record in after_topology["panels"]
        }
        for name in sorted(before_nonhood):
            if before_records[name] == after_records[name]:
                continue
            strict_nonhood_match = False
            if not _TORSO_PANEL.fullmatch(name):
                return _contract_fail(
                    "undeclared_topology_change",
                    f"unrelated panel topology changed: {name}",
                )
            if _non_collar_edge_signature(
                before_panels[name]
            ) != _non_collar_edge_signature(after_panels[name]):
                return _contract_fail(
                    "undeclared_topology_change",
                    f"non-collar torso topology changed: {name}",
                )
        for name in sorted(before_nonhood):
            if _TORSO_PANEL.fullmatch(name):
                continue
            comparison = compare_patterns(
                _single_panel_spec(before, name),
                _single_panel_spec(after, name),
                atol=_PATTERN_ATOL,
            )
            if not comparison.equal:
                strict_nonhood_match = False
                return _contract_fail(
                    "effect_contract_failed",
                    f"unrelated panel geometry changed: {name}",
                )

    before_stitches = Counter(dict(before_topology["stitches"]))
    after_stitches = Counter(dict(after_topology["stitches"]))
    removed = before_stitches - after_stitches
    added = after_stitches - before_stitches
    if not removed and not added:
        return _contract_fail(
            "effect_contract_failed",
            "hood panels changed without a corresponding stitch change",
        )
    for counter, specification, label in (
        (removed, before, "removed"),
        (added, after, "added"),
    ):
        unrelated = Counter(
            {
                stitch: count
                for stitch, count in counter.items()
                if not _hood_or_collar_local(stitch, specification)
                and not all(
                    _TORSO_PANEL.fullmatch(endpoint[0])
                    for endpoint in stitch[:2]
                )
            }
        )
        if unrelated:
            return _contract_fail(
                "undeclared_stitch_change",
                f"{label} stitches include a non-hood/non-collar change",
                {"unrelated_stitches": _counter_payload(unrelated)},
            )
    if _torso_relations(removed) != _torso_relations(added):
        return _contract_fail(
            "undeclared_stitch_change",
            "hood construction changed a torso seam relationship",
        )
    incident_before = sum(
        count
        for stitch, count in before_stitches.items()
        if any(panel in before_hood for panel, _edge_index in stitch[:2])
    )
    incident_after = sum(
        count
        for stitch, count in after_stitches.items()
        if any(panel in after_hood for panel, _edge_index in stitch[:2])
    )
    if direction == "add" and not (incident_before == 0 < incident_after):
        return _contract_fail(
            "effect_contract_failed",
            "added hood panels must gain incident stitches",
        )
    if direction == "remove" and not (incident_after == 0 < incident_before):
        return _contract_fail(
            "effect_contract_failed",
            "removed hood panels must lose their incident stitches",
        )
    return _contract_pass(
        {
            "removed_stitches": _counter_payload(removed),
            "added_stitches": _counter_payload(added),
            "hood_incident_stitches_before": incident_before,
            "hood_incident_stitches_after": incident_after,
            "topology_policy": topology_policy,
            "strict_metadata_match": metadata_match,
            "strict_nonhood_topology_match": strict_nonhood_match,
        }
    )


def _hood_evidence(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    delta: Mapping[str, Any],
    direction: str,
    threshold: float,
    passed: bool,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "metric_id": "hood_panel_stitch_signature",
        "before": before,
        "after": after,
        "delta": delta,
        "expected_direction": direction,
        "threshold": threshold,
        "passed": passed,
        "details": details,
        "metric_threshold_cm": None,
        "boundary_threshold_cm": None,
    }


def _hood_pattern_decision(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    direction: str,
    threshold: float,
    topology_policy: str,
) -> dict[str, Any]:
    metric_id = "hood_panel_stitch_signature"
    contract_id = "hood_panel_stitch_change"

    def bare_failure(exc: Exception) -> dict[str, Any]:
        return {
            "passed": False,
            "evidence": None,
            "failure": {
                "code": "effect_metric_failed",
                "metric_id": metric_id,
                "contract_id": contract_id,
                "message": f"{type(exc).__name__}: {exc}",
                "details": {},
            },
        }

    try:
        before_topology = topology_signature(before)
        after_topology = topology_signature(after)
        before_value = _hood_signature(before_topology)
        after_value = _hood_signature(after_topology)
        before_names = set(before_value["panel_names"])
        after_names = set(after_value["panel_names"])
        panel_delta = after_value["panel_count"] - before_value["panel_count"]
        stitch_delta = (
            after_value["incident_stitch_count"]
            - before_value["incident_stitch_count"]
        )
        delta = {
            "panel_count": panel_delta,
            "incident_stitch_count": stitch_delta,
            "added_panels": tuple(sorted(after_names - before_names)),
            "removed_panels": tuple(sorted(before_names - after_names)),
        }
        changed = bool(
            delta["added_panels"] or delta["removed_panels"] or stitch_delta
        )
        direction_ok = (
            panel_delta > 0 and stitch_delta > 0
            if direction == "add"
            else panel_delta < 0 and stitch_delta < 0
        )
    except Exception as exc:
        return bare_failure(exc)

    base_details = {
        "contract_id": contract_id,
        "scope": "garment",
        "target_panels": (),
    }

    def failure(
        code: str,
        message: str,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "passed": False,
            "evidence": _hood_evidence(
                before_value,
                after_value,
                delta,
                direction,
                threshold,
                False,
                details,
            ),
            "failure": {
                "code": code,
                "metric_id": metric_id,
                "contract_id": contract_id,
                "message": message,
                "details": details,
            },
        }

    try:
        noop = semantic_pattern_hash(before) == semantic_pattern_hash(after)
    except Exception as exc:
        return failure(
            "effect_metric_failed",
            f"{type(exc).__name__}: {exc}",
            base_details,
        )
    if noop:
        return failure(
            "operation_noop",
            "before and after patterns are semantically identical",
            base_details,
        )
    try:
        contract = _hood_contract(
            before,
            after,
            direction,
            topology_policy,
            before_topology,
            after_topology,
        )
    except Exception as exc:
        return failure(
            "effect_metric_failed",
            f"{type(exc).__name__}: {exc}",
            base_details,
        )
    details = {**base_details, **contract.details}
    if not contract.passed:
        return failure(
            contract.code or "effect_contract_failed",
            contract.message or "effect contract failed",
            details,
        )
    if not changed:
        return failure(
            "operation_noop",
            "the requested metric did not change",
            details,
        )
    if not direction_ok:
        return failure(
            "effect_direction_failed",
            "the observed metric moved outside the requested scope or direction",
            details,
        )
    return {
        "passed": True,
        "evidence": _hood_evidence(
            before_value,
            after_value,
            delta,
            direction,
            threshold,
            True,
            details,
        ),
        "failure": None,
    }


def _gate(decision: Mapping[str, Any]) -> dict[str, Any]:
    failure = decision["failure"]
    code = None if failure is None else failure["code"]
    stage = (
        "metric"
        if code == "metric_threshold_failed"
        else "boundary"
        if code == "boundary_threshold_failed"
        else "effect"
    )
    failures = []
    if failure is not None:
        failures.append(
            {
                "code": code,
                "operation_index": 0,
                "subject": failure["metric_id"],
                "details": {
                    "contract_id": failure["contract_id"],
                    "message": failure["message"],
                    "failure_details": failure["details"],
                },
            }
        )
    return {
        "gate": stage,
        "passed": failure is None,
        "failure_code": code,
        "failures": failures,
        "length_transitions": [],
        "parameter_evidence": [],
        "effect_evidence_count": 0 if decision["evidence"] is None else 1,
    }


def _validate_hood(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    direction: str,
    threshold: float,
    topology_policy: str,
) -> EffectResult:
    strict = _hood_pattern_decision(
        source,
        target,
        direction=direction,
        threshold=threshold,
        topology_policy="strict",
    )
    strict_failure = strict["failure"]
    local_recheck_eligible = bool(
        strict_failure is not None
        and strict_failure["code"] == "effect_contract_failed"
        and strict_failure["message"].startswith(
            "unrelated panel geometry changed:"
        )
    )
    if topology_policy == "local" and (strict["passed"] or local_recheck_eligible):
        observed = _hood_pattern_decision(
            source,
            target,
            direction=direction,
            threshold=threshold,
            topology_policy="local",
        )
    else:
        observed = strict
    observed_gate = _gate(observed)
    strict_gate = _gate(strict)
    code = "effect/passed" if observed["passed"] else str(observed_gate["failure_code"])
    if topology_policy == "strict" and not observed["passed"] and local_recheck_eligible:
        code = "effect/topology_mismatch"
    return _result(
        "hood_presence",
        bool(observed["passed"]),
        code,
        {
            "delegated_metric_id": "hood_panel_stitch_signature",
            "delegated_contract_id": "hood_panel_stitch_change",
            "topology_policy": topology_policy,
            "local_recheck_eligible": local_recheck_eligible,
            "v4_effect_result": observed,
            "v4_effect_gate": observed_gate,
            "strict_v4_effect_result": strict,
            "strict_v4_effect_gate": strict_gate,
        },
        stage=observed_gate["gate"],
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
    """Validate one collar/hood membership transition."""

    del source_design, target_design, body
    if topology_policy not in {"strict", "local"}:
        raise ValueError("topology_policy must be exactly 'strict' or 'local'")
    operation = planned_operation.operation
    if operation.rule_id not in {"hood_presence", "collar_presence"}:
        raise ValueError("operation names a foreign rule")
    try:
        rule_config = config.rules[operation.rule_id]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"config must contain {operation.rule_id}") from exc
    if operation.effect_metric_id != rule_config.effect_metric:
        raise ValueError("operation names the wrong collar/hood effect metric")
    if operation.effect_contract_id != rule_config.effect_contract:
        raise ValueError("operation names the wrong collar/hood effect contract")
    if operation.scope != "garment":
        raise ValueError("collar/hood presence requires scope 'garment'")
    if operation.direction not in {"add", "remove"}:
        raise ValueError("collar/hood direction must be exactly add or remove")
    if operation.rule_id == "hood_presence":
        return _validate_hood(
            source_specification,
            target_specification,
            operation.direction,
            rule_config.perceptibility_cm,
            topology_policy,
        )
    return _validate_collar(
        source_specification,
        target_specification,
        operation.direction,
        topology_policy,
    )


def build_rule(config: EditConfig, rule_id: str) -> Rule:
    """Bind one collar or hood rule to the loaded configuration."""

    if not isinstance(config, EditConfig):
        raise TypeError("config must be an EditConfig")
    definitions = {
        "hood_presence": ("hood_membership", (None, "Hood2Panels")),
        "collar_presence": ("collar_membership", (None, "SimpleLapel")),
    }
    if rule_id not in definitions:
        raise ValueError(f"unsupported collar/hood rule: {rule_id!r}")
    try:
        metric, transitions = definitions[rule_id]
        rule_config = config.rules[rule_id]
    except KeyError as exc:
        raise ValueError(f"config must contain {rule_id}") from exc
    parameter_spec = _validate_rule_config(
        rule_config,
        rule_id=rule_id,
        parameter_name="style",
        semantic_group="collar",
        structural_level=2,
        effect_metric=metric,
        effect_contract="exact_membership_change",
        transitions=transitions,
    )

    def candidates(context: RuleContext) -> tuple[PlannedOperation, ...]:
        return _presence_candidates(
            context,
            rule_config,
            parameter_spec,
            rule_id=rule_id,
        )
    return Rule(rule_id, candidates, apply, validate)


def build_rules(config: EditConfig) -> tuple[Rule, ...]:
    """Build the canonical hood-then-collar rule pair."""

    return tuple(
        build_rule(config, rule_id)
        for rule_id in ("hood_presence", "collar_presence")
    )
