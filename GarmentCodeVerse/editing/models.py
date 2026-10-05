"""Plain in-memory contracts and semantic-edit payload codecs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Optional, Tuple

from .canonical_io import canonical_hash, plain_copy


class OperationContractError(ValueError):
    """An in-memory operation violates its rule's declared contract."""


def _payload(value: Any) -> Any:
    """Return a detached JSON-shaped value for persistence."""

    if isinstance(value, Mapping):
        return {key: _payload(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload(child) for child in value]
    return value


def _mapping(value: Any, expected: frozenset[str], field_name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    actual = set(value)
    if actual != set(expected):
        raise ValueError(f"{field_name} fields are invalid")
    return value


@dataclass(slots=True)
class RuleContext:
    """Plain operation-local values consumed by semantic rules."""

    design: dict[str, Any]
    body: dict[str, Any]
    raw_body_yaml: bytes
    active_paths: Any
    constructor_facts: dict[str, Any]
    prompt_states: dict[str, Any]
    classify_prompt: Callable[[dict[str, Any], bytes], dict[str, Any]]


@dataclass(slots=True)
class EffectResult:
    """One semantic effect decision and its supporting evidence."""

    passed: bool
    code: str
    evidence: dict[str, Any]
    stage: str = "effect"


def rule_effect_result(
    rule_id: str,
    passed: bool,
    code: str,
    details: Mapping[str, Any],
    *,
    stage: str = "effect",
) -> EffectResult:
    """Build the common evidence envelope emitted by semantic rules."""

    return EffectResult(
        passed=passed,
        code=code,
        evidence={"rule_id": rule_id, "details": plain_copy(details)},
        stage=stage,
    )


@dataclass(slots=True)
class Rule:
    """The complete functional surface of one semantic rule."""

    id: str
    candidates: Callable[[RuleContext], tuple[Any, ...]]
    apply: Callable[[dict[str, Any], Any], dict[str, Any]]
    validate: Callable[..., EffectResult]


@dataclass(slots=True)
class BodySnapshot:
    """The random-body values used throughout an edit pair."""

    body: dict[str, Any]
    raw_yaml: bytes

    def __post_init__(self) -> None:
        self.body = plain_copy(self.body)


@dataclass(slots=True)
class SourceRecord:
    """The design and body facts consumed during source planning."""

    source_key: str
    sample_id: str
    design: dict[str, Any]
    body_snapshot: BodySnapshot

    def __post_init__(self) -> None:
        self.design = plain_copy(self.design)


@dataclass(slots=True)
class Operation:
    """One accepted body-independent semantic design operation."""

    rule_id: str
    semantic_group: str
    scope: str
    direction: str
    magnitude: str
    structural_level: int
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    conflicts: Tuple[str, ...] = ()
    prerequisites: Tuple[str, ...] = ()
    edit_mode: str = ""
    effect_metric_id: str = ""
    effect_contract_id: str = ""
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    relative_change: Optional[float] = None
    requested_magnitude: Optional[str] = None
    realized_magnitude: Optional[str] = None
    clamped: bool = False
    instruction_tokens: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.before = plain_copy(self.before)
        self.after = plain_copy(self.after)
        self.instruction_tokens = plain_copy(self.instruction_tokens)


@dataclass(slots=True)
class LengthCategoryChannel:
    """One ordered, canonical semantic length classification channel."""

    name: str
    category: str
    order_index: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "order_index": self.order_index,
        }


@dataclass(slots=True)
class LengthCategoryObservation:
    """The explicitly ordered category channels observed for one length rule."""

    rule_id: str
    scope: str
    channels: Tuple[LengthCategoryChannel, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "scope": self.scope,
            "channels": [channel.to_payload() for channel in self.channels],
        }


class StateUnavailableError(ValueError):
    """One rule-local state cannot be represented for this design."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class ProtectedStateSchemaError(ValueError):
    """The protected prompt contains missing or conflicting anchors."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def empty_conflict_evidence(
    rule_id: str,
    declared_conflicts: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"rule_id": rule_id}
    if declared_conflicts is not None:
        evidence["declared_conflicts"] = list(declared_conflicts)
    evidence.update(
        earlier_rule_ids=[],
        earlier_operation_hashes=[],
        earlier_write_paths=[],
        overlap_write_paths=[],
    )
    return evidence


class TopologyKind(str, Enum):
    NONE = "none"
    BOUNDARY = "boundary"
    MEMBERSHIP = "membership"


@dataclass(slots=True)
class SemanticState:
    rule_id: str
    state_id: str
    phrase: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.attributes = plain_copy(self.attributes)

    def to_payload(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "state_id": self.state_id,
            "phrase": self.phrase,
            "attributes": _payload(self.attributes),
        }


@dataclass(slots=True)
class SemanticTransition:
    rule_id: str
    scope: str
    channels: tuple[str, ...]
    feature_phrase: str
    source_state: SemanticState
    target_state: SemanticState
    direction: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "scope": self.scope,
            "channels": list(self.channels),
            "feature_phrase": self.feature_phrase,
            "source_state": self.source_state.to_payload(),
            "target_state": self.target_state.to_payload(),
            "direction": self.direction,
        }


@dataclass(slots=True)
class PlannedOperation:
    operation: Operation
    transitions: tuple[SemanticTransition, ...]
    topology_kind: TopologyKind
    applicability_evidence: dict[str, Any]
    conflict_evidence: dict[str, Any]

    def __post_init__(self) -> None:
        self.applicability_evidence = plain_copy(self.applicability_evidence)
        self.conflict_evidence = plain_copy(self.conflict_evidence)

    def to_payload(self) -> dict[str, Any]:
        return {
            "operation": _operation_to_payload(self.operation),
            "transitions": [
                transition.to_payload() for transition in self.transitions
            ],
            "topology_kind": self.topology_kind.value,
            "applicability_evidence": _payload(self.applicability_evidence),
            "conflict_evidence": _payload(self.conflict_evidence),
        }


_OPERATION_FIELDS = (
    "rule_id",
    "semantic_group",
    "scope",
    "direction",
    "magnitude",
    "structural_level",
    "reads",
    "writes",
    "conflicts",
    "prerequisites",
    "edit_mode",
    "effect_metric_id",
    "effect_contract_id",
    "before",
    "after",
    "relative_change",
    "requested_magnitude",
    "realized_magnitude",
    "clamped",
    "instruction_tokens",
)
_STATE_FIELDS = frozenset({"rule_id", "state_id", "phrase", "attributes"})
_TRANSITION_FIELDS = frozenset(
    {
        "rule_id",
        "scope",
        "channels",
        "feature_phrase",
        "source_state",
        "target_state",
        "direction",
    }
)
_PLANNED_FIELDS = frozenset(
    {
        "operation",
        "transitions",
        "topology_kind",
        "applicability_evidence",
        "conflict_evidence",
    }
)
_TOPOLOGY_BY_LEVEL = {
    0: TopologyKind.NONE,
    1: TopologyKind.BOUNDARY,
    2: TopologyKind.MEMBERSHIP,
}


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be non-empty text")
    return value


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise TypeError(f"{field_name} must be a JSON string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(value)


def _operation_to_payload(operation: Operation) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in _OPERATION_FIELDS:
        value = getattr(operation, field_name)
        payload[field_name] = (
            list(value) if isinstance(value, tuple) else _payload(value)
        )
    return payload


def _operation_from_payload(value: Any) -> Operation:
    mapping = _mapping(value, frozenset(_OPERATION_FIELDS), "operation")
    for field_name in (
        "rule_id",
        "semantic_group",
        "scope",
        "direction",
        "magnitude",
        "edit_mode",
        "effect_metric_id",
        "effect_contract_id",
    ):
        _text(mapping[field_name], f"operation.{field_name}")
    sequences = {
        field_name: _string_list(
            mapping[field_name],
            f"operation.{field_name}",
        )
        for field_name in ("reads", "writes", "conflicts", "prerequisites")
    }
    for field_name in ("before", "after", "instruction_tokens"):
        if not isinstance(mapping[field_name], Mapping):
            raise TypeError(f"operation.{field_name} must be a mapping")
    structural_level = mapping["structural_level"]
    if (
        type(structural_level) is not int
        or structural_level not in _TOPOLOGY_BY_LEVEL
    ):
        raise ValueError("operation.structural_level must be 0, 1, or 2")
    if not isinstance(mapping["clamped"], bool):
        raise TypeError("operation.clamped must be a boolean")
    for field_name in ("requested_magnitude", "realized_magnitude"):
        optional_text = mapping[field_name]
        if optional_text is not None:
            _text(optional_text, f"operation.{field_name}")
    relative_change = mapping["relative_change"]
    if relative_change is not None and (
        isinstance(relative_change, bool)
        or not isinstance(relative_change, (int, float))
        or not math.isfinite(relative_change)
    ):
        raise TypeError("operation.relative_change must be finite or null")
    write_keys = set(sequences["writes"])
    if (
        set(mapping["before"]) != write_keys
        or set(mapping["after"]) != write_keys
    ):
        raise ValueError("operation.before and operation.after must cover writes")
    if any(
        mapping["before"][path] == mapping["after"][path]
        for path in sequences["writes"]
    ):
        raise ValueError("operation writes must describe value changes")
    return Operation(
        rule_id=mapping["rule_id"],
        semantic_group=mapping["semantic_group"],
        scope=mapping["scope"],
        direction=mapping["direction"],
        magnitude=mapping["magnitude"],
        structural_level=structural_level,
        reads=sequences["reads"],
        writes=sequences["writes"],
        conflicts=sequences["conflicts"],
        prerequisites=sequences["prerequisites"],
        edit_mode=mapping["edit_mode"],
        effect_metric_id=mapping["effect_metric_id"],
        effect_contract_id=mapping["effect_contract_id"],
        before=plain_copy(mapping["before"]),
        after=plain_copy(mapping["after"]),
        relative_change=mapping["relative_change"],
        requested_magnitude=mapping["requested_magnitude"],
        realized_magnitude=mapping["realized_magnitude"],
        clamped=mapping["clamped"],
        instruction_tokens=plain_copy(mapping["instruction_tokens"]),
    )


def _state_from_payload(value: Any) -> SemanticState:
    mapping = _mapping(value, _STATE_FIELDS, "semantic_state")
    if not isinstance(mapping["attributes"], Mapping):
        raise TypeError("semantic_state.attributes must be a mapping")
    for field_name in ("rule_id", "state_id", "phrase"):
        _text(mapping[field_name], f"semantic_state.{field_name}")
    return SemanticState(
        rule_id=mapping["rule_id"],
        state_id=mapping["state_id"],
        phrase=mapping["phrase"],
        attributes=plain_copy(mapping["attributes"]),
    )


def semantic_transition_from_payload(value: Any) -> SemanticTransition:
    mapping = _mapping(value, _TRANSITION_FIELDS, "semantic_transition")
    for field_name in ("rule_id", "scope", "feature_phrase", "direction"):
        _text(mapping[field_name], f"semantic_transition.{field_name}")
    channels = _string_list(
        mapping["channels"],
        "semantic_transition.channels",
    )
    if not channels:
        raise TypeError("semantic_transition.channels must not be empty")
    source_state = _state_from_payload(mapping["source_state"])
    target_state = _state_from_payload(mapping["target_state"])
    if (
        source_state.rule_id != mapping["rule_id"]
        or target_state.rule_id != mapping["rule_id"]
    ):
        raise ValueError("transition state rules must match transition rule")
    if source_state.state_id == target_state.state_id:
        raise ValueError("transition states must describe a change")
    return SemanticTransition(
        rule_id=mapping["rule_id"],
        scope=mapping["scope"],
        channels=channels,
        feature_phrase=mapping["feature_phrase"],
        source_state=source_state,
        target_state=target_state,
        direction=mapping["direction"],
    )


def planned_operation_from_payload(value: Any) -> PlannedOperation:
    mapping = _mapping(value, _PLANNED_FIELDS, "planned_operation")
    transitions = mapping["transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise TypeError("planned_operation.transitions must be a non-empty list")
    for field_name in ("applicability_evidence", "conflict_evidence"):
        if not isinstance(mapping[field_name], Mapping):
            raise TypeError(f"planned_operation.{field_name} must be a mapping")
    try:
        topology_kind = TopologyKind(mapping["topology_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("planned_operation.topology_kind is invalid") from exc
    operation = _operation_from_payload(mapping["operation"])
    decoded_transitions = tuple(
        semantic_transition_from_payload(item) for item in transitions
    )
    if topology_kind is not _TOPOLOGY_BY_LEVEL[operation.structural_level]:
        raise ValueError("topology_kind must match operation.structural_level")
    for transition in decoded_transitions:
        if (
            transition.rule_id != operation.rule_id
            or transition.scope != operation.scope
            or transition.direction != operation.direction
        ):
            raise ValueError("transition identity must match operation")
    return PlannedOperation(
        operation=operation,
        transitions=decoded_transitions,
        topology_kind=topology_kind,
        applicability_evidence=plain_copy(mapping["applicability_evidence"]),
        conflict_evidence=plain_copy(mapping["conflict_evidence"]),
    )


def canonical_operation_signature(operation: PlannedOperation) -> str:
    """Hash the target-inclusive identity while excluding evaluator evidence."""

    return canonical_hash(
        [
            {
                "rule_id": operation.operation.rule_id,
                "scope": operation.operation.scope,
                "direction": operation.operation.direction,
                "writes": list(operation.operation.writes),
                "after": _payload(operation.operation.after),
                "source_states": [
                    transition.source_state.state_id
                    for transition in operation.transitions
                ],
                "target_states": [
                    transition.target_state.state_id
                    for transition in operation.transitions
                ],
            }
        ]
    )
