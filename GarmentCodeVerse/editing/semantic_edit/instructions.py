"""Canonical and deterministic rendering for semantic-edit instructions.

The canonical renderer is a persisted-data contract.  The versioned surface
renderer adds deterministic variation for new datasets without changing the
typed transition payload: every template contains the exact feature,
source-state, and target-state phrases supplied by the planned operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from editing.canonical_io import canonical_json_bytes, sha256_bytes
from editing.models import (
    PlannedOperation,
    SemanticTransition,
    canonical_operation_signature,
    planned_operation_from_payload,
)
from .registry import RULE_ORDER


INSTRUCTION_RENDERER_VERSION = "gcd-specific-instruction-v2"
INSTRUCTION_PROVENANCE_SCHEMA_VERSION = (
    "gcd-specific-instruction-provenance-v1"
)

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DISALLOWED_COMPONENT_PUNCTUATION = frozenset("{}[]().,;:!?")
_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "renderer_version",
        "template_id",
        "selection_sha256",
    }
)
_METADATA_KEYS = frozenset(
    {
        "source_key",
        "operation_signature",
        "operation",
        "specific_instruction",
        "canonical_instruction",
        "instruction_provenance",
    }
)


def _printable_ascii_text(value: Any, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or not value.isascii()
        or any(not 0x20 <= ord(character) <= 0x7E for character in value)
    ):
        raise ValueError(f"{field_name} must be non-empty trimmed printable ASCII")
    return value


def _instruction_component(value: Any, field_name: str) -> str:
    text = _printable_ascii_text(value, field_name)
    if any(character in _DISALLOWED_COMPONENT_PUNCTUATION for character in text):
        raise ValueError(
            f"{field_name} must not contain braces or sentence punctuation"
        )
    return text


def _instruction_sentence(value: Any, field_name: str) -> str:
    text = _printable_ascii_text(value, field_name)
    if (
        not text.endswith(".")
        or text.count(".") != 1
        or text[-2].isspace()
        or "\n" in text
        or "!" in text
        or "?" in text
    ):
        raise ValueError(
            f"{field_name} must be one printable ASCII sentence with exactly "
            "one terminal period"
        )
    return text


_TEMPLATES = (
    ("change_from_to", "Change {feature} from {source} to {target}"),
    ("modify_from_to", "Modify {feature} from {source} to {target}"),
    ("update_from_to", "Update {feature} from {source} to {target}"),
    ("adjust_from_to", "Adjust {feature} from {source} to {target}"),
    ("convert_from_to", "Convert {feature} from {source} to {target}"),
    ("transform_from_to", "Transform {feature} from {source} to {target}"),
    ("for_change", "For {feature}, change from {source} to {target}"),
    ("for_switch", "For {feature}, switch from {source} to {target}"),
)
_TEMPLATES_BY_ID = dict(_TEMPLATES)


@dataclass(frozen=True)
class RenderedInstruction:
    """A rendered instruction plus replayable, versioned provenance."""

    text: str
    canonical_text: str
    template_id: str
    selection_sha256: str

    def __post_init__(self) -> None:
        _instruction_sentence(self.text, "text")
        _instruction_sentence(self.canonical_text, "canonical_text")
        if self.template_id not in _TEMPLATES_BY_ID:
            raise ValueError("template_id is not registered")
        if (
            type(self.selection_sha256) is not str
            or _SHA256_RE.fullmatch(self.selection_sha256) is None
        ):
            raise ValueError("selection_sha256 must be a lowercase SHA-256 value")

    def provenance_payload(self) -> dict[str, str]:
        return {
            "schema_version": INSTRUCTION_PROVENANCE_SCHEMA_VERSION,
            "renderer_version": INSTRUCTION_RENDERER_VERSION,
            "template_id": self.template_id,
            "selection_sha256": self.selection_sha256,
        }


def _validated_source_key(value: Any) -> str:
    return _printable_ascii_text(value, "source_key")


def _validated_signature(value: Any, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 value")
    return value


def _selection_sha256(source_key: str, operation_signature: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "renderer_version": INSTRUCTION_RENDERER_VERSION,
                "source_key": source_key,
                "operation_signature": operation_signature,
            }
        )
    )


def _join_clauses(clauses: list[str]) -> str:
    if not clauses:
        raise ValueError("operation must contain at least one transition")
    normalized = [
        clause if index == 0 else clause[0].lower() + clause[1:]
        for index, clause in enumerate(clauses)
    ]
    if len(normalized) == 1:
        sentence = normalized[0]
    elif len(normalized) == 2:
        sentence = f"{normalized[0]} and {normalized[1]}"
    else:
        sentence = f"{', '.join(normalized[:-1])}, and {normalized[-1]}"
    return sentence + "."


def _clause_components(
    transition: SemanticTransition,
    *,
    qualified_fields: bool,
) -> dict[str, str]:
    if not isinstance(transition, SemanticTransition):
        raise TypeError("transition must be a SemanticTransition")
    fields = (
        (
            "feature",
            transition.feature_phrase,
            "transition.feature_phrase",
        ),
        (
            "source",
            transition.source_state.phrase,
            "transition.source_state.phrase",
        ),
        (
            "target",
            transition.target_state.phrase,
            "transition.target_state.phrase",
        ),
    )
    return {
        name: _instruction_component(
            value,
            field_name if qualified_fields else "instruction component",
        )
        for name, value, field_name in fields
    }


def render_specific_clause(transition: SemanticTransition) -> str:
    """Render one canonical typed transition without terminal punctuation."""

    return _TEMPLATES[0][1].format(
        **_clause_components(transition, qualified_fields=False)
    )


def render_specific_instruction(operation: PlannedOperation) -> str:
    """Render one operation's canonical clauses as a persisted instruction."""

    if not isinstance(operation, PlannedOperation):
        raise TypeError("operation must be a PlannedOperation")
    if operation.operation.rule_id not in RULE_ORDER:
        raise ValueError("operation contains a rule outside the registry")
    clauses = [
        render_specific_clause(transition)
        for transition in operation.transitions
    ]
    return _join_clauses(clauses)


def render_deterministic_specific_instruction(
    operation: PlannedOperation,
    *,
    source_key: str,
    expected_operation_signature: str,
) -> RenderedInstruction:
    """Render one planned operation with deterministic surface variation.

    A planned operation may contain one or more typed transitions.  One template
    is selected for the whole instruction, so clause style remains consistent.
    The exact canonical instruction remains available for audit and replay.
    """

    if not isinstance(operation, PlannedOperation):
        raise TypeError("operation must be a PlannedOperation")
    source_key = _validated_source_key(source_key)
    expected = _validated_signature(
        expected_operation_signature,
        "expected_operation_signature",
    )
    actual = canonical_operation_signature(operation)
    if actual != expected:
        raise ValueError("expected operation signature does not match operation")

    selection_sha256 = _selection_sha256(source_key, actual)
    template_index = int(selection_sha256.removeprefix("sha256:"), 16) % len(
        _TEMPLATES
    )
    template_id, pattern = _TEMPLATES[template_index]
    clauses = [
        pattern.format(
            **_clause_components(transition, qualified_fields=True)
        )
        for transition in operation.transitions
    ]
    text = _join_clauses(clauses)
    canonical_text = render_specific_instruction(operation)
    _instruction_sentence(text, "specific_instruction")
    _instruction_sentence(canonical_text, "canonical_instruction")
    return RenderedInstruction(
        text=text,
        canonical_text=canonical_text,
        template_id=template_id,
        selection_sha256=selection_sha256,
    )


def validate_instruction_payload_shape(
    *,
    specific_instruction: str,
    canonical_instruction: str,
    instruction_provenance: Mapping[str, Any],
) -> None:
    """Validate instruction text and the exact provenance-v1 payload shape."""

    _instruction_sentence(specific_instruction, "specific_instruction")
    _instruction_sentence(canonical_instruction, "canonical_instruction")
    if not isinstance(instruction_provenance, Mapping):
        raise ValueError("instruction_provenance must be a mapping")
    if frozenset(instruction_provenance) != _PROVENANCE_KEYS:
        missing = sorted(_PROVENANCE_KEYS - frozenset(instruction_provenance))
        unknown = sorted(frozenset(instruction_provenance) - _PROVENANCE_KEYS)
        raise ValueError(
            "instruction_provenance keys must be exact; "
            f"missing={missing}, unknown={unknown}"
        )
    if (
        instruction_provenance.get("schema_version")
        != INSTRUCTION_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("instruction provenance schema version is invalid")
    if (
        instruction_provenance.get("renderer_version")
        != INSTRUCTION_RENDERER_VERSION
    ):
        raise ValueError("instruction renderer version is invalid")
    template_id = instruction_provenance.get("template_id")
    if type(template_id) is not str or template_id not in _TEMPLATES_BY_ID:
        raise ValueError("instruction template_id is invalid")
    _validated_signature(
        instruction_provenance.get("selection_sha256"),
        "instruction_provenance.selection_sha256",
    )


def validate_instruction_metadata(
    metadata: Mapping[str, Any],
) -> RenderedInstruction:
    """Strictly replay and validate instruction fields in candidate metadata."""

    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    missing = sorted(_METADATA_KEYS - frozenset(metadata))
    if missing:
        raise ValueError(f"instruction metadata is missing required keys: {missing}")
    try:
        operation = planned_operation_from_payload(metadata["operation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metadata operation is invalid") from exc

    validate_instruction_payload_shape(
        specific_instruction=metadata["specific_instruction"],
        canonical_instruction=metadata["canonical_instruction"],
        instruction_provenance=metadata["instruction_provenance"],
    )
    rendered = render_deterministic_specific_instruction(
        operation,
        source_key=metadata["source_key"],
        expected_operation_signature=metadata["operation_signature"],
    )
    if metadata["specific_instruction"] != rendered.text:
        raise ValueError("specific instruction does not match deterministic replay")
    if metadata["canonical_instruction"] != rendered.canonical_text:
        raise ValueError("canonical instruction does not match operation replay")
    if dict(metadata["instruction_provenance"]) != rendered.provenance_payload():
        raise ValueError("instruction provenance does not match deterministic replay")
    return rendered
