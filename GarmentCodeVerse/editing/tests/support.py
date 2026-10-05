"""Shared semantic-edit test fixtures."""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any, Callable, Mapping

from editing.semantic_edit.prompt_state import resolve_active_paths
from editing.canonical_io import (
    dump_yaml_document,
    load_design_yaml_bytes,
)
from editing.models import (
    Operation,
    PlannedOperation,
    RuleContext,
    SemanticState,
    SemanticTransition,
    TopologyKind,
    canonical_operation_signature,
)
from editing.semantic_edit.instructions import (
    render_deterministic_specific_instruction,
)

ROOT = Path(__file__).resolve().parents[2]
BODY = {
    "height": 170.0,
    "bust": 100.0,
    "waist": 70.0,
    "hips": 90.0,
    "waist_line": 40.0,
    "hips_line": 20.0,
    "_leg_length": 80.0,
    "_waist_level": 100.0,
    "arm_length": 60.0,
}
RAW_BODY_YAML = dump_yaml_document({"body": BODY})

_SHIRT_DESIGN_VALUES = {
    "design.meta.upper.v": "Shirt",
    "design.meta.bottom.v": None,
    "design.meta.wb.v": None,
    "design.left.enable_asym.v": False,
    "design.shirt.strapless.v": False,
    "design.collar.component.style.v": None,
}


@cache
def _base_design() -> dict[str, Any]:
    return load_design_yaml_bytes(
        (ROOT / "assets/design_params/design_beautiful.yaml").read_bytes()
    )


def set_design_value(
    design: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    """Replace one existing value in a mutable unwrapped design."""

    parts = dotted_path.split(".")
    if parts[0] == "design":
        parts = parts[1:]
    current = design
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def make_design(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached default design with the given literal replacements."""

    design = deepcopy(_base_design())
    for path, value in values.items():
        set_design_value(design, path, value)
    return design


def make_shirt_design(
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shared symmetric-Shirt literal used by rule/state tests."""

    return make_design({**_SHIRT_DESIGN_VALUES, **(values or {})})


def make_rule_context(
    design: Mapping[str, Any],
    *,
    prompt_classifier: Callable[[dict[str, Any], bytes], Mapping[str, SemanticState]],
    prompt_states: Mapping[str, SemanticState],
    constructor_facts: Mapping[str, Any],
    body: Mapping[str, Any] = BODY,
    raw_body_yaml: bytes = RAW_BODY_YAML,
) -> RuleContext:
    """Build the plain shared context after rule-specific state setup."""

    plain_design = deepcopy(dict(design))

    def classify_prompt(
        target_design: dict[str, Any],
        target_body_yaml: bytes,
    ) -> dict[str, Any]:
        result = prompt_classifier(target_design, target_body_yaml)
        return dict(result) if isinstance(result, Mapping) else result

    return RuleContext(
        design=plain_design,
        body=deepcopy(dict(body)),
        raw_body_yaml=raw_body_yaml,
        active_paths=resolve_active_paths(plain_design),
        constructor_facts=deepcopy(dict(constructor_facts)),
        prompt_states=dict(prompt_states),
        classify_prompt=classify_prompt,
    )


def semantic_state(rule_id: str, state_id: str, phrase: str) -> SemanticState:
    """Build a literal semantic state for instruction tests."""

    return SemanticState(
        rule_id=rule_id,
        state_id=state_id,
        phrase=phrase,
        attributes={},
    )


def semantic_transition(
    rule_id: str,
    feature: str,
    source_id: str,
    source_phrase: str,
    target_id: str,
    target_phrase: str,
    *,
    scope: str = "garment",
    channels: tuple[str, ...] = ("garment",),
    direction: str = "increase",
) -> SemanticTransition:
    """Build a literal typed transition for instruction tests."""

    return SemanticTransition(
        rule_id=rule_id,
        scope=scope,
        channels=channels,
        feature_phrase=feature,
        source_state=semantic_state(rule_id, source_id, source_phrase),
        target_state=semantic_state(rule_id, target_id, target_phrase),
        direction=direction,
    )


def planned_operation(
    transitions: tuple[SemanticTransition, ...],
    *,
    path: str | None = None,
    before: object = 0,
    after: object = 1,
) -> PlannedOperation:
    """Build the minimal operation shared by instruction fixtures."""

    first = transitions[0]
    path = path or f"design.{first.rule_id}.value.v"
    return PlannedOperation(
        operation=Operation(
            rule_id=first.rule_id,
            semantic_group=first.rule_id,
            scope=first.scope,
            direction=first.direction,
            magnitude="specific",
            structural_level=0,
            reads=(path,),
            writes=(path,),
            conflicts=(),
            prerequisites=(),
            edit_mode="replace",
            effect_metric_id="semantic-state",
            effect_contract_id="state-change",
            before={path: before},
            after={path: after},
            requested_magnitude="specific",
            realized_magnitude="specific",
            instruction_tokens={},
        ),
        transitions=transitions,
        topology_kind=TopologyKind.NONE,
        applicability_evidence={"applicable": True},
        conflict_evidence={"conflict_free": True},
    )


def instruction_fixture(
    *,
    source_key: str = "batch_0/random_body/data/rand_TEST123",
) -> dict[str, object]:
    """Return one complete, valid instruction record for store tests."""

    rule_id = "shirt_length"
    path = "design.shirt.length.v"
    transition = semantic_transition(
        rule_id,
        "the shirt",
        "cropped",
        "cropped",
        "hip-length",
        "hip-length",
    )
    planned = planned_operation(
        transitions=(transition,),
        path=path,
        before=0.8,
        after=1.2,
    )
    signature = canonical_operation_signature(planned)
    rendered = render_deterministic_specific_instruction(
        planned,
        source_key=source_key,
        expected_operation_signature=signature,
    )
    return {
        "source_key": source_key,
        "operation_signature": signature,
        "operation": planned.to_payload(),
        "specific_instruction": rendered.text,
        "canonical_instruction": rendered.canonical_text,
        "instruction_provenance": rendered.provenance_payload(),
    }
