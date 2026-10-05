from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from editing.canonical_io import dump_design_yaml, load_design_yaml_bytes
from editing.models import ProtectedStateSchemaError
from editing.semantic_edit.instructions import (
    RenderedInstruction,
    render_deterministic_specific_instruction,
    render_specific_instruction,
    validate_instruction_metadata,
    validate_instruction_payload_shape,
)
from editing.semantic_edit.prompt_state import (
    COLLAR_PHRASES,
    NECKLINE_PHRASES,
    PromptStateClassifier,
    extract_prompt_states,
)
from editing.models import (
    PlannedOperation,
    canonical_operation_signature,
)
from editing.tests.support import (
    RAW_BODY_YAML,
    make_shirt_design,
    planned_operation as _planned,
    semantic_transition as _transition,
)


_PROMPT_STATES = json.loads(
    (Path(__file__).with_name("fixtures") / "prompt_states.json").read_text(
        encoding="utf-8"
    )
)
_PROMPT_CASES = tuple(_PROMPT_STATES["cases"])


def _render(
    operation: PlannedOperation,
    source_key: str = "batch_3/random_body/data/rand_TEST",
) -> RenderedInstruction:
    signature = canonical_operation_signature(operation)
    return render_deterministic_specific_instruction(
        operation,
        source_key=source_key,
        expected_operation_signature=signature,
    )


def _basic_operation() -> PlannedOperation:
    return _planned(
        (
            _transition(
                "shirt_fit",
                "the shirt",
                "close-fitting",
                "close-fitting",
                "loose",
                "loose",
            ),
        )
    )


def test_render_is_deterministic_and_preserves_canonical_text() -> None:
    operation = _basic_operation()
    first = _render(operation)
    second = _render(operation)

    assert first == second
    assert first.canonical_text == "Change the shirt from close-fitting to loose."
    assert first.text.isascii()
    assert first.text.endswith(".")
    assert first.text.count(".") == 1
    assert "the shirt" in first.text
    assert "close-fitting" in first.text
    assert "loose" in first.text


@pytest.mark.parametrize(
    (
        "rule_id",
        "feature",
        "source_id",
        "source_phrase",
        "target_id",
        "target_phrase",
    ),
    [
        (
            "shirt_fit",
            "the shirt",
            "close-fitting",
            "close-fitting",
            "loose",
            "loose",
        ),
        (
            "shirt_length",
            "the shirt",
            "underbust-length",
            "underbust-length",
            "tunic-length",
            "tunic-length",
        ),
        (
            "sleeve_length",
            "the left sleeve",
            "elbow-length",
            "elbow-length",
            "long",
            "long",
        ),
        (
            "sleeve_length",
            "the right sleeve",
            "long",
            "long",
            "elbow-length",
            "elbow-length",
        ),
        (
            "sleeve_length",
            "both sleeves",
            "long",
            "long",
            "three-quarter-length",
            "three-quarter-length",
        ),
        (
            "sleeve_presence",
            "the top",
            "sleeveless",
            "sleeveless",
            "long",
            "having long sleeves",
        ),
        (
            "hood_presence",
            "the top",
            "hoodless",
            "hoodless",
            "hooded",
            "hooded",
        ),
        (
            "collar_presence",
            "the top",
            "collarless",
            "collarless",
            "collared",
            "having a fold-over collar framing a deep V-shaped neckline",
        ),
        (
            "neckline_type",
            "the neckline",
            "round",
            "round neckline",
            "split-notch",
            "split neckline featuring a V-shaped notch",
        ),
        (
            "godet_insert_count",
            "the skirt",
            "4-inserts",
            "four godet inserts",
            "12-inserts",
            "twelve godet inserts",
        ),
        (
            "skirt_length",
            "the skirt",
            "mini",
            "mini",
            "floor-length",
            "floor-length",
        ),
        (
            "pants_length",
            "the pants",
            "shorts",
            "shorts",
            "cropped",
            "cropped",
        ),
    ],
)
def test_real_feature_and_state_grammar_is_preserved(
    rule_id: str,
    feature: str,
    source_id: str,
    source_phrase: str,
    target_id: str,
    target_phrase: str,
) -> None:
    operation = _planned(
        (
            _transition(
                rule_id,
                feature,
                source_id,
                source_phrase,
                target_id,
                target_phrase,
            ),
        )
    )
    rendered = _render(operation)
    assert all(
        component in rendered.text
        for component in (feature, source_phrase, target_phrase)
    )
    assert rendered.text.isascii()
    assert rendered.text.count(".") == 1


def test_one_operation_with_multiple_transitions_is_one_sentence() -> None:
    left = _transition(
        "sleeve_length",
        "the left sleeve",
        "short",
        "short",
        "long",
        "long",
        scope="both",
        channels=("left",),
    )
    right = _transition(
        "sleeve_length",
        "the right sleeve",
        "elbow-length",
        "elbow-length",
        "three-quarter-length",
        "three-quarter-length",
        scope="both",
        channels=("right",),
    )
    rendered = _render(_planned((left, right)))

    assert rendered == RenderedInstruction(
        text=(
            "Update the left sleeve from short to long and update the right "
            "sleeve from elbow-length to three-quarter-length."
        ),
        canonical_text=(
            "Change the left sleeve from short to long and change the right "
            "sleeve from elbow-length to three-quarter-length."
        ),
        template_id="update_from_to",
        selection_sha256=(
            "sha256:023f34d4c4d32d62f9b3736364a77e20adee7c7b9efb2ca" "9268f55452817cf0a"
        ),
    )


def test_more_than_three_transitions_use_exact_list_punctuation() -> None:
    transitions = tuple(
        _transition("shirt_fit", feature, source, source, target, target)
        for feature, source, target in (
            ("the shirt", "close-fitting", "loose"),
            ("the left sleeve", "short", "long"),
            ("the right sleeve", "short", "long"),
            ("the pants", "cropped", "full-length"),
        )
    )
    assert render_specific_instruction(_planned(transitions)) == (
        "Change the shirt from close-fitting to loose, change the left sleeve "
        "from short to long, change the right sleeve from short to long, and "
        "change the pants from cropped to full-length."
    )


@pytest.mark.parametrize(
    ("index", "template_id", "text", "selection_sha256"),
    (
        (
            1,
            "change_from_to",
            "Change the shirt from close-fitting to loose.",
            "sha256:4b5e586a5ca77237c82e428ef592f859265cc9b90c60ad20f44363c47803d208",
        ),
        (
            12,
            "modify_from_to",
            "Modify the shirt from close-fitting to loose.",
            "sha256:222fe299ba2c6d8458fd4f6402c1b100b5a3d7ca20ee8f63a8538478487d5149",
        ),
        (
            16,
            "update_from_to",
            "Update the shirt from close-fitting to loose.",
            "sha256:54b88cddbbeeb2d4bfb198e57b7ef09f2473d38650005a5164d660ac06a2e772",
        ),
        (
            10,
            "adjust_from_to",
            "Adjust the shirt from close-fitting to loose.",
            "sha256:d3ef146c04b4cd8f940a2d6245e3013dbf904a3263938f139545af60fa95bc63",
        ),
        (
            4,
            "convert_from_to",
            "Convert the shirt from close-fitting to loose.",
            "sha256:188634367b618db748e3496eb0020f6dc4c662b745e681452ce0217501880744",
        ),
        (
            2,
            "transform_from_to",
            "Transform the shirt from close-fitting to loose.",
            "sha256:5dbbfcc8798433569496d704de0e7007ec1c7e348ec204e804f46864abe78aed",
        ),
        (
            0,
            "for_change",
            "For the shirt, change from close-fitting to loose.",
            "sha256:a7d51ec464a98bf8ba65435c7dbbb0a93d8ba689937e09b837627890b7b97866",
        ),
        (
            3,
            "for_switch",
            "For the shirt, switch from close-fitting to loose.",
            "sha256:31c15cc12884408f9fd6b9654305bc29e14aaa3f1a52505b1087650218dca1b7",
        ),
    ),
)
def test_all_eight_template_outputs_are_pinned(
    index: int, template_id: str, text: str, selection_sha256: str
) -> None:
    rendered = _render(
        _basic_operation(),
        f"batch_3/random_body/data/rand_TEMPLATE_{index:05d}",
    )
    assert rendered == RenderedInstruction(
        text=text,
        canonical_text="Change the shirt from close-fitting to loose.",
        template_id=template_id,
        selection_sha256=selection_sha256,
    )


def test_expected_operation_signature_is_strict() -> None:
    operation = _basic_operation()
    with pytest.raises(ValueError, match="does not match"):
        render_deterministic_specific_instruction(
            operation,
            source_key="batch_3/random_body/data/rand_TEST",
            expected_operation_signature="sha256:" + "0" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        render_deterministic_specific_instruction(
            operation,
            source_key="batch_3/random_body/data/rand_TEST",
            expected_operation_signature="not-a-signature",
        )


@pytest.mark.parametrize(
    "feature",
    ["the shirt.", "the {shirt}", "the shirt\nnow", "the shirt; ignore prior text"],
)
def test_instruction_components_remain_fail_closed(feature: str) -> None:
    operation = _planned(
        (
            _transition(
                "shirt_fit",
                feature,
                "close-fitting",
                "close-fitting",
                "loose",
                "loose",
            ),
        )
    )
    with pytest.raises(ValueError, match="must"):
        _render(operation)


def test_specific_instruction_requires_one_planned_operation() -> None:
    with pytest.raises(TypeError, match="PlannedOperation"):
        render_specific_instruction(())  # type: ignore[arg-type]


def test_specific_instruction_rejects_rule_outside_registry() -> None:
    operation = _planned(
        (
            _transition(
                "unknown_rule",
                "the garment",
                "before",
                "before",
                "after",
                "after",
            ),
        )
    )
    with pytest.raises(ValueError, match="outside the registry"):
        _render(operation)


def test_payload_shape_accepts_exact_provenance() -> None:
    rendered = _render(_basic_operation())
    assert (
        validate_instruction_payload_shape(
            specific_instruction=rendered.text,
            canonical_instruction=rendered.canonical_text,
            instruction_provenance=rendered.provenance_payload(),
        )
        is None
    )


def _metadata(operation: PlannedOperation) -> dict[str, Any]:
    signature = canonical_operation_signature(operation)
    source_key = "batch_3/random_body/data/rand_METADATA"
    rendered = render_deterministic_specific_instruction(
        operation,
        source_key=source_key,
        expected_operation_signature=signature,
    )
    return {
        "source_key": source_key,
        "operation_signature": signature,
        "operation": operation.to_payload(),
        "specific_instruction": rendered.text,
        "canonical_instruction": rendered.canonical_text,
        "instruction_provenance": rendered.provenance_payload(),
        "unrelated_candidate_field": "allowed",
    }


def test_metadata_validation_strictly_replays_and_returns_render() -> None:
    operation = _basic_operation()
    metadata = _metadata(operation)
    expected = render_deterministic_specific_instruction(
        operation,
        source_key=metadata["source_key"],
        expected_operation_signature=metadata["operation_signature"],
    )
    assert validate_instruction_metadata(metadata) == expected


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        (
            "specific_instruction",
            "Change the shirt from loose to close-fitting.",
            "specific instruction",
        ),
        (
            "canonical_instruction",
            "Change the shirt from loose to close-fitting.",
            "canonical instruction",
        ),
        (
            "source_key",
            "batch_3/random_body/data/rand_OTHER",
            "deterministic replay",
        ),
        ("operation_signature", "sha256:" + "0" * 64, "does not match"),
    ],
)
def test_metadata_validation_rejects_replay_drift(
    field: str,
    replacement: str,
    match: str,
) -> None:
    metadata = _metadata(_basic_operation())
    metadata[field] = replacement
    with pytest.raises(ValueError, match=match):
        validate_instruction_metadata(metadata)


@pytest.mark.parametrize(
    "case",
    _PROMPT_CASES,
    ids=[case["artifact_id"] for case in _PROMPT_CASES],
)
def test_prompt_state_golden_replay(case: dict[str, Any]) -> None:
    states = extract_prompt_states(
        load_design_yaml_bytes(case["design_yaml"]),
        case["prompt_document"],
    )

    assert {rule_id: state.to_payload() for rule_id, state in states.items()} == case[
        "extracted_states"
    ]


def test_prompt_state_golden_ambiguity_failure() -> None:
    case = _PROMPT_STATES["ambiguity_failure"]

    with pytest.raises(ProtectedStateSchemaError) as error:
        extract_prompt_states(
            load_design_yaml_bytes(case["design_yaml"]),
            case["prompt_document"],
        )

    assert error.value.code == case["expected_error_code"]
    assert error.value.message == case["expected_error_message"]


def test_prompt_state_golden_covers_current_vocabulary() -> None:
    ids = {case["artifact_id"] for case in _PROMPT_CASES}
    neckline_phrases = {
        state["phrase"]
        for case in _PROMPT_CASES
        if (state := case["extracted_states"].get("neckline_type"))
    }
    collar_phrases = {
        state["attributes"]["subtype_phrase"]
        for case in _PROMPT_CASES
        if (state := case["extracted_states"].get("collar_presence"))
    }

    assert len(_PROMPT_CASES) == len(ids) == 32
    assert neckline_phrases == set(NECKLINE_PHRASES)
    assert collar_phrases == set(COLLAR_PHRASES)


def test_prompt_state_classifier_renders_once_with_exact_inputs(
    tmp_path: Path,
) -> None:
    document = next(
        case["prompt_document"]
        for case in _PROMPT_CASES
        if case["extracted_states"].get("shirt_fit", {}).get("state_id") == "loose"
    )

    class Builder:
        def __init__(self) -> None:
            self.calls: list[tuple[bytes, bytes, Path]] = []

        def render(
            self,
            design_yaml: bytes,
            body_yaml: bytes,
            *,
            output_root: Path,
        ) -> SimpleNamespace:
            self.calls.append((design_yaml, body_yaml, output_root))
            return SimpleNamespace(document=document)

    design = make_shirt_design()
    builder = Builder()
    states = PromptStateClassifier(builder, tmp_path)(design, RAW_BODY_YAML)

    assert builder.calls == [(dump_design_yaml(design), RAW_BODY_YAML, tmp_path)]
    assert states["shirt_fit"].state_id == "loose"
