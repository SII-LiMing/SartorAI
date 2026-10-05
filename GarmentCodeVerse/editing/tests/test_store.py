from __future__ import annotations

import json
from pathlib import Path

import pytest

from editing.canonical_io import canonical_json_bytes
from editing.publication import DatasetWriteError
from editing.tests.support import (
    instruction_fixture,
)
from editing.semantic_edit.store import (
    AcceptedProjection,
    PAIR_FORMAT,
    RUN_FORMAT,
    SimpleEditStore,
    SimpleEditStoreError,
    candidate_artifact_names,
)


CANDIDATE_ID = "rand_TEST123__edit_000"
INSTRUCTION_FIELDS = instruction_fixture()
PROMPT_BYTES = b'{"prompts":[]}\n'


def _run_manifest() -> dict:
    return {
        "format": RUN_FORMAT,
        "batch_id": "batch_0",
        "slots": [],
    }


def _candidate_artifacts() -> dict[str, bytes]:
    return {
        "design_params.yaml": b"design: {}\n",
        "body_measurements.yaml": b"body: {}\n",
        f"{CANDIDATE_ID}_specification.json": b"{}\n",
        f"{CANDIDATE_ID}_pattern.png": b"\x89PNG\r\n\x1a\npayload",
        f"{CANDIDATE_ID}_pattern.svg": b"<svg></svg>\n",
    }


def _candidate_metadata() -> dict:
    return {
        "format": RUN_FORMAT,
        "candidate_id": CANDIDATE_ID,
        "slot_id": "slot_000",
        **INSTRUCTION_FIELDS,
    }


def _final_pair() -> dict:
    return {
        "format": PAIR_FORMAT,
        "slot_id": "slot_000",
        "target_id": CANDIDATE_ID,
        "target_simulation_status": "pass",
        **INSTRUCTION_FIELDS,
    }


def _regular_tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and not path.is_symlink()
    }


def _store(tmp_path: Path) -> SimpleEditStore:
    return SimpleEditStore.create(tmp_path / "output", _run_manifest())


def _published_store(tmp_path: Path) -> tuple[SimpleEditStore, Path]:
    store = _store(tmp_path)
    target = store.publish_candidate(
        CANDIDATE_ID,
        artifacts=_candidate_artifacts(),
        metadata=_candidate_metadata(),
    )
    return store, target


def _prepare_acceptance(
    store: SimpleEditStore,
    pair: dict,
    *,
    product_png_bytes: bytes | None = None,
):
    store.initialize_slot_state(
        slot_id="slot_000",
        source_key="batch_0/random_body/data/rand_TEST123",
        source_sample_id="rand_TEST123",
    )
    state = store.load_slot_state("slot_000")
    state["attempts"].append(
        {
            "candidate_id": CANDIDATE_ID,
            "status": "in_simulation",
        }
    )
    store.save_slot_state(state)
    random_body = store.simulation_workspace_root / "random_body"
    random_body.mkdir(parents=True)
    product = random_body / CANDIDATE_ID
    product.mkdir()
    artifacts = _candidate_artifacts()
    (product / f"{CANDIDATE_ID}_pattern.png").write_bytes(
        artifacts[f"{CANDIDATE_ID}_pattern.png"]
        if product_png_bytes is None
        else product_png_bytes
    )
    (product / f"{CANDIDATE_ID}_pattern.svg").write_bytes(
        artifacts[f"{CANDIDATE_ID}_pattern.svg"]
    )
    accepted = AcceptedProjection(
        prompt_bytes=PROMPT_BYTES,
        pair_record=pair,
        candidate_metadata=store.candidate_metadata(CANDIDATE_ID),
        png_bytes=artifacts[f"{CANDIDATE_ID}_pattern.png"],
        svg_bytes=artifacts[f"{CANDIDATE_ID}_pattern.svg"],
    )
    return store.prepare_projection_transition(
        candidate_id=CANDIDATE_ID,
        workspace_root=store.simulation_workspace_root,
        state=state,
        attempt_index=0,
        outcome="accepted",
        failure_message=None,
        accepted=accepted,
    )


def test_store_formats_are_current() -> None:
    assert RUN_FORMAT == "semantic-edit-run"
    assert PAIR_FORMAT == "semantic-edit-pair"


def test_source_is_snapshotted_once_and_reopened_without_original_tree(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    design = b"design: {x: 1}\n"
    body = b"body: {body_sample: body.obj}\n"
    store.snapshot_source(
        "rand_TEST123",
        design_bytes=design,
        body_bytes=body,
        specification_bytes=b"{}\n",
    )
    reopened = SimpleEditStore.open(store.root)
    assert reopened.source_snapshot("rand_TEST123") == (design, body)
    assert reopened.patterns_vis_root.is_dir()
    assert reopened.renders_root.is_dir()


def test_slot_initialization_is_idempotent_after_attempts_exist(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    identity = {
        "slot_id": "slot_000",
        "source_key": "batch_0/random_body/data/rand_TEST123",
        "source_sample_id": "rand_TEST123",
    }
    store.initialize_slot_state(**identity)
    state = store.load_slot_state("slot_000")
    state["attempts"].append(
        {"candidate_id": CANDIDATE_ID, "status": "simulation_failed"}
    )
    store.save_slot_state(state)
    store.initialize_slot_state(**identity)
    assert store.load_slot_state("slot_000")["attempts"] == state["attempts"]


def test_candidate_is_exact_five_then_finalized_is_exact_seven(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.initialize_slot_state(
        slot_id="slot_000",
        source_key="batch_0/random_body/data/rand_TEST123",
        source_sample_id="rand_TEST123",
    )
    target = store.publish_candidate(
        CANDIDATE_ID,
        artifacts=_candidate_artifacts(),
        metadata=_candidate_metadata(),
    )
    assert frozenset(path.name for path in target.iterdir()) == (
        candidate_artifact_names(CANDIDATE_ID)
    )
    pair = _final_pair()
    transition = _prepare_acceptance(store, pair)
    store.commit_projection_transition(transition)
    finalized = store.data_root / CANDIDATE_ID
    assert frozenset(path.name for path in finalized.iterdir()) == frozenset(
        {
            "design_params.yaml",
            "body_measurements.yaml",
            f"{CANDIDATE_ID}_specification.json",
            f"{CANDIDATE_ID}_pattern.png",
            f"{CANDIDATE_ID}_pattern.svg",
            f"{CANDIDATE_ID}_prompts.json",
            f"{CANDIDATE_ID}_edit_metadata.json",
        }
    )


def test_publish_candidate_is_write_once(tmp_path: Path) -> None:
    store, target = _published_store(tmp_path)
    before = _regular_tree_bytes(target)
    with pytest.raises(FileExistsError):
        store.publish_candidate(
            CANDIDATE_ID,
            artifacts=_candidate_artifacts(),
            metadata=_candidate_metadata(),
        )
    assert _regular_tree_bytes(target) == before


def test_finalize_adds_two_files_once_without_replay(tmp_path: Path) -> None:
    store, _ = _published_store(tmp_path)
    transition = _prepare_acceptance(store, _final_pair())
    store.commit_projection_transition(transition)
    target = store.data_root / CANDIDATE_ID
    before = _regular_tree_bytes(target)
    with pytest.raises((FileExistsError, DatasetWriteError)):
        store.commit_projection_transition(transition)
    assert _regular_tree_bytes(target) == before


def test_finalize_rejects_simulation_pattern_drift_before_commit(
    tmp_path: Path,
) -> None:
    store, _ = _published_store(tmp_path)

    with pytest.raises(
        SimpleEditStoreError,
        match="pattern PNG.*differs from pending candidate",
    ):
        _prepare_acceptance(
            store,
            _final_pair(),
            product_png_bytes=b"different PNG bytes",
        )

    state = store.load_slot_state("slot_000")
    assert state["accepted_target_id"] is None
    assert state["attempts"][0]["status"] == "in_simulation"
    assert not (store.body_root / CANDIDATE_ID).exists()
    assert store.simulation_workspace_root.is_dir()


def test_accepted_jsonl_contains_only_slot_marked_accepted(
    tmp_path: Path,
) -> None:
    store, _ = _published_store(tmp_path)
    pair = _final_pair()
    before = store.rebuild_accepted_pairs()
    assert before.read_bytes() == b""
    transition = _prepare_acceptance(store, pair)
    store.commit_projection_transition(transition)
    state = store.load_slot_state("slot_000")
    assert state["format"] == RUN_FORMAT
    accepted = store.rebuild_accepted_pairs()
    rows = [json.loads(line) for line in accepted.read_text().splitlines()]
    assert rows == [pair]
    assert accepted.read_bytes() == canonical_json_bytes(pair) + b"\n"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("semantic_drift", "instruction payload"),
        ("replacement_contract", "differs from candidate"),
    ),
)
def test_finalize_candidate_rejects_instruction_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    store, _ = _published_store(tmp_path)
    pair = _final_pair()
    if mutation == "semantic_drift":
        pair["specific_instruction"] = "Update the shirt from cropped to knee-length."
    else:
        pair.update(
            instruction_fixture(
                source_key="batch_0/random_body/data/rand_OTHER",
            )
        )
    with pytest.raises(SimpleEditStoreError, match=message):
        _prepare_acceptance(store, pair)
