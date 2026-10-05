"""Real dependencies and the two external filesystem boundaries."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from editing.canonical_io import (
    canonical_json_bytes,
    dump_design_yaml,
    dump_yaml_document,
    load_body_yaml_bytes,
    load_yaml_bytes,
    sha256_bytes,
)
from editing.semantic_edit.construction import (
    ConstructionDependencies,
    InvalidTargetSpecification,
    build_target_pattern,
    preserve_unchanged_panels,
)
from editing.semantic_edit.prompt import PromptBuilder
from editing.semantic_edit.store import (
    RUN_FORMAT,
    SimpleEditStore,
    candidate_artifact_names,
)
from editing.simulation_workspace import (
    load_filter_projection,
    load_simulation_workspace,
    prepare_simulation_workspace,
    remove_simulation_workspace,
)
from editing.tests.support import instruction_fixture, make_shirt_design


ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_ID = "rand_TEST123__edit_000"
_SECOND_ID = "rand_TEST456__edit_000"


def _triangle_panel() -> dict[str, Any]:
    return {
        "edges": [
            {"endpoints": [0, 1]},
            {"endpoints": [1, 2]},
            {"endpoints": [2, 0]},
        ],
        "vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    }


def _valid_specification() -> dict[str, Any]:
    return {
        "pattern": {
            "panels": {
                "front": _triangle_panel(),
                "back": _triangle_panel(),
            },
            "stitches": [
                [
                    {"panel": "front", "edge": 0},
                    {"panel": "back", "edge": 0},
                ]
            ],
            "panel_order": ["front", "back"],
        },
        "properties": {"units_in_meter": 100.0},
    }


class _StubBody:
    def load_from_dict(self, value: dict[str, Any]) -> None:
        value["builder_only"] = True


class _StubGarment:
    def __init__(
        self,
        name: str,
        body: _StubBody,
        design: dict[str, Any],
    ) -> None:
        assert name == "target"
        design["builder_only"] = True

    def assert_non_empty(self) -> None:
        pass

    def assert_total_length(self) -> None:
        pass

    def is_self_intersecting(self) -> bool:
        return False

    def assembly(self) -> Mapping[str, Any]:
        return _valid_specification()


def test_stub_construction_detaches_inputs_and_returns_canonical_data() -> None:
    design = {"target": {"length": 0.5}}
    original = deepcopy(design)
    body_yaml = dump_yaml_document({"body": {"height": 170.0}})
    dependencies = ConstructionDependencies(
        body_parameters_cls=_StubBody,
        meta_garment_cls=_StubGarment,
        domain_errors=(ValueError,),
        validate_design=lambda _design: None,
    )

    result = build_target_pattern(
        design,
        body_yaml,
        "target",
        dependencies=dependencies,
    )

    assert result.success
    assert design == original
    assert result.specification_json_bytes == canonical_json_bytes(result.specification)
    assert result.specification_sha256 == sha256_bytes(result.specification_json_bytes)


def test_default_garmentcode_construction_smoke() -> None:
    """Exercise the production dependency loader, not injected doubles."""

    design = make_shirt_design({"design.shirt.length.v": 1.5})
    result = build_target_pattern(
        design,
        (ROOT / "assets/bodies/mean_all.yaml").read_bytes(),
        "target",
    )

    assert result.success, (result.failure_code, result.failure_message)
    assert result.specification is not None
    assert result.specification["pattern"]["panels"]
    assert result.specification["pattern"]["stitches"]


def _preservation_specs():
    source = _valid_specification()
    for panel in source["pattern"]["panels"].values():
        panel.update(translation=[0.0, 10.0, 0.0], rotation=[0.0, 0.0, 0.0])
    source["pattern"]["panels"]["back"]["edges"][1]["curvature"] = [0.5, 0.1]
    baseline = deepcopy(source)
    panel = baseline["pattern"]["panels"]["back"]
    panel["vertices"][1][0] += 0.005
    panel["edges"][1]["curvature"][1] += 0.001
    panel["translation"][1] += 0.05
    panel["rotation"][0] += 0.01
    target = deepcopy(baseline)
    target["pattern"]["panels"]["front"]["vertices"][2][1] += 1.0
    return source, baseline, target


def test_preservation_restores_full_geometry_and_pose_without_mutating_inputs():
    source, baseline, target = _preservation_specs()
    inputs = deepcopy((source, baseline, target))
    # Constructor roundoff must not make an unrelated panel look edited.
    target["pattern"]["panels"]["back"]["vertices"][0][0] += 1e-12
    expected_target = deepcopy(target)
    result, names = preserve_unchanged_panels(source, baseline, target)
    assert names == ("back",)
    assert result.specification["pattern"]["panels"]["back"] == source["pattern"]["panels"]["back"]
    assert result.specification["pattern"]["panels"]["front"] == target["pattern"]["panels"]["front"]
    assert result.specification["pattern"]["stitches"] == target["pattern"]["stitches"]
    assert (source, baseline) == inputs[:2]
    assert target == expected_target
    assert result.specification_json_bytes == canonical_json_bytes(result.specification)
    assert result.specification_sha256 == sha256_bytes(result.specification_json_bytes)


def test_no_edit_restores_all_original_panels():
    source, baseline, _ = _preservation_specs()
    result, names = preserve_unchanged_panels(source, baseline, baseline)
    assert set(names) == set(source["pattern"]["panels"])
    assert result.specification == source


@pytest.mark.parametrize("change", ["pose", "curve", "add_remove"])
def test_preservation_keeps_real_dependencies_and_topology_changes(change):
    source, baseline, target = _preservation_specs()
    if change == "pose":
        target["pattern"]["panels"]["back"]["translation"][1] += 3.0
    elif change == "curve":
        target["pattern"]["panels"]["back"]["edges"][1]["curvature"][1] += 0.05
    else:
        pattern = target["pattern"]
        pattern["panels"]["sleeve"] = pattern["panels"].pop("back")
        pattern["panel_order"] = ["front", "sleeve"]
        pattern["stitches"][0][1]["panel"] = "sleeve"
    result, names = preserve_unchanged_panels(source, baseline, target)
    assert names == ()
    assert result.specification == target


def test_preservation_rejects_incompatible_original_edge_indices():
    source, baseline, target = _preservation_specs()
    edges = source["pattern"]["panels"]["back"]["edges"]
    edges[:] = edges[1:] + edges[:1]
    with pytest.raises(InvalidTargetSpecification, match="edge topology"):
        preserve_unchanged_panels(source, baseline, target)


def test_preservation_revalidates_final_stitch_references():
    source, baseline, target = _preservation_specs()
    target["pattern"]["stitches"][0][1]["edge"] = 99
    with pytest.raises(InvalidTargetSpecification, match="missing edge"):
        preserve_unchanged_panels(source, baseline, target)


def _stub_prompt_source(document: object) -> bytes:
    design = b"design: \xce\xb1\n"
    body = b"body: \xce\xb2\n"
    return (
        "from pathlib import Path\n"
        "def parse_design_yaml(path):\n"
        "    path = Path(path)\n"
        "    assert path.parent.name == 'case'\n"
        "    assert path.name == 'case_design_params.yaml'\n"
        f"    assert path.read_bytes() == {design!r}\n"
        "    body = path.with_name('case_body_measurements.yaml')\n"
        f"    assert body.read_bytes() == {body!r}\n"
        f"    return {document!r}\n"
    ).encode("utf-8")


def test_prompt_builder_uses_paired_yaml_and_cleans_temporary_input(
    tmp_path: Path,
) -> None:
    document = {
        "sample_prompt": "蓝色衬衫",
        "body_prompt": "成人体型",
        "parts": {"upper": {"part_prompt": "圆领衬衫"}},
    }
    source = _stub_prompt_source(document)
    builder_path = tmp_path / "build_prompts.py"
    builder_path.write_bytes(source)
    builder = PromptBuilder.load(
        builder_path,
        expected_sha256=sha256_bytes(source),
    )
    output_root = tmp_path / "output"
    output_root.mkdir()

    artifact = builder.render(
        b"design: \xce\xb1\n",
        b"body: \xce\xb2\n",
        output_root=output_root,
    )

    assert json.loads(artifact.json_bytes) == document
    assert list(output_root.iterdir()) == []


def _real_prompt_builder_path() -> Path:
    relative = Path("gen_garment_texture/build_prompt_split_match/build_prompts.py")
    candidates = (
        ROOT / relative,
        ROOT.parent / relative,
        Path("/amax/liming/projects/Reweaver2-Code-upload") / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("external Reweaver2 prompt builder is not in this checkout")


def test_real_prompt_builder_smoke(tmp_path: Path) -> None:
    """Render through the approved Reweaver2 parser when it is available."""

    body = load_body_yaml_bytes((ROOT / "assets/bodies/mean_all.yaml").read_bytes())
    body["_leg_length"] = (
        body["height"] - body["head_l"] - body["waist_line"] - body["hips_line"]
    )
    artifact = PromptBuilder.load(_real_prompt_builder_path()).render(
        dump_design_yaml(make_shirt_design({"design.shirt.length.v": 1.5})),
        dump_yaml_document({"body": body}),
        output_root=tmp_path,
    )

    assert isinstance(artifact.document, dict)
    assert json.loads(artifact.json_bytes) == artifact.document
    assert artifact.document["sample_prompt"]
    assert artifact.document["body_prompt"]


def _workspace_store(tmp_path: Path) -> SimpleEditStore:
    return SimpleEditStore.create(
        tmp_path / "output",
        {"format": RUN_FORMAT, "batch_id": "batch_0", "slots": []},
    )


def _publish_candidate(store: SimpleEditStore, candidate_id: str) -> None:
    source_id = candidate_id.split("__edit_", 1)[0]
    store.publish_candidate(
        candidate_id,
        artifacts={
            "design_params.yaml": b"design: {}\n",
            "body_measurements.yaml": b"body: {}\n",
            f"{candidate_id}_specification.json": (
                b'{"pattern":{"panels":{"front":{"edges":[]}}}}\n'
            ),
            f"{candidate_id}_pattern.png": b"\x89PNG\r\n\x1a\nfixture",
            f"{candidate_id}_pattern.svg": b"<svg></svg>\n",
        },
        metadata={
            "format": RUN_FORMAT,
            "candidate_id": candidate_id,
            "slot_id": "slot_000",
            **instruction_fixture(source_key=f"batch_0/random_body/data/{source_id}"),
        },
    )


def _pending_candidate(
    store: SimpleEditStore,
    candidate_id: str,
) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "target_id": candidate_id,
        "target_path": f"{store.batch_id}/random_body/data/{candidate_id}",
        "target_simulation_status": "pending",
    }


def _filter_report(
    workspace_root: Path,
    candidate_ids: tuple[str, ...],
    *,
    passed: list[str],
    failed: list[str],
) -> Path:
    path = workspace_root / "random_body" / "filter_fail_report_full.yaml"
    path.write_bytes(
        dump_yaml_document(
            {
                "schema_version": 1,
                "report_type": "filter_fail_report_full",
                "gcd_dir": workspace_root.as_posix(),
                "body_type": "random_body",
                "body_root": (workspace_root / "random_body").as_posix(),
                "dataset_properties": (
                    workspace_root
                    / "random_body"
                    / "dataset_properties_random_body.yaml"
                ).as_posix(),
                "parameters": {"no_delete": True},
                "summary": {},
                "case_names": list(candidate_ids),
                "passed_cases": passed,
                "final_failed_cases": failed,
                "final_fail_reasons": {
                    candidate_id: ["fixture_failure"] for candidate_id in failed
                },
            }
        )
    )
    return path


def test_simulation_workspace_stages_exact_five_files_and_reopens(
    tmp_path: Path,
) -> None:
    store = _workspace_store(tmp_path)
    _publish_candidate(store, _CANDIDATE_ID)
    workspace = prepare_simulation_workspace(
        store.root,
        [_pending_candidate(store, _CANDIDATE_ID)],
        global_seed="42",
        batch_id=store.batch_id,
    )
    staged = workspace.root / "random_body" / "data" / _CANDIDATE_ID

    assert load_simulation_workspace(store.root) == workspace
    assert frozenset(path.name for path in staged.iterdir()) == (
        candidate_artifact_names(_CANDIDATE_ID)
    )
    properties = load_yaml_bytes(
        (workspace.root / "dataset_properties.yaml").read_bytes()
    )
    assert properties["body_samples"] == (
        "/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/"
        "DataSet/5000_body_shapes_and_measures"
    )


def test_filter_projection_maps_pass_and_failure_in_workspace_order(
    tmp_path: Path,
) -> None:
    store = _workspace_store(tmp_path)
    for candidate_id in (_CANDIDATE_ID, _SECOND_ID):
        _publish_candidate(store, candidate_id)
    workspace = prepare_simulation_workspace(
        store.root,
        [
            _pending_candidate(store, _SECOND_ID),
            _pending_candidate(store, _CANDIDATE_ID),
        ],
        global_seed="42",
        batch_id=store.batch_id,
    )
    report = _filter_report(
        workspace.root,
        workspace.candidate_ids,
        passed=[_SECOND_ID],
        failed=[_CANDIDATE_ID],
    )

    projection = load_filter_projection(workspace, report)

    assert [row["candidate_id"] for row in projection.results] == list(
        workspace.candidate_ids
    )
    assert [row["target_simulation_status"] for row in projection.results] == [
        "fail",
        "pass",
    ]
    assert projection.results[0]["failure_reasons"] == ("fixture_failure",)


def test_simulation_workspace_can_be_removed(tmp_path: Path) -> None:
    store = _workspace_store(tmp_path)
    _publish_candidate(store, _CANDIDATE_ID)
    workspace = prepare_simulation_workspace(
        store.root,
        [_pending_candidate(store, _CANDIDATE_ID)],
        global_seed="42",
        batch_id=store.batch_id,
    )

    remove_simulation_workspace(store.root)

    assert not workspace.root.exists()
