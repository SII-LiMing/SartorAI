"""Focused integration tests for the editing package.

These tests intentionally exercise only public editing boundaries.  They must
remain runnable from the clean GarmentCodeRC repository root with:

    python -m pytest editing/tests
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from assets.garment_programs.meta_garment import IncorrectElementConfiguration
from pattern_sampler import (
    ONE_PIECE_SHIRT_DRESS_MIN_LENGTH,
    assert_param_combinations,
)
from editing.semantic_edit.construction import instantiate_body_parameters


EDITING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EDITING_ROOT.parent
FORBIDDEN_IMPORT_ROOTS = (
    "edit_generation",
    "assets.garment_programs.design_constraints",
    "assets.garment_programs.exceptions",
    "pygarment.pattern.geometry",
)
LIGHTWEIGHT_CLI_BLOCKLIST = (
    "editing.semantic_edit.construction",
    "editing.semantic_edit.registry",
    "pattern_sampler",
    "pygarment",
    "matplotlib",
    "cairo",
    "scipy",
)
_MODULES_MARKER = "__semantic_edit_modules__="


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (REPOSITORY_ROOT.as_posix(), inherited)
        if part
    )
    return environment


def _run_python_code(code: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-B", "-c", code, *arguments),
        cwd=REPOSITORY_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


def _run_python_module(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, "-B", "-m", *arguments),
        cwd=REPOSITORY_ROOT,
        env=_subprocess_environment(),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


def _run_cli_with_modules(
    *arguments: str,
) -> tuple[subprocess.CompletedProcess[str], tuple[str, ...]]:
    harness = f"""
import json
import runpy
import sys

sys.argv = ["editing.semantic_edit", *sys.argv[1:]]
try:
    runpy.run_module("editing.semantic_edit", run_name="__main__")
except SystemExit as error:
    return_code = error.code if isinstance(error.code, int) else 1
else:
    return_code = 0
print({_MODULES_MARKER!r} + json.dumps(sorted(sys.modules)))
raise SystemExit(return_code)
"""
    result = _run_python_code(harness, *arguments)
    module_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(_MODULES_MARKER)
    ]
    assert len(module_lines) == 1, result
    modules = tuple(json.loads(module_lines[0][len(_MODULES_MARKER) :]))
    return result, modules


def _blocked_modules(modules: tuple[str, ...]) -> list[str]:
    return [
        name
        for name in modules
        if any(
            name == blocked or name.startswith(blocked + ".")
            for blocked in LIGHTWEIGHT_CLI_BLOCKLIST
        )
    ]


@pytest.mark.parametrize("command", ("plan", "generate", "project", "status"))
def test_cli_help_is_lightweight(command: str) -> None:
    assert sys.version_info[:2] == (3, 10)
    result, modules = _run_cli_with_modules(command, "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert _blocked_modules(modules) == []


def test_cli_argument_error_is_lightweight() -> None:
    assert sys.version_info[:2] == (3, 10)
    result, modules = _run_cli_with_modules("status")

    assert result.returncode == 2
    assert "the following arguments are required: --output-root" in result.stderr
    assert _blocked_modules(modules) == []


def test_explicit_semantic_edit_submodules_import_without_package_facade() -> None:
    result = _run_python_code(
        """
import json
import editing
import editing.semantic_edit
import editing.semantic_edit.pipeline
import editing.semantic_edit.store

print(json.dumps({
    "editing_file": getattr(editing, "__file__", None),
    "editing_origin": editing.__spec__.origin,
    "namespace_file": getattr(editing.semantic_edit, "__file__", None),
    "namespace_origin": editing.semantic_edit.__spec__.origin,
    "pipeline_file": editing.semantic_edit.pipeline.__file__,
    "store_file": editing.semantic_edit.store.__file__,
}, sort_keys=True))
"""
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["editing_file"] is None
    assert payload["editing_origin"] is None
    assert payload["namespace_file"] is None
    assert payload["namespace_origin"] is None
    assert Path(payload["pipeline_file"]).resolve() == (
        EDITING_ROOT / "semantic_edit" / "pipeline.py"
    )
    assert Path(payload["store_file"]).resolve() == (
        EDITING_ROOT / "semantic_edit" / "store.py"
    )


def test_namespace_overlay_runs_real_cli_and_geometry_imports() -> None:
    cli = _run_python_module("editing.semantic_edit", "--help")
    assert cli.returncode == 0, cli.stderr
    assert "usage:" in cli.stdout

    imported = _run_python_code(
        "import editing.pattern_geometry; import editing.semantic_edit.pipeline"
    )
    assert imported.returncode == 0, imported.stderr


def _design(
    *,
    upper: str | None = None,
    bottom: str | None = None,
    waistband: str | None = None,
) -> dict:
    """Return the smallest complete mapping needed by the constraints."""

    return {
        "meta": {
            "upper": {"v": upper},
            "bottom": {"v": bottom},
            "wb": {"v": waistband},
        },
        "shirt": {"length": {"v": 1.0}},
        "pants": {
            "length": {"v": 0.8},
            "cuff": {
                "type": {"v": None},
                "cuff_len": {"v": 0.0},
            },
        },
        "skirt": {"ruffle": {"v": 1.0}},
        "godet-skirt": {"base": {"v": "Skirt2"}},
        "levels-skirt": {
            "base": {"v": "Skirt2"},
            "length": {"v": 0.4},
        },
        "flare-skirt": {
            "length": {"v": 0.4},
            "suns": {"v": 0.5},
        },
        "waistband": {
            "waist": {"v": 1.0},
            "width": {"v": 0.3},
        },
    }


def test_original_sampler_constraints_accept_and_reject_cases() -> None:
    complete_outfit = _design(upper="Shirt", bottom="Pants")
    assert assert_param_combinations(complete_outfit) is None

    invalid_circle = _design(upper="Shirt", bottom="SkirtCircle")
    invalid_circle["flare-skirt"]["suns"]["v"] = 0.0
    with pytest.raises(
        IncorrectElementConfiguration,
        match="SkirtCircle suns must be finite and positive",
    ):
        assert_param_combinations(invalid_circle)

    inactive_circle_parameter = _design(upper="Shirt", bottom="Pants")
    inactive_circle_parameter["flare-skirt"]["suns"]["v"] = 0.0
    assert assert_param_combinations(inactive_circle_parameter) is None

    pants_too_long = _design(upper="Shirt", bottom="Pants")
    pants_too_long["pants"]["length"]["v"] = 0.93
    with pytest.raises(IncorrectElementConfiguration, match="Pants too long"):
        assert_param_combinations(pants_too_long)

    collapsed_cuff = _design(upper="Shirt", bottom="Pants")
    collapsed_cuff["pants"]["length"] = {
        "v": 0.2,
        "range": [0.2, 0.9],
    }
    collapsed_cuff["pants"]["cuff"] = {
        "type": {"v": "CuffBand"},
        "cuff_len": {"v": 0.108, "range": [0.05, 0.9]},
    }
    with pytest.raises(
        IncorrectElementConfiguration,
        match="Pants cuff length collapses below its minimum",
    ):
        assert_param_combinations(collapsed_cuff)

    one_piece_shirt = _design(upper="Shirt")
    one_piece_shirt["shirt"]["length"]["v"] = (
        ONE_PIECE_SHIRT_DRESS_MIN_LENGTH
    )
    assert assert_param_combinations(one_piece_shirt) is None


def test_legacy_body_parameters_can_start_in_memory() -> None:
    class LegacyBodyParameters:
        """Reproduce the clean checkout's unconditional ``open('')``."""

        def __init__(self, param_file: str = "") -> None:
            with open(param_file, encoding="utf-8"):
                pass

        def load_from_dict(self, values: dict) -> None:
            self.params.update(values)
            self.params["_double_height"] = self.params["height"] * 2

    body = instantiate_body_parameters(LegacyBodyParameters)
    body.load_from_dict({"height": 171.25})

    assert body.params == {
        "height": 171.25,
        "_double_height": 342.5,
    }


def test_body_adapter_does_not_hide_a_real_missing_file() -> None:
    missing_path = "/definitely/not/a/real/body.yaml"

    class BrokenBodyParameters:
        def __init__(self) -> None:
            with open(missing_path, encoding="utf-8"):
                pass

    with pytest.raises(FileNotFoundError) as caught:
        instantiate_body_parameters(BrokenBodyParameters)

    assert caught.value.filename == missing_path


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return tuple(modules)


def test_production_overlay_has_no_legacy_or_absent_clean_imports() -> None:
    violations: list[str] = []
    for path in sorted(EDITING_ROOT.rglob("*.py")):
        if "tests" in path.relative_to(EDITING_ROOT).parts:
            continue
        for module in _imported_modules(path):
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in FORBIDDEN_IMPORT_ROOTS
            ):
                violations.append(
                    f"{path.relative_to(EDITING_ROOT)} imports {module}"
                )

    assert violations == []
