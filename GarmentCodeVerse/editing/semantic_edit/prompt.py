"""Load the approved prompt builder and render one JSON prompt."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from typing import Any

from editing.canonical_io import sha256_bytes


APPROVED_PROMPT_BUILDER_SHA256 = (
    "sha256:7f57d56e40aad42325830de68d999bcef3d618a7761bca55984d360af6320803"
)


class PromptError(RuntimeError):
    """The prompt builder failed identity checking or JSON rendering."""


@dataclass(frozen=True, slots=True)
class PromptArtifact:
    """Prompt values consumed by state classification and publication."""

    document: Any
    json_bytes: bytes
    builder_sha256: str


@dataclass(frozen=True, slots=True)
class PromptBuilder:
    """Call ``parse_design_yaml`` from one approved builder snapshot."""

    sha256: str
    _parse_design_yaml: Callable[[Path], Any] = field(repr=False, compare=False)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str = APPROVED_PROMPT_BUILDER_SHA256,
    ) -> "PromptBuilder":
        source_path = Path(path).expanduser().resolve()
        source = source_path.read_bytes()
        digest = sha256_bytes(source)
        if digest != expected_sha256:
            raise PromptError("prompt builder identity changed")
        namespace: dict[str, Any] = {
            "__name__": "_semantic_edit_build_prompts",
            "__file__": str(source_path),
            "__package__": None,
        }
        exec(compile(source, str(source_path), "exec"), namespace)
        return cls(
            sha256=digest,
            _parse_design_yaml=namespace["parse_design_yaml"],
        )

    def render(
        self,
        design_bytes: bytes,
        body_bytes: bytes,
        *,
        output_root: str | Path,
    ) -> PromptArtifact:
        """Render one prompt using the builder's required paired-file layout."""

        with tempfile.TemporaryDirectory(
            prefix=".prompt-",
            dir=output_root,
        ) as temporary:
            case = Path(temporary) / "case"
            case.mkdir()
            design_path = case / "case_design_params.yaml"
            design_path.write_bytes(design_bytes)
            (case / "case_body_measurements.yaml").write_bytes(body_bytes)
            try:
                document = self._parse_design_yaml(design_path)
                encoded = (
                    json.dumps(
                        document,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            except Exception as exc:
                raise PromptError(str(exc) or type(exc).__name__) from exc
        return PromptArtifact(
            document=document,
            json_bytes=encoded,
            builder_sha256=self.sha256,
        )
