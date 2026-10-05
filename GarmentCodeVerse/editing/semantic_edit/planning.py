"""Plan and select balanced single-operation semantic edits.

This lightweight boundary owns source discovery, capability enumeration,
balanced assignment, run-manifest construction, and deterministic candidate
selection.  Planning deliberately does not reconstruct source patterns,
discover or read source specifications, compare source topology, hash source
directories, render targets, or generate target prompts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import heapq
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from editing.canonical_io import (
    CanonicalDataError,
    canonical_hash,
    canonical_json_bytes,
    load_body_yaml_bytes,
    load_design_yaml_bytes,
    load_json_bytes,
    plain_copy,
    sha256_bytes,
)
import editing.config as config_module
from editing.models import (
    BodySnapshot,
    PlannedOperation,
    RuleContext,
    SourceRecord,
    canonical_operation_signature,
)
from editing.publication import require_disjoint_roots
import editing.semantic_edit.prompt as prompt_module
import editing.semantic_edit.prompt_state as prompt_state_module
import editing.semantic_edit.registry as registry_module
import editing.semantic_edit.rules.godet as godet_module
from editing.semantic_edit.store import PLAN_FORMAT
__all__ = (
    "Capability",
    "build_run_manifest",
    "plan",
    "select_candidate",
)


_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_ASSIGNMENT_SOURCE_KEY_RE = re.compile(
    r"batch_[0-9]+/random_body/(?:data/)?"
    r"(rand_[A-Za-z0-9][A-Za-z0-9_-]*)\Z",
    re.ASCII,
)


class CoverageImpossibleError(ValueError):
    """The eligible pool cannot assign one required candidate per source."""


def _ascii_text(value: Any, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or not value.isascii()
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be non-empty trimmed ASCII text")
    return value


def _digest(value: Any, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must use sha256:<64 lowercase hexadecimal characters>"
        )
    return value


@dataclass(frozen=True, order=True)
class Capability:
    """One concrete or wildcarded semantic transition family."""

    rule_id: str
    direction: str
    source_state_id: str
    target_state_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "rule_id",
            "direction",
            "source_state_id",
            "target_state_id",
        ):
            _ascii_text(getattr(self, field_name), field_name)
        if self.rule_id == "*" or self.direction == "*":
            raise ValueError("rule_id and direction cannot be wildcards")
        if (
            "*" not in (self.source_state_id, self.target_state_id)
            and self.source_state_id == self.target_state_id
        ):
            raise ValueError("a capability must describe a state change")

    def matches(self, actual: "Capability") -> bool:
        if not isinstance(actual, Capability):
            raise TypeError("actual must be a Capability")
        return (
            self.rule_id == actual.rule_id
            and self.direction == actual.direction
            and self.source_state_id in {"*", actual.source_state_id}
            and self.target_state_id in {"*", actual.target_state_id}
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Capability":
        """Build the canonical value represented by a capability payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("capability payload must be a mapping")
        return cls(
            rule_id=payload["rule_id"],
            direction=payload["direction"],
            source_state_id=payload["source_state_id"],
            target_state_id=payload["target_state_id"],
        )

    @property
    def concrete(self) -> bool:
        return "*" not in (self.source_state_id, self.target_state_id)

    @property
    def count_key(self) -> str:
        """Return the persisted distribution identity for this transition."""

        return ":".join(
            (
                self.rule_id,
                self.direction,
                self.source_state_id,
                self.target_state_id,
            )
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "direction": self.direction,
            "source_state_id": self.source_state_id,
            "target_state_id": self.target_state_id,
        }


_REQUIRED_CAPABILITIES = (
    Capability("shirt_length", "increase", "*", "*"),
    Capability("shirt_length", "decrease", "*", "*"),
    Capability("sleeve_length", "increase", "*", "*"),
    Capability("sleeve_length", "decrease", "*", "*"),
    Capability("skirt_length", "increase", "*", "*"),
    Capability("skirt_length", "decrease", "*", "*"),
    Capability("pants_length", "increase", "*", "*"),
    Capability("pants_length", "decrease", "*", "*"),
    Capability("shirt_fit", "increase", "close-fitting", "loose"),
    Capability("shirt_fit", "decrease", "loose", "close-fitting"),
    Capability("sleeve_presence", "add", "sleeveless", "*"),
    Capability("sleeve_presence", "remove", "*", "sleeveless"),
    Capability("hood_presence", "add", "hoodless", "hooded"),
    Capability("hood_presence", "remove", "hooded", "hoodless"),
    Capability("collar_presence", "add", "collarless", "*"),
    Capability("collar_presence", "remove", "*", "collarless"),
    Capability("godet_insert_count", "increase", "*", "*"),
    Capability("godet_insert_count", "decrease", "*", "*"),
    Capability("neckline_type", "change", "*", "CircleNeckHalf"),
    Capability("neckline_type", "change", "*", "CurvyNeckHalf"),
    Capability("neckline_type", "change", "*", "VNeckHalf"),
    Capability("neckline_type", "change", "*", "SquareNeckHalf"),
    Capability("neckline_type", "change", "*", "CircleArcNeckHalf"),
    Capability("neckline_type", "change", "*", "Bezier2NeckHalf"),
)


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _required_pattern_for_capability(
    capability: Capability,
) -> Capability:
    matches = tuple(
        required
        for required in _REQUIRED_CAPABILITIES
        if required.matches(capability)
    )
    if len(matches) != 1:
        raise CoverageImpossibleError(
            "a concrete candidate capability must match exactly one required "
            f"capability: {capability.to_payload()}"
        )
    return matches[0]


def _balanced_candidate_assignment(
    sources: Mapping[str, Mapping[Capability, str]],
) -> dict[str, tuple[Capability, str]]:
    """Assign one candidate per source with hierarchical balance."""

    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping of source candidates")
    if not sources:
        raise CoverageImpossibleError(
            "candidate mapping has no source with a required capability"
        )
    probes_by_source: dict[str, dict[Capability, str]] = {}
    for source_key, probes in sources.items():
        _ascii_text(source_key, "source_key")
        if _ASSIGNMENT_SOURCE_KEY_RE.fullmatch(source_key) is None:
            raise ValueError(
                "source_key must match batch_<id>/random_body/"
                "[data/]<rand_id>"
            )
        if not isinstance(probes, Mapping):
            raise TypeError("source candidates must be a capability mapping")
        validated: dict[Capability, str] = {}
        for capability, operation_signature in probes.items():
            if not isinstance(capability, Capability):
                raise TypeError("candidate capability must be a Capability")
            if not capability.concrete:
                raise ValueError("a candidate capability must be concrete")
            _digest(operation_signature, "operation_signature")
            try:
                _required_pattern_for_capability(capability)
            except CoverageImpossibleError:
                continue
            validated[capability] = operation_signature
        probes_by_source[source_key] = {
            capability: validated[capability]
            for capability in sorted(validated)
        }

    ordered_sources = tuple(
        sorted(
            probes_by_source,
            key=str.encode,
        )
    )
    if any(not probes for probes in probes_by_source.values()):
        raise CoverageImpossibleError(
            "balanced assignment requires at least one required candidate "
            "for every source"
        )
    exact_capabilities = tuple(
        sorted(
            {
                capability
                for grouped in probes_by_source.values()
                for capability in grouped
            }
        )
    )
    required_patterns = tuple(
        required
        for required in _REQUIRED_CAPABILITIES
        if any(
            required.matches(capability)
            for capability in exact_capabilities
        )
    )
    rule_ids = tuple(
        rule_id
        for rule_id in registry_module.RULE_ORDER
        if any(
            capability.rule_id == rule_id
            for capability in exact_capabilities
        )
    )

    source_node_start = 1
    exact_node_start = source_node_start + len(ordered_sources)
    required_node_start = exact_node_start + len(exact_capabilities)
    rule_node_start = required_node_start + len(required_patterns)
    sink = rule_node_start + len(rule_ids)
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]

    def add_edge(left: int, right: int, capacity: int, cost: int) -> _FlowEdge:
        if capacity <= 0:
            raise ValueError("flow edge capacity must be positive")
        forward = _FlowEdge(
            to=right,
            reverse=len(graph[right]),
            capacity=capacity,
            cost=cost,
        )
        reverse = _FlowEdge(
            to=left,
            reverse=len(graph[left]),
            capacity=0,
            cost=-cost,
        )
        graph[left].append(forward)
        graph[right].append(reverse)
        return forward

    source_nodes = {
        source_key: source_node_start + index
        for index, source_key in enumerate(ordered_sources)
    }
    exact_nodes = {
        capability: exact_node_start + index
        for index, capability in enumerate(exact_capabilities)
    }
    required_nodes = {
        required: required_node_start + index
        for index, required in enumerate(required_patterns)
    }
    rule_nodes = {
        rule_id: rule_node_start + index
        for index, rule_id in enumerate(rule_ids)
    }

    source_choice_edges: dict[tuple[str, Capability], _FlowEdge] = {}
    for source_key in ordered_sources:
        source_node = source_nodes[source_key]
        add_edge(0, source_node, 1, 0)
        for capability in probes_by_source[source_key]:
            source_choice_edges[(source_key, capability)] = add_edge(
                source_node,
                exact_nodes[capability],
                1,
                0,
            )

    source_count = len(ordered_sources)
    hierarchy_base = source_count * source_count + 1
    exact_weight = 1
    required_weight = hierarchy_base
    rule_weight = hierarchy_base * hierarchy_base

    for capability in exact_capabilities:
        required = _required_pattern_for_capability(capability)
        support = sum(
            capability in probes_by_source[source_key]
            for source_key in ordered_sources
        )
        for index in range(1, support + 1):
            add_edge(
                exact_nodes[capability],
                required_nodes[required],
                1,
                (2 * index - 1) * exact_weight,
            )

    for required in required_patterns:
        support = sum(
            any(
                required.matches(capability)
                for capability in probes_by_source[source_key]
            )
            for source_key in ordered_sources
        )
        for index in range(1, support + 1):
            add_edge(
                required_nodes[required],
                rule_nodes[required.rule_id],
                1,
                (2 * index - 1) * required_weight,
            )

    for rule_id in rule_ids:
        support = sum(
            any(
                capability.rule_id == rule_id
                for capability in probes_by_source[source_key]
            )
            for source_key in ordered_sources
        )
        for index in range(1, support + 1):
            add_edge(
                rule_nodes[rule_id],
                sink,
                1,
                (2 * index - 1) * rule_weight,
            )

    potential = [0] * len(graph)
    flow = 0
    while flow < source_count:
        infinity = 10**30
        distance = [infinity] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edge = [-1] * len(graph)
        distance[0] = 0
        queue: list[tuple[int, int]] = [(0, 0)]
        while queue:
            current_distance, left = heapq.heappop(queue)
            if current_distance != distance[left]:
                continue
            for edge_index, edge in enumerate(graph[left]):
                if edge.capacity == 0:
                    continue
                reduced = edge.cost + potential[left] - potential[edge.to]
                candidate_distance = current_distance + reduced
                if candidate_distance < distance[edge.to]:
                    distance[edge.to] = candidate_distance
                    previous_node[edge.to] = left
                    previous_edge[edge.to] = edge_index
                    heapq.heappush(queue, (candidate_distance, edge.to))
        if distance[sink] == infinity:
            raise CoverageImpossibleError(
                f"balanced candidate flow stopped at "
                f"{flow}/{source_count} sources"
            )
        for node, value in enumerate(distance):
            if value != infinity:
                potential[node] += value
        node = sink
        while node != 0:
            left = previous_node[node]
            edge_index = previous_edge[node]
            if left < 0 or edge_index < 0:  # pragma: no cover
                raise AssertionError("flow predecessor chain is incomplete")
            edge = graph[left][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = left
        flow += 1

    result: dict[str, tuple[Capability, str]] = {}
    for source_key in ordered_sources:
        selected = tuple(
            capability
            for capability in probes_by_source[source_key]
            if source_choice_edges[
                (source_key, capability)
            ].capacity == 0
        )
        if len(selected) != 1:
            raise AssertionError(
                "balanced flow did not select exactly one capability for "
                f"{source_key}"
            )
        capability = selected[0]
        result[source_key] = (
            _required_pattern_for_capability(capability),
            probes_by_source[source_key][capability],
        )
    return result


_BATCH_ID_RE = re.compile(r"batch_[0-9]+\Z")
_SAMPLE_ID_RE = re.compile(
    r"rand_[A-Za-z0-9][A-Za-z0-9_-]*\Z"
)
_SOURCE_KEY_RE = re.compile(
    r"batch_[0-9]+/random_body/(?:data/)?"
    r"(rand_[A-Za-z0-9][A-Za-z0-9_-]*)\Z"
)


class SourcePlanError(RuntimeError):
    """A source batch cannot produce a safe candidate plan."""


def _directory(path: Path | str, field_name: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise SourcePlanError(f"{field_name} is missing or unsafe: {candidate}")
    return candidate.resolve()


def _regular_file(path: Path | str, field_name: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise SourcePlanError(f"{field_name} is missing or unsafe: {candidate}")
    return candidate.resolve()


def _batch_id(value: str) -> str:
    if type(value) is not str or _BATCH_ID_RE.fullmatch(value) is None:
        raise SourcePlanError("batch_id must match batch_<nonnegative integer>")
    return value


def _required_source_yaml_file(
    sample_dir: Path,
    sample_id: str,
    filename: str,
) -> Path:
    """Resolve canonical or sample-prefixed source YAML without ambiguity."""

    names = (filename, f"{sample_id}_{filename}")
    matches = [
        sample_dir / name
        for name in names
        if os.path.lexists(sample_dir / name)
    ]
    if len(matches) != 1:
        raise SourcePlanError(
            f"{sample_dir}: requires exactly one source YAML named "
            f"{names[0]!r} or {names[1]!r}; found {len(matches)}"
        )
    return _regular_file(matches[0], f"{sample_id} source {filename}")


def _portable_body_sample(body: Mapping[str, Any], body_path: Path) -> str:
    body_sample = body.get("body_sample")
    if (
        not isinstance(body_sample, str)
        or not body_sample
        or body_sample.strip() != body_sample
        or "/" in body_sample
        or "\\" in body_sample
        or "\x00" in body_sample
    ):
        raise SourcePlanError(
            f"{body_path}: body.body_sample must be a non-empty portable string"
        )
    return body_sample


def _read_record(
    sample_dir: Path,
    *,
    source_key: str,
) -> SourceRecord:
    source_match = (
        _SOURCE_KEY_RE.fullmatch(source_key)
        if type(source_key) is str
        else None
    )
    if source_match is None or source_match.group(1) != sample_dir.name:
        raise SourcePlanError(
            "source_key must be a portable relative source path ending in "
            f"{sample_dir.name}"
        )
    design_path = _required_source_yaml_file(
        sample_dir,
        sample_dir.name,
        "design_params.yaml",
    )
    body_path = _required_source_yaml_file(
        sample_dir,
        sample_dir.name,
        "body_measurements.yaml",
    )

    try:
        design_bytes = design_path.read_bytes()
        design = load_design_yaml_bytes(design_bytes)
    except (OSError, CanonicalDataError) as error:
        raise SourcePlanError(
            f"invalid design_params.yaml at {design_path}: {error}"
        ) from error
    try:
        body_bytes = body_path.read_bytes()
        body = load_body_yaml_bytes(body_bytes)
    except (OSError, CanonicalDataError) as error:
        raise SourcePlanError(
            f"invalid body_measurements.yaml at {body_path}: {error}"
        ) from error
    _portable_body_sample(body, body_path)

    body_snapshot = BodySnapshot(
        body=body,
        raw_yaml=body_bytes,
    )
    return SourceRecord(
        source_key=source_key,
        sample_id=sample_dir.name,
        design=design,
        body_snapshot=body_snapshot,
    )


def _batch_records(input_root: Path, batch_id: str) -> tuple[Any, ...]:
    body_root = input_root / batch_id / "random_body"
    canonical_data_root = body_root / "data"
    if os.path.lexists(canonical_data_root):
        if canonical_data_root.is_symlink() or not canonical_data_root.is_dir():
            raise SourcePlanError(
                "batch canonical data root is unsafe: "
                f"{canonical_data_root}"
            )
        data_root = canonical_data_root
        source_prefix = f"{batch_id}/random_body/data"
        reject_non_sample_directories = True
    elif body_root.is_dir() and not body_root.is_symlink():
        data_root = body_root
        source_prefix = f"{batch_id}/random_body"
        reject_non_sample_directories = False
    else:
        raise SourcePlanError(
            f"batch random-body root is missing or unsafe: {body_root}"
        )
    directories = tuple(
        child
        for child in data_root.iterdir()
        if child.is_dir() and not child.is_symlink()
    )
    invalid = [
        child.name
        for child in directories
        if _SAMPLE_ID_RE.fullmatch(child.name) is None
    ]
    if reject_non_sample_directories and invalid:
        raise SourcePlanError(
            "batch contains non-portable source IDs: " + ", ".join(invalid[:5])
        )
    sample_dirs = tuple(
        sorted(
            (
                child
                for child in directories
                if _SAMPLE_ID_RE.fullmatch(child.name) is not None
            ),
            key=lambda child: child.name.encode("ascii"),
        )
    )
    if not sample_dirs:
        raise SourcePlanError(f"batch has no source samples: {data_root}")
    keys = tuple(
        f"{source_prefix}/{child.name}"
        for child in sample_dirs
    )
    return tuple(
        _read_record(
            child,
            source_key=key,
        )
        for key, child in zip(keys, sample_dirs)
    )


def _concrete_capability(
    planned: PlannedOperation,
) -> Capability | None:
    pairs = {
        (
            transition.source_state.state_id,
            transition.target_state.state_id,
        )
        for transition in planned.transitions
    }
    if len(pairs) != 1:
        return None
    source_state_id, target_state_id = next(iter(pairs))
    if source_state_id == target_state_id:
        return None
    capability = Capability(
        planned.operation.rule_id,
        planned.operation.direction,
        source_state_id,
        target_state_id,
    )
    if not any(
        required.matches(capability)
        for required in _REQUIRED_CAPABILITIES
    ):
        return None
    return capability


def _source_candidates(
    record: Any,
    *,
    registry: Any,
    config: Any,
    prompt_classifier: prompt_state_module.PromptStateClassifier,
) -> tuple[
    list[dict[str, Any]],
    dict[Capability, str] | None,
    dict[str, dict[str, Any]],
]:
    facts = godet_module.probe_constructor_facts(
        record.design,
        record.body_snapshot,
    )
    states = prompt_classifier(record.design, record.body_snapshot.raw_yaml)
    design = plain_copy(record.design, tuples_as_lists=True)
    body = plain_copy(record.body_snapshot.body, tuples_as_lists=True)

    def classify_prompt(
        target_design: dict[str, Any],
        raw_body_yaml: bytes,
    ) -> dict[str, Any]:
        return plain_copy(
            prompt_classifier(target_design, raw_body_yaml),
            tuples_as_lists=True,
        )

    context = RuleContext(
        design=design,
        body=body,
        raw_body_yaml=record.body_snapshot.raw_yaml,
        active_paths=prompt_state_module.resolve_active_paths(design),
        constructor_facts=plain_copy(facts, tuples_as_lists=True),
        prompt_states=plain_copy(states, tuples_as_lists=True),
        classify_prompt=classify_prompt,
    )

    rows_by_signature: dict[str, dict[str, Any]] = {}
    first_signature_by_capability: dict[Capability, str] = {}
    for rule_id in registry_module.RULE_ORDER:
        if not config.rules[rule_id].enabled:
            continue
        candidates = registry[rule_id].candidates(context)
        if not isinstance(candidates, tuple):
            raise SourcePlanError(
                f"{record.source_key}: {rule_id} candidates are not materialized"
            )
        for planned in candidates:
            if not isinstance(planned, PlannedOperation):
                raise SourcePlanError(
                    f"{record.source_key}: {rule_id} emitted an invalid candidate"
                )
            capability = _concrete_capability(planned)
            if capability is None:
                continue
            signature = canonical_operation_signature(planned)
            payload = planned.to_payload()
            row = {
                "capability": capability.to_payload(),
                "operation_signature": signature,
                "planned_operation": payload,
            }
            existing = rows_by_signature.get(signature)
            if (
                existing is not None
                and canonical_json_bytes(existing) != canonical_json_bytes(row)
            ):
                raise SourcePlanError(
                    f"{record.source_key}: conflicting candidate signature"
                )
            rows_by_signature.setdefault(signature, row)
            first_signature_by_capability.setdefault(
                capability,
                signature,
            )

    rows = sorted(
        rows_by_signature.values(),
        key=lambda row: (
            row["capability"]["rule_id"],
            row["capability"]["direction"],
            row["capability"]["source_state_id"],
            row["capability"]["target_state_id"],
            row["operation_signature"],
        ),
    )
    if not rows:
        return rows, None, rows_by_signature

    eligibility = {
        key: first_signature_by_capability[key]
        for key in sorted(first_signature_by_capability)
    }
    return rows, eligibility, rows_by_signature


def _preferred_payload(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    capability = row["capability"]
    planned_payload = row["planned_operation"]
    return {
        **capability,
        "scope": planned_payload["operation"]["scope"],
        "operation_signature": row["operation_signature"],
    }


def plan(
    *,
    input_root: Path | str,
    batch_id: str,
    analysis_root: Path | str,
    config_path: Path | str,
    prompt_builder: Path | str,
    seed: str = "42",
) -> dict[str, Any]:
    """Write one canonical plan report consumed directly by generation."""

    source_root = _directory(input_root, "input_root")
    roots = require_disjoint_roots(
        input_root=source_root,
        analysis_root=Path(analysis_root).expanduser().resolve(),
    )
    source_root = roots["input_root"]
    output = roots["analysis_root"]
    batch = _batch_id(batch_id)
    config_file = _regular_file(config_path, "config")
    prompt_file = _regular_file(
        prompt_builder,
        "prompt builder",
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"analysis root already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = registry_module.load_semantic_config(config_file)
    registry = registry_module.build_semantic_registry(config)
    builder = prompt_module.PromptBuilder.load(prompt_file)
    records = _batch_records(source_root, batch)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".partial",
            dir=output.parent,
        )
    )
    prompt_work = staging / ".source_prompt_work"
    prompt_work.mkdir()
    try:
        prompt_classifier = prompt_state_module.PromptStateClassifier(
            builder,
            prompt_work,
        )
        source_rows: dict[str, dict[str, Any]] = {}
        eligible_sources: dict[
            str,
            tuple[Any, dict[str, dict[str, Any]], dict[Capability, str]],
        ] = {}
        planning_failures: dict[str, dict[str, str]] = {}
        for index, record in enumerate(records, start=1):
            try:
                rows, eligibility, rows_by_signature = _source_candidates(
                    record,
                    registry=registry,
                    config=config,
                    prompt_classifier=prompt_classifier,
                )
            except godet_module.ConstructorFactError as exc:
                planning_failures[record.source_key] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "source_key": record.source_key,
                    "source_sample_id": record.sample_id,
                }
                print(
                    f"[semantic-edit-plan] {batch}: "
                    f"skipped={record.sample_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                source_rows[record.source_key] = {
                    "source_key": record.source_key,
                    "probes": rows,
                }
                if eligibility is not None:
                    eligible_sources[record.source_key] = (
                        record,
                        rows_by_signature,
                        eligibility,
                    )
            if index % 10 == 0 or index == len(records):
                print(
                    f"[semantic-edit-plan] {batch}: "
                    f"planned={index}/{len(records)} "
                    f"eligible={len(eligible_sources)} "
                    f"errors={len(planning_failures)}",
                    flush=True,
                )

        if not eligible_sources:
            raise SourcePlanError("no source has an applicable edit candidate")
        assignment = _balanced_candidate_assignment(
            {
                source_key: eligibility
                for source_key, (
                    _record,
                    _rows_by_signature,
                    eligibility,
                ) in eligible_sources.items()
            }
        )

        preferences: dict[str, dict[str, Any]] = {}
        assigned_rules: Counter[str] = Counter()
        for slot_index, source_key in enumerate(sorted(eligible_sources)):
            record, rows_by_signature, _eligibility = eligible_sources[
                source_key
            ]
            required, signature = assignment[source_key]
            row = rows_by_signature[signature]
            assigned_rules[required.rule_id] += 1
            preferences[f"slot_{slot_index:03d}"] = {
                "source_key": record.source_key,
                "source_sample_id": record.sample_id,
                "preferred_capability": _preferred_payload(row),
            }

        report: dict[str, Any] = {
            "format": PLAN_FORMAT,
            "batch_id": batch,
            "selection_seed": str(seed),
            "config_path": config_file.as_posix(),
            "config_file_sha256": sha256_bytes(config_file.read_bytes()),
            "effective_config_hash": config_module.config_digest(config),
            "prompt_builder_sha256": builder.sha256,
            "source_count": len(records),
            "eligible_source_count": len(preferences),
            "excluded_source_count": len(records) - len(preferences),
            "planning_failure_count": len(planning_failures),
            "planning_failures": {
                key: planning_failures[key]
                for key in sorted(planning_failures, key=str.encode)
            },
            "assigned_rule_distribution": dict(sorted(assigned_rules.items())),
            "source_probe_results": {
                key: source_rows[key]
                for key in sorted(source_rows, key=str.encode)
            },
            "selected_slot_preferences": preferences,
        }
        report["report_hash"] = canonical_hash(report)
        shutil.rmtree(prompt_work)
        (staging / "pilot_eligibility_report.json").write_bytes(
            canonical_json_bytes(report)
        )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return {
        "format": PLAN_FORMAT,
        "analysis_root": output.as_posix(),
        "batch_id": batch,
        "source_count": len(records),
        "eligible_source_count": len(preferences),
        "excluded_source_count": len(records) - len(preferences),
        "planning_failure_count": len(planning_failures),
        "assigned_rule_distribution": dict(sorted(assigned_rules.items())),
    }


class ManifestError(ValueError):
    """The current source-plan report cannot produce a manifest."""


def _candidate(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "operation_signature": row["operation_signature"],
        "planned_operation_payload": row["planned_operation"],
        "capability": dict(row["capability"]),
    }


def _seeded_candidate_key(
    candidate: Mapping[str, Any],
    *,
    seed: str,
    slot_id: str,
) -> tuple[str, str]:
    tie_identity = {
        "seed": seed,
        "slot_id": slot_id,
        "operation_signature": candidate["operation_signature"],
    }
    tie_digest = hashlib.sha256(canonical_json_bytes(tie_identity)).hexdigest()
    return tie_digest, candidate["operation_signature"]


def _build_slot(
    *,
    slot_id: str,
    preference_value: Any,
    source_rows: Mapping[str, Any],
    seed: str,
) -> dict[str, Any]:
    preference = preference_value
    source_key = preference["source_key"]
    primary_signature = preference["preferred_capability"][
        "operation_signature"
    ]
    probes = source_rows[source_key]["probes"]
    primary_indexes = tuple(
        index
        for index, row in enumerate(probes)
        if row["operation_signature"] == primary_signature
    )
    if len(primary_indexes) != 1:
        raise ManifestError(
            "preferred operation signature must match exactly one source probe"
        )
    primary_index = primary_indexes[0]
    primary = _candidate(probes[primary_index])
    remaining = [
        _candidate(row)
        for index, row in enumerate(probes)
        if index != primary_index
    ]
    remaining.sort(
        key=lambda candidate: _seeded_candidate_key(
            candidate,
            seed=seed,
            slot_id=slot_id,
        )
    )
    return {
        "slot_id": slot_id,
        "source_key": source_key,
        "source_sample_id": preference["source_sample_id"],
        "primary_operation_signature": primary_signature,
        "candidates": [primary, *remaining],
    }


def build_run_manifest(
    analysis_report_path: Path | str,
    seed: str,
) -> dict[str, Any]:
    """Project selected sources and their one-operation probes."""

    report_path = Path(analysis_report_path).expanduser().absolute()
    try:
        report = load_json_bytes(
            report_path.read_bytes(),
            expected_top_level=(
                "source_probe_results",
                "selected_slot_preferences",
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ManifestError(
            f"analysis_report_path is not strict canonical JSON: {report_path}"
        ) from exc
    source_rows = report["source_probe_results"]
    slots = [
        _build_slot(
            slot_id=slot_id,
            preference_value=preference,
            source_rows=source_rows,
            seed=seed,
        )
        for slot_id, preference in report["selected_slot_preferences"].items()
    ]

    # The returned document is a detached JSON value.  Persisting its canonical
    # bytes makes the artifact immutable; no returned value aliases the report.
    return {"slots": slots}


def select_candidate(
    slot: Mapping[str, Any],
    attempted_signatures: Iterable[str],
    rule_counts: Mapping[str, int],
    capability_counts: Mapping[Any, int],
) -> Mapping[str, Any] | None:
    """Choose the next unattempted candidate with deterministic balancing.

    Counts are the already accepted targets plus candidates reserved for the
    active simulation round.  The least-used rule wins, then the least-used
    concrete capability.  The source-plan primary is only a stable tie-break;
    it must not override a deficit in the final accepted distribution.
    """

    primary_signature = slot["primary_operation_signature"]
    attempted = frozenset(attempted_signatures)
    eligible = [
        (index, candidate)
        for index, candidate in enumerate(slot["candidates"])
        if candidate["operation_signature"] not in attempted
    ]
    if not eligible:
        return None

    def score(
        item: tuple[int, Mapping[str, Any]],
    ) -> tuple[int, int, int, int, str]:
        index, candidate = item
        capability = Capability.from_payload(candidate["capability"])
        return (
            rule_counts.get(capability.rule_id, 0),
            capability_counts.get(capability.count_key, 0),
            (
                0
                if candidate["operation_signature"] == primary_signature
                else 1
            ),
            index,
            candidate["operation_signature"],
        )

    _, selected = min(eligible, key=score)
    return plain_copy(selected, tuples_as_lists=True)
