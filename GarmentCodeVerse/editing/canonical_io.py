"""Canonical, strict and side-effect-free data I/O for edit generation.

This module deliberately has no dependency on :mod:`pygarment`, keeping its
canonical serialization policy independent of the pattern runtime and sampler
configuration writers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Optional, Union

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.representer import SafeRepresenter


class CanonicalDataError(ValueError):
    """Raised when input cannot be represented by the canonical data model."""


def plain_copy(value: Any, *, tuples_as_lists: bool = False) -> Any:
    """Detach nested values into ordinary containers.

    Tuple-preserving copies are used by in-memory rule data.  Persistence and
    mutable design data opt into list conversion with ``tuples_as_lists``.
    """

    if isinstance(value, Mapping):
        return {
            key: plain_copy(child, tuples_as_lists=tuples_as_lists)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        items = [
            plain_copy(child, tuples_as_lists=tuples_as_lists)
            for child in value
        ]
        return items if tuples_as_lists else tuple(items)
    if isinstance(value, list):
        return [
            plain_copy(child, tuples_as_lists=tuples_as_lists)
            for child in value
        ]
    return deepcopy(value)


def _normalize(value: Any, *, _path: str = "$") -> Any:
    """Validate and copy a JSON-like value into ordinary ``dict``/``list`` data.

    YAML-specific values such as timestamps, sets and binary blobs are rejected
    rather than being silently stringified.  This makes hashes independent of
    loader-specific Python types.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalDataError(f"non-finite float at {_path}")
        return value
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise CanonicalDataError(
                    f"canonical mapping key at {_path} must be a string, "
                    f"got {type(key).__name__}"
                )
        return {
            key: _normalize(value[key], _path=f"{_path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize(child, _path=f"{_path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise CanonicalDataError(
        f"unsupported canonical value at {_path}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-like value using the project's deterministic JSON form."""

    canonical = _normalize(value)
    try:
        text = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalDataError(f"cannot encode canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Hash raw bytes, returning the metadata form ``sha256:<hex>``."""

    if not isinstance(data, bytes):
        raise TypeError("SHA-256 input must be bytes")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> Any:
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                f"expected a mapping node, got {node.id}",
                node.start_mark,
            )
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from exc
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


# Take independent copies of the representer tables and explicitly pin the safe
# full-precision float representer.  Later global calls to yaml.add_representer
# cannot mutate this dumper, and earlier GarmentCode imports cannot replace its
# float formatting.
class _EditSafeDumper(yaml.SafeDumper):
    pass


_EditSafeDumper.yaml_representers = dict(yaml.SafeDumper.yaml_representers)
_EditSafeDumper.yaml_multi_representers = dict(
    yaml.SafeDumper.yaml_multi_representers
)
_EditSafeDumper.add_representer(float, SafeRepresenter.represent_float)


def _decode_utf8(data: Union[str, bytes], *, format_name: str) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, bytes):
        raise TypeError(f"{format_name} input must be str or bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalDataError(f"{format_name} input is not valid UTF-8") from exc


def _required_keys(expected_top_level: Optional[Union[str, Iterable[str]]]) -> tuple[str, ...]:
    if expected_top_level is None:
        return ()
    if isinstance(expected_top_level, str):
        keys = (expected_top_level,)
    else:
        keys = tuple(expected_top_level)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("expected top-level keys must be non-empty strings")
    return keys


def _validate_document(
    value: Any,
    expected_top_level: Optional[Union[str, Iterable[str]]],
    *,
    exact_top_level: bool,
    format_name: str,
) -> dict[str, Any]:
    normalized = _normalize(value)
    if not isinstance(normalized, dict):
        raise CanonicalDataError(
            f"{format_name} document must be a top-level mapping"
        )
    required = _required_keys(expected_top_level)
    missing = [key for key in required if key not in normalized]
    if missing:
        raise CanonicalDataError(
            f"{format_name} document is missing top-level key(s): "
            + ", ".join(missing)
        )
    if exact_top_level and set(normalized) != set(required):
        unexpected = sorted(set(normalized) - set(required))
        raise CanonicalDataError(
            f"{format_name} document has unexpected top-level key(s): "
            + ", ".join(unexpected)
        )
    for key in required:
        if not isinstance(normalized[key], dict):
            raise CanonicalDataError(
                f"top-level {key!r} payload must be a mapping"
            )
    return normalized


def load_yaml_bytes(
    data: Union[str, bytes],
    expected_top_level: Optional[Union[str, Iterable[str]]] = None,
    *,
    exact_top_level: bool = False,
) -> dict[str, Any]:
    """Load a strict, finite, mapping-shaped YAML document."""

    text = _decode_utf8(data, format_name="YAML")
    try:
        value = yaml.load(text, Loader=_StrictSafeLoader)
    except yaml.YAMLError as exc:
        raise CanonicalDataError(f"invalid YAML: {exc}") from exc
    return _validate_document(
        value,
        expected_top_level,
        exact_top_level=exact_top_level,
        format_name="YAML",
    )


def load_json_bytes(
    data: Union[str, bytes],
    expected_top_level: Optional[Union[str, Iterable[str]]] = None,
    *,
    exact_top_level: bool = False,
) -> dict[str, Any]:
    """Load a strict, finite, mapping-shaped JSON document."""

    text = _decode_utf8(data, format_name="JSON")

    def reject_constant(token: str) -> Any:
        raise CanonicalDataError(f"non-finite JSON number {token!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalDataError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except CanonicalDataError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CanonicalDataError(f"invalid JSON: {exc}") from exc
    return _validate_document(
        value,
        expected_top_level,
        exact_top_level=exact_top_level,
        format_name="JSON",
    )


def load_design_yaml_bytes(data: Union[str, bytes]) -> dict[str, Any]:
    return load_yaml_bytes(data, "design", exact_top_level=True)["design"]


def load_body_yaml_bytes(data: Union[str, bytes]) -> dict[str, Any]:
    return load_yaml_bytes(data, "body", exact_top_level=True)["body"]


def _normalize_dotted_path(path: Union[str, Sequence[str]]) -> tuple[str, ...]:
    if isinstance(path, str):
        parts = tuple(path.split("."))
    elif isinstance(path, Sequence):
        parts = tuple(path)
    else:
        raise TypeError("dotted path must be a string or sequence of strings")
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("dotted path components must be non-empty strings")
    return parts


def dotted_get(mapping: Mapping[str, Any], path: Union[str, Sequence[str]]) -> Any:
    """Read an existing mapping-only dotted path."""

    if not isinstance(mapping, Mapping):
        raise TypeError("dotted path root must be a mapping")
    parts = _normalize_dotted_path(path)
    current: Any = mapping
    for index, part in enumerate(parts):
        if not isinstance(current, Mapping):
            prefix = ".".join(parts[:index])
            raise TypeError(f"dotted path traverses non-mapping value at {prefix!r}")
        if part not in current:
            raise KeyError(".".join(parts[: index + 1]))
        current = current[part]
    return current


def _validate_parameter_value(parent: Mapping[str, Any], value: Any, path: str) -> None:
    schema = parent.get("type")
    if not isinstance(schema, str):
        raise TypeError(f"parameter leaf {path!r} has no string type schema")
    if schema == "float":
        # GarmentCode's checked-in design files legitimately use integer
        # literals such as ``0`` and ``1`` for parameters declared as floats.
        # Preserve that exact representation to retain the input's precise
        # canonical identity instead of coercing it to ``0.0``.
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise TypeError(f"parameter {path!r} requires a finite number")
        return
    if schema == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"parameter {path!r} requires an int")
        return
    if schema == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"parameter {path!r} requires a bool")
        return
    if schema in {"select", "select_null"}:
        if value is None:
            if schema == "select_null":
                return
            raise TypeError(f"parameter {path!r} does not allow null")
        if not isinstance(value, (str, bool, int, float)):
            raise TypeError(f"parameter {path!r} requires a scalar selection")
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError(f"parameter {path!r} requires a finite selection")
        choices = parent.get("range")
        if not isinstance(choices, (tuple, list)):
            raise TypeError(f"parameter {path!r} has no selection range")
        if value not in choices:
            raise ValueError(f"value {value!r} is not allowed by parameter {path!r}")
        return
    raise TypeError(f"parameter {path!r} has unsupported type schema {schema!r}")


def _normalize_replacement(existing: Any, value: Any, path: str) -> Any:
    normalized = _normalize(value, _path=path)
    if existing is None:
        if value is not None:
            raise TypeError(
                f"cannot infer a non-null replacement type for unschematized {path!r}"
            )
        return normalized
    if isinstance(existing, bool):
        valid = isinstance(value, bool)
    elif isinstance(existing, int):
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(existing, float):
        valid = isinstance(value, float) and math.isfinite(value)
    elif isinstance(existing, str):
        valid = isinstance(value, str)
    elif isinstance(existing, Mapping):
        valid = isinstance(value, Mapping)
    elif isinstance(existing, (list, tuple)):
        valid = isinstance(value, (tuple, list))
    else:
        valid = False
    if not valid:
        raise TypeError(
            f"replacement for {path!r} has unexpected type {type(value).__name__}"
        )
    return normalized


def dotted_set(
    mapping: Mapping[str, Any],
    path: Union[str, Sequence[str]],
    value: Any,
) -> dict[str, Any]:
    """Return a detached copy with one existing dotted path replaced.

    No branch is ever created.  A parameter ``.v`` leaf is checked against its
    sibling ``type``/``range`` schema; other leaves must retain their Python
    scalar/container type.
    """

    if not isinstance(mapping, Mapping):
        raise TypeError("dotted path root must be a mapping")
    root = _normalize(mapping)
    parts = _normalize_dotted_path(path)

    current: Any = root
    for index, part in enumerate(parts[:-1]):
        if not isinstance(current, Mapping):
            prefix = ".".join(parts[:index])
            raise TypeError(f"dotted path traverses non-mapping value at {prefix!r}")
        if part not in current:
            raise KeyError(".".join(parts[: index + 1]))
        current = current[part]
    if not isinstance(current, Mapping):
        raise TypeError(f"dotted path parent for {'.'.join(parts)!r} is not a mapping")
    leaf = parts[-1]
    if leaf not in current:
        raise KeyError(".".join(parts))
    if leaf == "v" and "type" in current:
        _validate_parameter_value(current, value, ".".join(parts))
        replacement = _normalize(value, _path=".".join(parts))
    else:
        replacement = _normalize_replacement(
            current[leaf], value, ".".join(parts)
        )

    target = root
    for part in parts[:-1]:
        child = target[part]
        assert isinstance(child, dict)
        target = child
    target[leaf] = replacement
    return root


def _design_relative_path(path: Union[str, Sequence[str]]) -> tuple[str, ...]:
    """Convert a metadata path rooted at ``design`` to an internal path."""

    parts = _normalize_dotted_path(path)
    if len(parts) < 2 or parts[0] != "design":
        raise ValueError("canonical design paths must start with 'design.'")
    return parts[1:]


def design_dotted_get(
    design: Mapping[str, Any], path: Union[str, Sequence[str]]
) -> Any:
    """Read an unwrapped design using its canonical metadata path."""

    return dotted_get(design, _design_relative_path(path))


def design_dotted_set(
    design: Mapping[str, Any],
    path: Union[str, Sequence[str]],
    value: Any,
) -> dict[str, Any]:
    """Update an unwrapped design using a ``design.*`` metadata path."""

    return dotted_set(design, _design_relative_path(path), value)


def dump_yaml_document(document: Mapping[str, Any]) -> bytes:
    """Serialize canonical data with a process-isolated safe dumper."""

    normalized = _normalize(document)
    if not isinstance(normalized, dict):
        raise CanonicalDataError("YAML document must be a mapping")
    text = yaml.dump(
        normalized,
        Dumper=_EditSafeDumper,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
        width=4096,
    )
    return text.encode("utf-8")


def dump_design_yaml(design: Mapping[str, Any]) -> bytes:
    return dump_yaml_document({"design": design})


@dataclass(frozen=True)
class SerializedDesign:
    yaml_bytes: bytes
    design: dict[str, Any]
    canonical_hash: str


def serialize_and_reload_design(design: Mapping[str, Any]) -> SerializedDesign:
    """Create final YAML bytes, reload them, and prove semantic identity."""

    normalized = _normalize(design)
    if not isinstance(normalized, dict):
        raise CanonicalDataError("design must be a mapping")
    expected_hash = canonical_hash(normalized)
    yaml_bytes = dump_design_yaml(normalized)
    reloaded = load_design_yaml_bytes(yaml_bytes)
    reloaded_hash = canonical_hash(reloaded)
    if reloaded_hash != expected_hash:
        raise CanonicalDataError(
            "serialized design changed canonical content: "
            f"{expected_hash} != {reloaded_hash}"
        )
    return SerializedDesign(
        yaml_bytes=yaml_bytes,
        design=reloaded,
        canonical_hash=reloaded_hash,
    )
