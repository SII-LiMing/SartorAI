#!/usr/bin/env python3
"""Archive accepted editing results into ``garmentcode_144k``.

This combines the workflows formerly provided by:

1. ``mv_to_original_dir.py``: move each accepted product to its source sample;
2. ``mv_edit_prompt.py``: extract the accepted natural-language instruction;
3. ``rename.py``: remove ``__edit_NNN`` from archived file names.

For example::

    rand_ABC__edit_002_sim.obj -> rand_ABC_edit_sim.obj

The default mode is a read-only preflight. Pass ``--execute`` to perform all
three phases. The workflow is resumable after interruption and refuses
ambiguous paths, symbolic links, conflicting prompts, and rename collisions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import sys
from typing import Any


GCD_ROOT = Path(
    "/inspire/hdd/global_user/liming-253108120187/ReWeaver2/"
    "ReWeaver2/GarmentCodeRC"
)
ORIGINAL_ROOT = GCD_ROOT / "garmentcode_144k"
TMP_ROOT = GCD_ROOT / "editing" / "tmp"

EXPECTED_BATCH_COUNT = 148
EXPECTED_ACCEPTED_PAIR_COUNT = 12_353

BATCH_RE = re.compile(r"batch_[0-9]+\Z")
OUTPUT_RE = re.compile(r"(batch_[0-9]+)_edit_output\Z")
SOURCE_RE = re.compile(r"rand_[A-Z0-9]+\Z")
TARGET_RE = re.compile(r"(rand_[A-Z0-9]+)__edit_([0-9]+)\Z")
OLD_FILE_RE = re.compile(
    r"^(?P<sample>.+)__edit_(?P<index>[0-9]+)_(?P<suffix>.+)$"
)

REQUIRED_PRODUCT_SUFFIXES = (
    "body_measurements.yaml",
    "boxmesh.obj",
    "boxmesh.ply",
    "design_params.yaml",
    "material.mtl",
    "part_seg.json",
    "pattern.png",
    "pattern.svg",
    "prompts.json",
    "sim.obj",
    "sim.ply",
    "sim_segmentation.txt",
    "specification.json",
    "texture.png",
    "vertex_labels.yaml",
)
CONSISTENCY_FIELDS = (
    "source_sample_id",
    "target_id",
    "specific_instruction",
    "canonical_instruction",
    "instruction_provenance",
    "operation_signature",
)
REFERENCE_SUFFIXES = {".obj", ".mtl"}


class ArchiveError(RuntimeError):
    """The archive operation is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class ArchiveItem:
    batch_id: str
    source_id: str
    target_id: str
    product: Path
    destination: Path
    metadata_path: Path
    prompt_payload: bytes
    already_moved: bool
    prompt_already_written: bool


@dataclass(frozen=True, slots=True)
class RenameItem:
    source: Path
    target: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move accepted products, archive natural-language prompts, and "
            "normalize all archived file names. Defaults to dry-run."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="执行归档；不传时只进行完整预检查",
    )
    return parser.parse_args()


def lexists(path: Path) -> bool:
    return os.path.lexists(path)


def require_directory(path: Path, description: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ArchiveError(f"{description} 不存在或不安全：{path}")


def require_regular_file(
    path: Path,
    description: str,
    *,
    allow_empty: bool = False,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ArchiveError(f"{description} 不存在或不安全：{path}")
    if not allow_empty and path.stat().st_size <= 0:
        raise ArchiveError(f"{description} 是空文件：{path}")


def require_below(path: Path, root: Path, description: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArchiveError(f"{description} 越过允许的根目录：{path}") from exc


def progress(phase: str, completed: int, total: int, label: str = "") -> None:
    width = 34
    ratio = completed / total if total else 1.0
    filled = min(width, int(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    line = f"[{phase}] [{bar}] {completed}/{total} ({ratio * 100:6.2f}%)"
    if label:
        line += f" {label}"
    if sys.stdout.isatty():
        print(f"\r\033[K{line}", end="\n" if completed == total else "", flush=True)
    elif completed == 1 or completed % 250 == 0 or completed == total:
        print(line, flush=True)


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    require_regular_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"{description} 不是有效 JSON：{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArchiveError(f"{description} 必须是 JSON object：{path}")
    return value


def read_pair(line: str, path: Path, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise ArchiveError(f"accepted pair 存在空行：{path}:{line_number}")
    try:
        pair = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ArchiveError(
            f"accepted pair 不是有效 JSON：{path}:{line_number}: {exc}"
        ) from exc
    if not isinstance(pair, dict):
        raise ArchiveError(f"accepted pair 必须是 object：{path}:{line_number}")
    return pair


def make_prompt_payload(instruction: str) -> bytes:
    return (
        json.dumps(
            {"edit_prompt": instruction},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def old_product_name(item: ArchiveItem, suffix: str) -> str:
    return f"{item.target_id}_{suffix}"


def final_product_name(item: ArchiveItem, suffix: str) -> str:
    return f"{item.source_id}_edit_{suffix}"


def old_prompt_name(item: ArchiveItem) -> str:
    return f"{item.target_id}_edit_prompts.json"


def final_prompt_name(item: ArchiveItem) -> str:
    return f"{item.source_id}_edit_edit_prompts.json"


def normalized_name(name: str, source_id: str) -> str:
    match = OLD_FILE_RE.fullmatch(name)
    if match is None:
        return name
    if match.group("sample") != source_id:
        raise ArchiveError(
            f"旧格式文件名前缀与样本目录不一致：{name}，样本为 {source_id}"
        )
    return f"{source_id}_edit_{match.group('suffix')}"


def validate_product_container(container: Path, item: ArchiveItem) -> None:
    require_directory(container, "editing 产物目录")
    for child in container.rglob("*"):
        if child.is_symlink():
            raise ArchiveError(f"editing 产物包含符号链接：{child}")

    for suffix in REQUIRED_PRODUCT_SUFFIXES:
        old_path = container / old_product_name(item, suffix)
        final_path = container / final_product_name(item, suffix)
        existing = [path for path in (old_path, final_path) if lexists(path)]
        if len(existing) != 1:
            raise ArchiveError(
                f"产物应且仅应存在一个旧版或新版文件：{old_path} / {final_path}"
            )
        require_regular_file(existing[0], "editing 产物文件")


def locate_prompt(container: Path, item: ArchiveItem) -> Path | None:
    old_path = container / old_prompt_name(item)
    final_path = container / final_prompt_name(item)
    existing = [path for path in (old_path, final_path) if lexists(path)]
    if len(existing) > 1:
        raise ArchiveError(f"新旧 prompt 文件同时存在：{old_path} / {final_path}")
    if not existing:
        return None
    prompt_path = existing[0]
    require_regular_file(prompt_path, "已归档 prompt")
    if prompt_path.read_bytes() != item.prompt_payload:
        raise ArchiveError(f"已归档 prompt 内容冲突：{prompt_path}")
    return prompt_path


def validate_projected_names(
    container: Path,
    item: ArchiveItem,
    *,
    prompt_exists: bool,
) -> int:
    """Check the final namespace before any file is moved or written."""
    projected: dict[str, str] = {}
    rename_count = 0
    for path in container.iterdir():
        if path.is_symlink():
            raise ArchiveError(f"拒绝归档符号链接：{path}")
        new_name = normalized_name(path.name, item.source_id) if path.is_file() else path.name
        previous = projected.get(new_name)
        if previous is not None and previous != path.name:
            raise ArchiveError(
                f"改名冲突：{container / previous} 和 {path} -> {container / new_name}"
            )
        projected[new_name] = path.name
        rename_count += new_name != path.name

    if not prompt_exists:
        prompt_old = old_prompt_name(item)
        prompt_final = normalized_name(prompt_old, item.source_id)
        previous = projected.get(prompt_final)
        if previous is not None:
            raise ArchiveError(
                f"待写入 prompt 与现有文件冲突：{container / previous}"
            )
        projected[prompt_final] = prompt_old
        rename_count += prompt_old != prompt_final

    return rename_count


def validate_pair(
    pair: dict[str, Any],
    *,
    batch_id: str,
    output_root: Path,
    pair_path: Path,
    line_number: int,
) -> tuple[ArchiveItem, int]:
    location = f"{pair_path}:{line_number}"
    if pair.get("format") != "semantic-edit-pair":
        raise ArchiveError(f"pair format 错误：{location}")
    if pair.get("target_simulation_status") != "pass":
        raise ArchiveError(f"accepted pair 的模拟状态不是 pass：{location}")
    provenance = pair.get("simulation_provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("final_acceptance_status") != "pass"
    ):
        raise ArchiveError(f"accepted pair 缺少最终通过记录：{location}")

    source_path = pair.get("source_path")
    if not isinstance(source_path, str) or source_path != pair.get("source_key"):
        raise ArchiveError(f"source mapping 错误：{location}")
    source_parts = PurePosixPath(source_path).parts
    if len(source_parts) != 3 or source_parts[:2] != (batch_id, "random_body"):
        raise ArchiveError(f"source path 与 batch 不匹配：{location}")
    source_id = source_parts[2]
    if SOURCE_RE.fullmatch(source_id) is None:
        raise ArchiveError(f"source ID 格式错误：{location}")
    if pair.get("source_sample_id") != source_id:
        raise ArchiveError(f"source ID 不一致：{location}")

    target_id = pair.get("target_id")
    if not isinstance(target_id, str):
        raise ArchiveError(f"缺少 target ID：{location}")
    target_match = TARGET_RE.fullmatch(target_id)
    if target_match is None or target_match.group(1) != source_id:
        raise ArchiveError(f"target ID 与 source 不匹配：{location}")
    expected_target_path = (
        PurePosixPath(batch_id) / "random_body" / "data" / target_id
    ).as_posix()
    if pair.get("target_path") != expected_target_path:
        raise ArchiveError(f"target path 错误：{location}")

    instruction = pair.get("specific_instruction")
    if (
        not isinstance(instruction, str)
        or not instruction
        or instruction.strip() != instruction
    ):
        raise ArchiveError(f"specific instruction 无效：{location}")

    metadata_path = (
        output_root
        / batch_id
        / "random_body"
        / "data"
        / target_id
        / f"{target_id}_edit_metadata.json"
    )
    require_below(metadata_path, TMP_ROOT, "edit metadata")
    metadata = read_json_object(metadata_path, "edit metadata")
    mismatched = [
        field
        for field in CONSISTENCY_FIELDS
        if metadata.get(field) != pair.get(field)
    ]
    if mismatched:
        raise ArchiveError(
            f"metadata 与 accepted pair 不一致：{location}: {mismatched}"
        )

    source_directory = ORIGINAL_ROOT / batch_id / "random_body" / source_id
    product = output_root / batch_id / "random_body" / target_id
    destination = source_directory / "edit"
    require_directory(source_directory, "原始样本目录")
    require_below(product, TMP_ROOT, "editing 产物")
    require_below(destination, ORIGINAL_ROOT, "归档目标")

    product_exists = lexists(product)
    destination_exists = lexists(destination)
    if product_exists == destination_exists:
        raise ArchiveError(
            "editing 产物与归档目标必须且只能存在一个："
            f"{product} / {destination}"
        )

    item = ArchiveItem(
        batch_id=batch_id,
        source_id=source_id,
        target_id=target_id,
        product=product,
        destination=destination,
        metadata_path=metadata_path,
        prompt_payload=make_prompt_payload(instruction),
        already_moved=destination_exists,
        prompt_already_written=False,
    )
    container = destination if destination_exists else product
    validate_product_container(container, item)
    if product_exists and product.stat().st_dev != source_directory.stat().st_dev:
        raise ArchiveError(f"产物和目标不在同一文件系统：{product}")

    prompt = locate_prompt(container, item)
    item = ArchiveItem(
        batch_id=item.batch_id,
        source_id=item.source_id,
        target_id=item.target_id,
        product=item.product,
        destination=item.destination,
        metadata_path=item.metadata_path,
        prompt_payload=item.prompt_payload,
        already_moved=item.already_moved,
        prompt_already_written=prompt is not None,
    )
    rename_count = validate_projected_names(
        container,
        item,
        prompt_exists=prompt is not None,
    )
    return item, rename_count


def discover_items() -> tuple[list[ArchiveItem], int]:
    require_directory(GCD_ROOT, "GarmentCodeRC 根目录")
    require_directory(ORIGINAL_ROOT, "原始数据集根目录")
    require_directory(TMP_ROOT, "editing tmp 根目录")

    workspaces = tuple(TMP_ROOT.glob("*/.state/simulation_workspace"))
    if workspaces:
        raise ArchiveError(f"仍存在模拟工作区：{workspaces[0]}")

    output_roots: list[tuple[str, Path]] = []
    for path in TMP_ROOT.iterdir():
        match = OUTPUT_RE.fullmatch(path.name)
        if match is None:
            continue
        require_directory(path, "batch edit output")
        batch_id = match.group(1)
        if BATCH_RE.fullmatch(batch_id) is None:
            raise ArchiveError(f"batch output 名称无效：{path}")
        output_roots.append((batch_id, path))
    output_roots.sort(key=lambda value: int(value[0].split("_", 1)[1]))
    if len(output_roots) != EXPECTED_BATCH_COUNT:
        raise ArchiveError(
            f"预期 {EXPECTED_BATCH_COUNT} 个 batch output，实际 {len(output_roots)} 个"
        )

    items: list[ArchiveItem] = []
    projected_rename_count = 0
    seen_batches: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    seen_destinations: set[Path] = set()
    for batch_id, output_root in output_roots:
        if batch_id in seen_batches:
            raise ArchiveError(f"重复 batch output：{batch_id}")
        seen_batches.add(batch_id)
        pair_path = output_root / batch_id / "accepted_pairs.jsonl"
        require_regular_file(pair_path, "accepted-pair inventory")
        with pair_path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                pair = read_pair(line, pair_path, line_number)
                item, rename_count = validate_pair(
                    pair,
                    batch_id=batch_id,
                    output_root=output_root,
                    pair_path=pair_path,
                    line_number=line_number,
                )
                source_key = (item.batch_id, item.source_id)
                if source_key in seen_sources:
                    raise ArchiveError(
                        f"重复 accepted source：{item.batch_id}/{item.source_id}"
                    )
                if item.destination in seen_destinations:
                    raise ArchiveError(f"重复归档目标：{item.destination}")
                seen_sources.add(source_key)
                seen_destinations.add(item.destination)
                items.append(item)
                projected_rename_count += rename_count
                progress(
                    "preflight",
                    len(items),
                    EXPECTED_ACCEPTED_PAIR_COUNT,
                    f"{item.batch_id}/{item.source_id}",
                )

    if len(items) != EXPECTED_ACCEPTED_PAIR_COUNT:
        raise ArchiveError(
            f"预期 {EXPECTED_ACCEPTED_PAIR_COUNT} 个 accepted pair，实际 {len(items)} 个"
        )
    return items, projected_rename_count


def execute_moves(items: list[ArchiveItem]) -> int:
    pending = [item for item in items if not item.already_moved]
    for index, item in enumerate(pending, start=1):
        if not lexists(item.product) or lexists(item.destination):
            raise ArchiveError(f"预检查后目录状态发生变化：{item.product}")
        validate_product_container(item.product, item)
        item.product.rename(item.destination)
        progress(
            "move",
            index,
            len(pending),
            f"{item.batch_id}/{item.source_id}",
        )
    return len(pending)


def write_new_atomic(path: Path, payload: bytes) -> None:
    if lexists(path):
        raise ArchiveError(f"拒绝覆盖文件：{path}")
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    file_descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(file_descriptor, "wb", closefd=True) as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.link(temporary, path)
    finally:
        if lexists(temporary):
            temporary.unlink()


def execute_prompts(items: list[ArchiveItem]) -> int:
    pending = [item for item in items if not item.prompt_already_written]
    for index, item in enumerate(pending, start=1):
        old_path = item.destination / old_prompt_name(item)
        final_path = item.destination / final_prompt_name(item)
        if lexists(old_path) or lexists(final_path):
            raise ArchiveError(f"预检查后 prompt 状态发生变化：{item.destination}")
        write_new_atomic(old_path, item.prompt_payload)
        progress(
            "prompt",
            index,
            len(pending),
            f"{item.batch_id}/{item.source_id}",
        )
    return len(pending)


def build_rename_plan(items: list[ArchiveItem]) -> list[RenameItem]:
    plan: list[RenameItem] = []
    for index, item in enumerate(items, start=1):
        names: dict[str, str] = {}
        for source in item.destination.iterdir():
            if source.is_symlink():
                raise ArchiveError(f"拒绝改名符号链接：{source}")
            target_name = (
                normalized_name(source.name, item.source_id)
                if source.is_file()
                else source.name
            )
            previous = names.get(target_name)
            if previous is not None and previous != source.name:
                raise ArchiveError(
                    f"改名冲突：{item.destination / previous} 和 {source}"
                )
            names[target_name] = source.name
            if target_name != source.name:
                plan.append(RenameItem(source, item.destination / target_name))
        progress(
            "rename-preflight",
            index,
            len(items),
            f"{item.batch_id}/{item.source_id}",
        )
    return plan


def execute_renames(plan: list[RenameItem]) -> int:
    for index, item in enumerate(plan, start=1):
        if not lexists(item.source) or lexists(item.target):
            raise ArchiveError(f"预检查后改名路径状态发生变化：{item.source}")
        item.source.rename(item.target)
        progress("rename", index, len(plan), item.source.name)
    return len(plan)


def rewrite_references(items: list[ArchiveItem]) -> int:
    changed = 0
    for index, item in enumerate(items, start=1):
        old_prefix_re = re.compile(re.escape(item.source_id) + r"__edit_[0-9]+_")
        new_prefix = f"{item.source_id}_edit_"
        for path in item.destination.iterdir():
            if not path.is_file() or path.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            updated = old_prefix_re.sub(new_prefix, content)
            if updated != content:
                path.write_text(updated, encoding="utf-8")
                changed += 1
        progress(
            "references",
            index,
            len(items),
            f"{item.batch_id}/{item.source_id}",
        )
    return changed


def verify(items: list[ArchiveItem]) -> None:
    for index, item in enumerate(items, start=1):
        if lexists(item.product):
            raise ArchiveError(f"归档后 tmp 产物目录仍然存在：{item.product}")
        require_directory(item.destination, "最终 edit 目录")
        for suffix in REQUIRED_PRODUCT_SUFFIXES:
            require_regular_file(
                item.destination / final_product_name(item, suffix),
                "最终 editing 产物",
            )
        prompt_path = item.destination / final_prompt_name(item)
        require_regular_file(prompt_path, "最终 prompt")
        if prompt_path.read_bytes() != item.prompt_payload:
            raise ArchiveError(f"最终 prompt 内容校验失败：{prompt_path}")

        old_names = [
            path.name
            for path in item.destination.iterdir()
            if path.is_file() and OLD_FILE_RE.fullmatch(path.name)
        ]
        if old_names:
            raise ArchiveError(
                f"归档后仍有旧格式文件名：{item.destination / old_names[0]}"
            )
        for path in item.destination.iterdir():
            if not path.is_file() or path.suffix.lower() not in REFERENCE_SUFFIXES:
                continue
            content = path.read_text(encoding="utf-8")
            old_prefix_re = re.compile(
                re.escape(item.source_id) + r"__edit_[0-9]+_"
            )
            if old_prefix_re.search(content):
                raise ArchiveError(f"OBJ/MTL 仍含旧文件引用：{path}")
        progress(
            "verify",
            index,
            len(items),
            f"{item.batch_id}/{item.source_id}",
        )


def main() -> int:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "chatgarment":
        print("错误：请先激活 conda 环境 chatgarment", file=sys.stderr)
        return 2

    try:
        require_directory(TMP_ROOT, "editing tmp 根目录")
        lock_fd = os.open(TMP_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    except (ArchiveError, OSError) as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        lock_mode = fcntl.LOCK_EX if args.execute else fcntl.LOCK_SH
        try:
            fcntl.flock(lock_fd, lock_mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError("另一个 editing 归档进程正在运行") from exc

        items, projected_renames = discover_items()
        moved = sum(item.already_moved for item in items)
        prompts = sum(item.prompt_already_written for item in items)
        print("Preflight passed")
        print(f"  accepted pairs: {len(items)}")
        print(f"  pending moves: {len(items) - moved}")
        print(f"  pending prompts: {len(items) - prompts}")
        print(f"  projected renames: {projected_renames}")
        print(f"  source root: {TMP_ROOT}")
        print(f"  destination root: {ORIGINAL_ROOT}")

        if not args.execute:
            print("DRY RUN：没有移动、写入或改名任何文件。")
            print("确认后使用 --execute 完成全部归档。")
            return 0

        move_count = execute_moves(items)
        prompt_count = execute_prompts(items)
        rename_plan = build_rename_plan(items)
        rename_count = execute_renames(rename_plan)
        reference_count = rewrite_references(items)
        verify(items)
        print("Archive completed and verified")
        print(f"  moved directories: {move_count}")
        print(f"  written prompts: {prompt_count}")
        print(f"  renamed files: {rename_count}")
        print(f"  updated OBJ/MTL files: {reference_count}")
        print(f"  verified samples: {len(items)}")
        return 0
    except (ArchiveError, OSError, UnicodeError) as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
