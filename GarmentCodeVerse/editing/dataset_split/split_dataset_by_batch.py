#!/usr/bin/env python3
"""Deterministically split every batch, then aggregate the two subsets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import os
from pathlib import Path
import random
import re
from typing import Sequence


DEFAULT_INPUT_ROOT = Path(
    "/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/"
    "GarmentCodeRC/garmentcode_144k"
)
DEFAULT_OUTPUT_DIR = Path(
    "/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/"
    "GarmentCodeRC/editing/dataset_split"
)
BATCH_PATTERN = re.compile(r"batch_([0-9]+)")
SAMPLE_PATTERN = re.compile(r"rand_[A-Za-z0-9][A-Za-z0-9_-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 batch 分层划分 garmentcode 数据集并聚合结果。"
    )
    parser.add_argument(
        "--ratios",
        nargs=2,
        default=("0.15", "0.85"),
        metavar=("RATIO_0", "RATIO_1"),
        help="两个非负比例，默认 0.15 0.85；不要求两者之和为 1。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="每个 batch 内确定性随机划分使用的种子，默认 42。",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"数据集根目录，默认 {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"JSON 和日志目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def parse_ratios(raw_ratios: Sequence[str]) -> tuple[Decimal, Decimal]:
    try:
        ratios = tuple(Decimal(value) for value in raw_ratios)
    except InvalidOperation as exc:
        raise ValueError("ratios 必须是有效数字") from exc
    if len(ratios) != 2 or any(not value.is_finite() for value in ratios):
        raise ValueError("必须提供两个有限比例")
    if any(value < 0 for value in ratios) or sum(ratios) <= 0:
        raise ValueError("比例必须非负且总和必须大于 0")
    return ratios[0], ratios[1]


def ratio_label(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", "_")


def dataset_keys(ratios: tuple[Decimal, Decimal]) -> tuple[str, str]:
    keys = tuple(f"dataset_{ratio_label(value)}" for value in ratios)
    if keys[0] == keys[1]:
        return f"{keys[0]}_0", f"{keys[1]}_1"
    return keys[0], keys[1]


def discover_batches(input_root: Path) -> list[Path]:
    batches: list[tuple[int, Path]] = []
    for path in input_root.iterdir():
        match = BATCH_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"batch 不是安全的普通目录：{path}")
        batches.append((int(match.group(1)), path))
    batches.sort(key=lambda item: item[0])
    if not batches:
        raise ValueError(f"没有发现 batch_<数字> 目录：{input_root}")
    return [path for _, path in batches]


def discover_samples(batch: Path) -> list[Path]:
    random_body = batch / "random_body"
    if random_body.is_symlink() or not random_body.is_dir():
        raise ValueError(f"缺少安全的 random_body 目录：{random_body}")

    roots = [random_body]
    nested_data = random_body / "data"
    if nested_data.exists() or nested_data.is_symlink():
        if nested_data.is_symlink() or not nested_data.is_dir():
            raise ValueError(f"random_body/data 不是安全的普通目录：{nested_data}")
        roots.append(nested_data)

    samples: list[Path] = []
    for root in roots:
        for path in root.iterdir():
            if SAMPLE_PATTERN.fullmatch(path.name) is None:
                continue
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"样本不是安全的普通目录：{path}")
            samples.append(path.absolute())

    samples.sort(key=lambda path: path.as_posix())
    if len(samples) != len(set(samples)):
        raise ValueError(f"batch 中发现重复样本目录：{batch}")
    if not samples:
        raise ValueError(f"batch 中没有发现 rand_* 样本目录：{batch}")
    return samples


def split_count(total: int, ratios: tuple[Decimal, Decimal]) -> int:
    share = ratios[0] / sum(ratios)
    return int((Decimal(total) * share).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    keys = dataset_keys(ratios)
    input_root = args.input_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if input_root.is_symlink() or not input_root.is_dir():
        raise ValueError(f"input root 不是安全的普通目录：{input_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(f"output dir 不是安全的普通目录：{output_dir}")

    name_suffix = "_".join(ratio_label(value) for value in ratios)
    json_path = output_dir / f"dataset_split_{name_suffix}.json"
    log_path = output_dir / f"dataset_split_{name_suffix}.log"

    subsets: dict[str, list[str]] = {keys[0]: [], keys[1]: []}
    all_samples: list[str] = []
    log_lines = [
        f"created_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"input_root={input_root}",
        f"ratios={ratios[0]}:{ratios[1]}",
        f"dataset_keys={keys[0]},{keys[1]}",
        f"seed={args.seed}",
    ]

    batches = discover_batches(input_root)
    for batch in batches:
        samples = discover_samples(batch)
        random.Random(f"{args.seed}:{batch.name}").shuffle(samples)
        first_count = split_count(len(samples), ratios)
        first = samples[:first_count]
        second = samples[first_count:]
        first_paths = [path.as_posix() for path in first]
        second_paths = [path.as_posix() for path in second]
        subsets[keys[0]].extend(first_paths)
        subsets[keys[1]].extend(second_paths)
        all_samples.extend(first_paths)
        all_samples.extend(second_paths)
        log_lines.append(
            f"{batch.name}: total={len(samples)} "
            f"{keys[0]}={len(first)} {keys[1]}={len(second)}"
        )

    first_set = set(subsets[keys[0]])
    second_set = set(subsets[keys[1]])
    all_set = set(all_samples)
    if len(all_samples) != len(all_set):
        raise RuntimeError("全量样本目录存在重复项")
    if first_set & second_set:
        raise RuntimeError("两个数据集存在交集")
    if first_set | second_set != all_set:
        raise RuntimeError("两个数据集的并集不等于全部发现的样本")

    log_lines.extend(
        (
            f"TOTAL: batches={len(batches)} samples={len(all_set)} "
            f"{keys[0]}={len(first_set)} {keys[1]}={len(second_set)}",
            "validation: intersection=0 union_equals_all=true",
            f"json_path={json_path}",
        )
    )
    write_text_atomic(
        json_path,
        json.dumps(subsets, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomic(log_path, "\n".join(log_lines) + "\n")

    print(f"完成：{json_path}")
    print(log_lines[-3])
    print(log_lines[-2])
    print(f"日志：{log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
