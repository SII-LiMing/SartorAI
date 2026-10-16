#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path


PARTS = ("upper", "bottom", "whole")
REQUIRED_RESULT_FILES = (
    "results.obj",
    "results.mtl",
    "results.jpg",
    "results_metallic.jpg",
    "results_roughness.jpg",
)
OUTPUT_DIR_PATTERN = re.compile(r"NI-Tex_output(?:_regen(\d+))?")
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)


@dataclass(frozen=True)
class CaseResult:
    has_outputs: bool
    copied_parts: int = 0
    fallback_messages: tuple[str, ...] = ()
    source_dirs: tuple[Path, ...] = ()
    error: str | None = None


class ProgressBar:
    def __init__(self, label: str, total: int, width: int = 30) -> None:
        self.label = label
        self.total = total
        self.width = width
        self.start_time = time.monotonic()
        self.last_update = 0.0
        self._render(0)

    def update(self, completed: int) -> None:
        now = time.monotonic()
        if completed < self.total and now - self.last_update < 0.1:
            return
        self.last_update = now
        self._render(completed)

    def _render(self, completed: int) -> None:
        ratio = completed / self.total if self.total else 1.0
        filled = round(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.monotonic() - self.start_time, 1e-6)
        rate = completed / elapsed
        line = (
            f"\r{self.label:<8} [{bar}] {completed}/{self.total} "
            f"({ratio:6.2%}) {rate:,.1f}/s"
        )
        print(line, end="\n" if completed >= self.total else "", flush=True)


def parse_batch_list(value: str) -> list[int]:
    batch_ids: set[int] = set()
    for token in re.split(r"[,\s]+", value.strip()):
        match = re.fullmatch(r"(?:batch_)?(\d+)(?:-(?:batch_)?(\d+))?", token)
        if match is None:
            raise argparse.ArgumentTypeError(f"Invalid batch ID/range: {token!r}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start > end:
            raise argparse.ArgumentTypeError(f"Reversed batch range: {token}")
        batch_ids.update(range(start, end + 1))
    return sorted(batch_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the latest complete NI-Tex result for each garment part, then "
            "delete intermediate generation directories after all copies succeed."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Dataset root containing batch_*/random_body/rand_* cases.",
    )
    parser.add_argument(
        "--batch_list",
        type=parse_batch_list,
        help="Batch IDs/ranges, e.g. 148-171 or 148,150-153 (default: all batches).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker processes (default: {DEFAULT_WORKERS}).",
    )
    return parser.parse_args()


def find_cases(data_root: Path, batch_ids: list[int] | None = None) -> list[Path]:
    if batch_ids is None:
        batches = list(data_root.glob("batch_*"))
    else:
        batches = [data_root / f"batch_{batch_id}" for batch_id in batch_ids]
        missing = [str(batch / "random_body") for batch in batches
                   if not (batch / "random_body").is_dir()]
        if missing:
            raise ValueError("Missing random_body directories: " + ", ".join(missing))
    return sorted(
        path
        for batch in batches
        for path in (batch / "random_body").glob("rand_*")
        if path.is_dir() and not path.is_symlink()
    )


def find_output_dirs(case_dir: Path) -> list[tuple[int, Path]]:
    outputs: list[tuple[int, Path]] = []
    for path in case_dir.iterdir():
        match = OUTPUT_DIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"NI-Tex output is not a regular directory: {path}")
        generation = int(match.group(1)) if match.group(1) is not None else 0
        outputs.append((generation, path))
    return sorted(outputs, key=lambda item: (item[0], item[1].name), reverse=True)


def is_complete(part_dir: Path) -> bool:
    return all((part_dir / name).is_file() for name in REQUIRED_RESULT_FILES)


def select_parts(
    case_dir: Path, outputs: list[tuple[int, Path]]
) -> tuple[dict[str, tuple[int, Path]], list[str], tuple[str, ...]]:
    existing_parts: set[str] = set()
    for _, output_dir in outputs:
        for part in PARTS:
            part_dir = output_dir / part
            if part_dir.exists() or part_dir.is_symlink():
                existing_parts.add(part)

    selected: dict[str, tuple[int, Path]] = {}
    unresolved: list[str] = []
    fallback_messages: list[str] = []
    for part in PARTS:
        if part not in existing_parts:
            continue

        candidates = [
            (generation, output_dir / part)
            for generation, output_dir in outputs
            if (output_dir / part).is_dir()
            and not (output_dir / part).is_symlink()
        ]
        complete = [candidate for candidate in candidates if is_complete(candidate[1])]
        if not complete:
            unresolved.append(part)
            continue

        selected[part] = complete[0]
        if candidates and complete[0][1] != candidates[0][1]:
            fallback_messages.append(
                f"{case_dir}: latest {part} is incomplete; using {complete[0][1]}"
            )

    return selected, unresolved, tuple(fallback_messages)


def replace_textured(
    case_dir: Path,
    selected: dict[str, tuple[int, Path]],
) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".textured.tmp.", dir=case_dir))
    try:
        for part in PARTS:
            if part in selected:
                shutil.copytree(selected[part][1], temporary / part)

        textured = case_dir / "textured"
        if textured.is_symlink() or textured.is_file():
            textured.unlink()
        elif textured.is_dir():
            shutil.rmtree(textured)
        temporary.rename(textured)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def process_case(case_dir: Path) -> CaseResult:
    try:
        outputs = find_output_dirs(case_dir)
        if not outputs:
            return CaseResult(has_outputs=False)

        selected, unresolved, fallback_messages = select_parts(case_dir, outputs)
        if unresolved:
            return CaseResult(
                has_outputs=True,
                error=f"no complete output for {', '.join(unresolved)}",
            )
        if not selected:
            return CaseResult(
                has_outputs=True,
                error="no upper, bottom, or whole output found",
            )

        replace_textured(case_dir, selected)
        return CaseResult(
            has_outputs=True,
            copied_parts=len(selected),
            fallback_messages=fallback_messages,
            source_dirs=tuple(path for _, path in outputs),
        )
    except (OSError, ValueError) as exc:
        return CaseResult(has_outputs=True, error=str(exc))


def bounded_futures(executor, function, items, max_pending):
    iterator = iter(items)
    pending = {}

    for _ in range(max_pending):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending[executor.submit(function, item)] = item

    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            item = pending.pop(future)
            yield item, future
            try:
                next_item = next(iterator)
            except StopIteration:
                continue
            pending[executor.submit(function, next_item)] = next_item


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        print(f"error: data_root is not a directory: {data_root}", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("error: workers must be a positive integer", file=sys.stderr)
        return 2

    try:
        cases = find_cases(data_root, args.batch_list)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not cases:
        print(f"error: no case directories found under {data_root}", file=sys.stderr)
        return 2

    cases_with_outputs = 0
    aggregated_cases = 0
    copied_parts = 0
    fallback_parts = 0
    sources_to_delete: list[Path] = []
    deleted_dirs = 0
    warnings: list[str] = []
    errors: list[str] = []

    print(f"Found {len(cases)} cases; using {args.workers} worker processes.")
    completed_cases = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        progress = ProgressBar("Gather", len(cases))
        for case_dir, future in bounded_futures(
            executor, process_case, cases, args.workers * 4
        ):
            try:
                result = future.result()
            except Exception as exc:
                errors.append(f"{case_dir}: worker failed: {exc}")
            else:
                if result.has_outputs:
                    cases_with_outputs += 1
                if result.error is not None:
                    errors.append(f"{case_dir}: {result.error}")
                elif result.has_outputs:
                    aggregated_cases += 1
                    copied_parts += result.copied_parts
                    fallback_parts += len(result.fallback_messages)
                    warnings.extend(result.fallback_messages)
                    sources_to_delete.extend(result.source_dirs)
            completed_cases += 1
            progress.update(completed_cases)

        # Cleanup starts only after every copy has finished successfully.
        if errors:
            print("Cleanup skipped because gathering failed.", file=sys.stderr)
        elif sources_to_delete:
            progress = ProgressBar("Delete", len(sources_to_delete))
            for completed, (output_dir, future) in enumerate(
                bounded_futures(executor, shutil.rmtree, sources_to_delete, args.workers * 4),
                start=1,
            ):
                try:
                    future.result()
                    deleted_dirs += 1
                except Exception as exc:
                    errors.append(f"failed to delete {output_dir}: {exc}")
                progress.update(completed)

    for warning in sorted(warnings):
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"cases={len(cases)}, cases_with_outputs={cases_with_outputs}, "
        f"aggregated_cases={aggregated_cases}, copied_parts={copied_parts}, "
        f"fallback_parts={fallback_parts}, deleted_dirs={deleted_dirs}"
    )
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
