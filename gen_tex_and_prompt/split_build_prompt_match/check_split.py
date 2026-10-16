"""Check connected-component counts for GarmentCode simulated meshes."""

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import trimesh
from tqdm import tqdm


DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 64


def parse_batch_list(value: str) -> set[int]:
    batch_ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("batch_list contains an empty item")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise argparse.ArgumentTypeError(f"invalid batch range: {item}")
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid batch range: {item}")
            batch_ids.update(range(start, end + 1))
        elif item.isdigit():
            batch_ids.add(int(item))
        else:
            raise argparse.ArgumentTypeError(f"invalid batch id: {item}")
    return batch_ids


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check simulated mesh component counts before upper/bottom splitting."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing batch_<id> directories.",
    )
    parser.add_argument(
        "--batch_list",
        type=parse_batch_list,
        default=None,
        help="Optional batch selection, for example 0-12 or 0,2,5-7.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Directory for per-batch JSON logs.",
    )
    parser.add_argument("--workers", type=positive_int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete case directories whose meshes fail the check.",
    )
    return parser.parse_args()


def batch_sort_key(path: Path) -> int:
    match = re.fullmatch(r"batch_(\d+)", path.name)
    if match is None:
        raise ValueError(f"invalid batch directory name: {path.name}")
    return int(match.group(1))


def find_batch_dirs(root: Path, batch_ids: set[int] | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"root directory does not exist: {root}")

    if batch_ids is None:
        batch_dirs = [
            path
            for path in root.glob("batch_*")
            if path.is_dir() and re.fullmatch(r"batch_\d+", path.name)
        ]
        batch_dirs.sort(key=batch_sort_key)
    else:
        batch_dirs = [root / f"batch_{batch_id}" for batch_id in sorted(batch_ids)]
        missing = [path for path in batch_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "missing batch directories: " + ", ".join(str(path) for path in missing)
            )

    if not batch_dirs:
        raise FileNotFoundError(f"no batch directories found in {root}")
    return batch_dirs


def face_component_count(mesh: trimesh.Trimesh) -> int:
    components = trimesh.graph.connected_components(
        edges=mesh.face_adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=1,
    )
    return len(components)


def check_mesh(mesh_path: Path) -> tuple[str, dict | None]:
    batch_name = mesh_path.parents[2].name
    case_id = mesh_path.parent.name

    if not mesh_path.is_file():
        return batch_name, {
            "case_id": case_id,
            "error": "missing_sim_obj",
        }

    try:
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        component_count = face_component_count(mesh)
    except Exception as exc:
        return batch_name, {
            "case_id": case_id,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if component_count in (1, 2):
        return batch_name, None
    return batch_name, {
        "case_id": case_id,
        "component_count": component_count,
    }


def write_logs(
    log_root: Path,
    batch_dirs: list[Path],
    unexpected: dict[str, list[dict]],
) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    for batch_dir in batch_dirs:
        records = sorted(
            unexpected[batch_dir.name], key=lambda item: item["case_id"]
        )
        (log_root / f"{batch_dir.name}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def delete_unexpected_cases(
    batch_dirs: list[Path],
    unexpected: dict[str, list[dict]],
) -> tuple[int, int]:
    deleted = 0
    failed = 0
    for batch_dir in batch_dirs:
        case_root = (batch_dir / "random_body").resolve()
        for record in unexpected[batch_dir.name]:
            case_dir = case_root / record["case_id"]
            try:
                if case_dir.parent != case_root or not case_dir.name.startswith("rand_"):
                    raise ValueError(f"refusing to delete unsafe path: {case_dir}")
                if not case_dir.is_dir():
                    raise FileNotFoundError(f"case directory does not exist: {case_dir}")
                shutil.rmtree(case_dir)
                record["deleted"] = True
                deleted += 1
            except Exception as exc:
                record["deleted"] = False
                record["delete_error"] = f"{type(exc).__name__}: {exc}"
                failed += 1
    return deleted, failed


def main() -> None:
    args = parse_args()
    batch_dirs = find_batch_dirs(args.root, args.batch_list)
    mesh_paths = [
        case_dir / f"{case_dir.name}_sim.obj"
        for batch_dir in batch_dirs
        for case_dir in sorted((batch_dir / "random_body").glob("rand_*"))
        if case_dir.is_dir()
    ]
    unexpected = {batch_dir.name: [] for batch_dir in batch_dirs}

    # Recycling workers bounds native allocations retained by trimesh/NumPy.
    with mp.Pool(args.workers, maxtasksperchild=MAX_TASKS_PER_WORKER) as pool:
        results = pool.imap_unordered(check_mesh, mesh_paths, chunksize=1)
        for batch_name, record in tqdm(results, total=len(mesh_paths), unit="mesh"):
            if record is not None:
                unexpected[batch_name].append(record)

    unexpected_cases = sum(len(records) for records in unexpected.values())
    write_logs(args.log, batch_dirs, unexpected)

    deleted_cases = 0
    delete_failures = 0
    if args.delete:
        deleted_cases, delete_failures = delete_unexpected_cases(
            batch_dirs, unexpected
        )
        write_logs(args.log, batch_dirs, unexpected)

    print(f"total cases: {len(mesh_paths)}")
    print(f"unexpected cases: {unexpected_cases}")
    if args.delete:
        print(f"deleted cases: {deleted_cases}")
        print(f"delete failures: {delete_failures}")
    print(f"logs: {args.log}")

    if delete_failures:
        raise RuntimeError(
            f"failed to delete {delete_failures} case directories; see logs in {args.log}"
        )


if __name__ == "__main__":
    main()
