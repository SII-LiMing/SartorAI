#!/usr/bin/env python3
"""Filter GarmentCode cases by simulation log failures and bust coverage.

By default the final failed cases are deleted in place after reports are built.
dataset_properties_*.yaml files are never modified by this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GCD_DIR = SCRIPT_DIR / "garmentcode_v2_100"
BODY_TYPES = ("default_body", "random_body")
BODY_SPLITS = set(BODY_TYPES)
HARDCODED_NIPPLE_POINTS = {
    "mean_all.obj": {
        "left": [0.0894095302, 1.244089745, 0.1275354475],
        "right": [-0.0798654109, 1.244302772, 0.1296166927],
    },
    "mean_all_apart.obj": {
        "left": [0.0776688457, 1.2440347169, 0.1278667897],
        "right": [-0.080494456, 1.2448708509, 0.1280961633],
    },
}


def is_case_name(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("rand_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report failed GarmentCode cases from dataset_properties logs and "
            "from uncovered bust/nipple points."
        )
    )
    parser.add_argument("--gcd_dir", default=str(DEFAULT_GCD_DIR))
    parser.add_argument("--body_type", default="default_body", choices=BODY_TYPES)
    parser.add_argument(
        "--dataset_properties",
        default=None,
        help="Optional explicit dataset_properties_*.yaml path.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Only check the named case. Can be passed multiple times.",
    )
    parser.add_argument("--max_cases", type=int, default=0, help="Debug limit; 0 means all cases.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional report directory. Defaults to the selected body directory.",
    )
    parser.add_argument(
        "--no_write",
        action="store_true",
        help="Print summary only; do not write report files. This does not disable deletion.",
    )
    parser.add_argument(
        "--no-delete",
        dest="no_delete",
        action="store_true",
        help="Only write filter_fail_report.yaml; do not delete filtered files.",
    )
    parser.add_argument(
        "--report_only",
        dest="no_delete",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Build the delete plan but do not remove files.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale garment PLY vertices before coverage checks. Current render path uses 0.01.",
    )
    parser.add_argument(
        "--coverage_max_distance",
        type=float,
        default=0.08,
        help="Max forward distance in meters from nipple point to garment hit.",
    )
    parser.add_argument(
        "--ray_eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for ray-triangle tests.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    return data if isinstance(data, dict) else {}


def strip_yaml_scalar(value: str) -> str | None:
    value = value.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value or None


def read_top_level_yaml_scalar(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    prefix = f"{key}:"
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            if raw_line.startswith(prefix):
                return strip_yaml_scalar(raw_line.split(":", 1)[1])
    return None


def resolve_existing_path(value: str | None, base: Path, fallback: Path) -> Path:
    if not value:
        return fallback.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_system_config() -> dict[str, Any]:
    path = SCRIPT_DIR / "system.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data if isinstance(data, dict) else {}


def infer_body_root(case_dir: Path, body_type: str, gcd_dir: Path) -> Path:
    for parent in [case_dir, *case_dir.parents]:
        if parent.name in BODY_SPLITS:
            return parent
    return gcd_dir / body_type


def default_dataset_properties_path(gcd_dir: Path, body_type: str) -> Path:
    body_root = gcd_dir / body_type
    candidates = [
        body_root / f"dataset_properties_{body_type}.yaml",
        body_root / "dataset_properties.yaml",
        gcd_dir / f"dataset_properties_{body_type}.yaml",
        gcd_dir / "dataset_properties.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def collect_failures_from_log(props: dict[str, Any]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = defaultdict(list)

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            if "fails" in node and isinstance(node["fails"], dict):
                for fail_name, cases in node["fails"].items():
                    if isinstance(cases, list):
                        reason = ".".join((*path, "fails", str(fail_name)))
                        for case_name in cases:
                            if is_case_name(case_name):
                                failures[case_name].append(reason)
            for key, value in node.items():
                if key == "fails":
                    continue
                walk(value, (*path, str(key)))

    walk(props, tuple())
    return {case: sorted(set(reasons)) for case, reasons in failures.items()}


def collect_case_names(props: dict[str, Any], body_root: Path, explicit_cases: list[str]) -> list[str]:
    if explicit_cases:
        return sorted(set(explicit_cases))

    case_names: set[str] = set()
    generator_stats = props.get("generator", {}).get("stats", {})
    if isinstance(generator_stats, dict):
        panel_count = generator_stats.get("panel_count")
        if isinstance(panel_count, dict):
            case_names.update(str(key) for key in panel_count if is_case_name(key))

    sim_stats = props.get("sim", {}).get("stats", {})
    if isinstance(sim_stats, dict):
        for key, value in sim_stats.items():
            if isinstance(value, dict):
                if key == "fails":
                    continue
                case_names.update(str(item) for item in value if is_case_name(item))
            elif isinstance(value, list):
                case_names.update(str(item) for item in value if is_case_name(item))

    for sim_path in body_root.rglob("*_sim.ply"):
        if sim_path.parent.name == "renders":
            continue
        case_names.add(sim_path.stem.replace("_sim", ""))

    return sorted(case_names)


def find_case_dir(body_root: Path, case_name: str) -> Path | None:
    candidates = [
        body_root / case_name,
        body_root / "data" / case_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for sim_path in body_root.rglob(f"{case_name}_sim.ply"):
        return sim_path.parent
    for candidate in body_root.rglob(case_name):
        if candidate.is_dir():
            return candidate
    return None


def find_case_file(case_dir: Path, case_name: str, suffix: str) -> Path | None:
    candidates = [
        case_dir / f"{case_name}_{suffix}",
        case_dir / suffix,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(case_dir.glob(f"*_{suffix}"))
    return matches[0] if matches else None


def find_body_measurements(case_dir: Path, case_name: str) -> Path | None:
    candidates = [
        case_dir / f"{case_name}_body_measurements.yaml",
        case_dir / "body_measurements.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(case_dir.glob("*body_measurements.yaml"))
    return matches[0] if matches else None


def body_obj_filename(body_sample: str) -> str:
    sample_path = Path(body_sample)
    if sample_path.suffix == ".obj":
        return sample_path.name
    return f"{sample_path.stem}.obj"


def resolve_body_obj(
    *,
    body_params: dict[str, Any],
    body_type: str,
    body_root: Path,
    dataset_props_path: Path,
) -> Path:
    body_sample = body_params.get("body_sample")
    if not body_sample:
        body_sample = read_top_level_yaml_scalar(dataset_props_path, "body_default")
    if not body_sample:
        raise ValueError("No body_sample in body measurements and no body_default in dataset properties.")

    body_sample = str(body_sample)
    sample_path = Path(body_sample).expanduser()
    if sample_path.is_absolute() and sample_path.suffix == ".obj":
        body_obj = sample_path
    else:
        system_config = load_system_config()
        body_filename = body_obj_filename(body_sample)
        if body_type == "random_body":
            body_samples = read_top_level_yaml_scalar(dataset_props_path, "body_samples")
            if body_samples:
                samples_path = Path(body_samples).expanduser()
                if not samples_path.is_absolute():
                    samples_base = resolve_existing_path(
                        system_config.get("body_samples_path"),
                        SCRIPT_DIR,
                        SCRIPT_DIR,
                    )
                    samples_path = samples_base / samples_path
                bodies_dir = samples_path / "meshes"
            else:
                bodies_dir = body_root / "meshes"
        else:
            bodies_dir = resolve_existing_path(
                system_config.get("bodies_default_path"),
                SCRIPT_DIR,
                SCRIPT_DIR / "assets" / "bodies",
            )
        body_obj = bodies_dir / body_filename

    body_obj = body_obj.resolve()
    if not body_obj.exists():
        raise FileNotFoundError(body_obj)
    return body_obj


def load_garment_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh from {path}, got {type(mesh)}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Empty garment mesh: {path}")
    return vertices, faces


def ray_triangle_hits(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    max_distance: float,
    eps: float,
) -> tuple[int, float | None]:
    tri = vertices[faces]
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(direction, edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(a) > eps
    inv_a = np.zeros_like(a)
    inv_a[valid] = 1.0 / a[valid]

    s = origin - v0
    u = inv_a * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inv_a * np.einsum("j,ij->i", direction, q)
    t = inv_a * np.einsum("ij,ij->i", edge2, q)

    hit = (
        valid
        & (u >= -eps)
        & (v >= -eps)
        & ((u + v) <= 1.0 + eps)
        & (t >= -eps)
        & (t <= max_distance)
    )
    if not np.any(hit):
        return 0, None
    hit_t = t[hit]
    return int(hit_t.size), float(hit_t.min())


def check_nipple_coverage(
    *,
    garment_vertices: np.ndarray,
    garment_faces: np.ndarray,
    nipple_points: dict[str, list[float]],
    max_distance: float,
    eps: float,
) -> dict[str, dict[str, Any]]:
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}
    for side, point in nipple_points.items():
        origin = np.asarray(point, dtype=np.float64)
        hit_count, min_distance = ray_triangle_hits(
            garment_vertices,
            garment_faces,
            origin,
            direction,
            max_distance=max_distance,
            eps=eps,
        )
        result[side] = {
            "covered": hit_count > 0,
            "hit_count": hit_count,
            "min_forward_distance": min_distance,
        }
    return result


def check_case(
    *,
    case_name: str,
    case_dir: Path | None,
    body_type: str,
    body_root: Path,
    dataset_props_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if case_dir is None:
        return {"status": "error", "error": "case_dir_not_found"}

    sim_path = find_case_file(case_dir, case_name, "sim.ply")
    measurements_path = find_body_measurements(case_dir, case_name)
    if sim_path is None:
        return {"status": "error", "case_dir": str(case_dir), "error": "sim_ply_not_found"}
    if measurements_path is None:
        return {"status": "error", "case_dir": str(case_dir), "sim_ply": str(sim_path), "error": "body_measurements_not_found"}

    try:
        measurements = load_yaml(measurements_path)
        body_params = measurements.get("body", measurements)
        if not isinstance(body_params, dict):
            raise ValueError(f"No body section in {measurements_path}")
        body_obj = resolve_body_obj(
            body_params=body_params,
            body_type=body_type,
            body_root=body_root,
            dataset_props_path=dataset_props_path,
        )
        nipple_points = HARDCODED_NIPPLE_POINTS[body_obj.name]
        garment_vertices, garment_faces = load_garment_mesh(sim_path, args.scale)
        coverage = check_nipple_coverage(
            garment_vertices=garment_vertices,
            garment_faces=garment_faces,
            nipple_points=nipple_points,
            max_distance=args.coverage_max_distance,
            eps=args.ray_eps,
        )
        uncovered_sides = sorted(side for side, value in coverage.items() if not value["covered"])
        status = "uncovered" if uncovered_sides else "covered"
        return {
            "status": status,
            "case_dir": str(case_dir),
            "sim_ply": str(sim_path),
            "body_measurements": str(measurements_path),
            "body_obj": str(body_obj),
            "body_sample": body_params.get("body_sample"),
            "nipple_points": nipple_points,
            "coverage": coverage,
            "uncovered_sides": uncovered_sides,
        }
    except Exception as exc:
        return {
            "status": "error",
            "case_dir": str(case_dir),
            "sim_ply": str(sim_path),
            "body_measurements": str(measurements_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_report = {
        "log_failed_cases": report["log_failed_cases"],
        "nipple_uncovered_cases": report["nipple_uncovered_cases"],
        "log_failures_by_case": report["final_fail_reasons"],
    }
    yaml_report_path = output_dir / "filter_fail_report.yaml"
    yaml_report_path.write_text(
        yaml.safe_dump(compact_report, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def deletion_candidates_for_case(
    *,
    body_root: Path,
    case_name: str,
    case_dir: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        if path.name.startswith("dataset_properties"):
            return
        if not path.exists():
            return
        seen.add(resolved)
        candidates.append(path)

    if case_dir is not None:
        add(case_dir)

    for path in body_root.glob(f"{case_name}*"):
        add(path)

    for subdir_name in ("data", "patterns_vis", "renders"):
        subdir = body_root / subdir_name
        if not subdir.exists():
            continue
        direct_case_dir = subdir / case_name
        add(direct_case_dir)
        for path in subdir.glob(f"{case_name}*"):
            add(path)

    return sorted(candidates, key=lambda path: (len(path.parts), str(path)), reverse=True)


def apply_deletion(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    body_root = Path(report["body_root"]).resolve()
    deletion: dict[str, Any] = {
        "enabled": not args.no_delete,
        "dry_run": bool(args.dry_run),
        "body_root": str(body_root),
        "deleted_cases": [],
        "planned_cases": [],
        "cases": {},
        "errors": [],
    }

    if args.no_delete:
        return deletion

    for case_name in report["final_failed_cases"]:
        case_info = report["nipple_checks"].get(case_name, {})
        case_dir_value = case_info.get("case_dir")
        case_dir = Path(case_dir_value) if isinstance(case_dir_value, str) else None
        candidates = deletion_candidates_for_case(
            body_root=body_root,
            case_name=case_name,
            case_dir=case_dir,
        )
        case_record = {
            "paths": [str(path.resolve()) for path in candidates],
            "deleted_paths": [],
            "planned_paths": [],
            "errors": [],
        }

        for path in candidates:
            resolved = path.resolve()
            if not is_relative_to(resolved, body_root):
                message = f"Refuse to delete outside body root: {resolved}"
                case_record["errors"].append(message)
                deletion["errors"].append({"case": case_name, "error": message})
                continue
            if resolved.name.startswith("dataset_properties"):
                message = f"Refuse to delete dataset properties file: {resolved}"
                case_record["errors"].append(message)
                deletion["errors"].append({"case": case_name, "error": message})
                continue

            if args.dry_run:
                case_record["planned_paths"].append(str(resolved))
                continue

            try:
                if resolved.is_dir():
                    shutil.rmtree(resolved)
                else:
                    resolved.unlink()
                case_record["deleted_paths"].append(str(resolved))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                case_record["errors"].append(message)
                deletion["errors"].append({"case": case_name, "path": str(resolved), "error": message})

        if case_record["planned_paths"]:
            deletion["planned_cases"].append(case_name)
        if case_record["deleted_paths"]:
            deletion["deleted_cases"].append(case_name)
        deletion["cases"][case_name] = case_record

    deletion["planned_paths_count"] = sum(
        len(record["planned_paths"]) for record in deletion["cases"].values()
    )
    deletion["deleted_paths_count"] = sum(
        len(record["deleted_paths"]) for record in deletion["cases"].values()
    )
    deletion["error_count"] = len(deletion["errors"])
    return deletion


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    gcd_dir = Path(args.gcd_dir).expanduser().resolve()
    body_root = gcd_dir / args.body_type
    if not body_root.exists():
        raise FileNotFoundError(body_root)

    dataset_props_path = (
        Path(args.dataset_properties).expanduser().resolve()
        if args.dataset_properties
        else default_dataset_properties_path(gcd_dir, args.body_type)
    )
    if not dataset_props_path.exists():
        raise FileNotFoundError(dataset_props_path)

    props = load_yaml(dataset_props_path)
    log_failures = collect_failures_from_log(props)
    case_names = collect_case_names(props, body_root, args.case)
    if args.max_cases:
        case_names = case_names[: args.max_cases]

    nipple_checks: dict[str, dict[str, Any]] = {}
    final_fail_reasons: dict[str, list[str]] = {}
    log_failed_cases = sorted(case for case in case_names if case in log_failures)

    for index, case_name in enumerate(case_names, start=1):
        print(f"[{index}/{len(case_names)}] checking {case_name}", flush=True)
        case_dir = find_case_dir(body_root, case_name)
        if case_dir is not None:
            body_root_for_case = infer_body_root(case_dir, args.body_type, gcd_dir)
        else:
            body_root_for_case = body_root

        check = check_case(
            case_name=case_name,
            case_dir=case_dir,
            body_type=args.body_type,
            body_root=body_root_for_case,
            dataset_props_path=dataset_props_path,
            args=args,
        )
        nipple_checks[case_name] = check

        reasons = list(log_failures.get(case_name, []))
        if check["status"] == "uncovered":
            reasons.append("nipple_uncovered:" + ",".join(check.get("uncovered_sides", [])))

        if reasons:
            final_fail_reasons[case_name] = sorted(set(reasons))

    final_failed_cases = sorted(final_fail_reasons)
    passed_cases = sorted(case for case in case_names if case not in final_fail_reasons)
    nipple_uncovered_cases = sorted(
        case for case, check in nipple_checks.items() if check.get("status") == "uncovered"
    )
    nipple_error_cases = sorted(
        case for case, check in nipple_checks.items() if check.get("status") == "error"
    )

    return {
        "schema_version": 1,
        "gcd_dir": str(gcd_dir),
        "body_type": args.body_type,
        "body_root": str(body_root),
        "dataset_properties": str(dataset_props_path),
        "parameters": {
            "scale": args.scale,
            "coverage_max_distance": args.coverage_max_distance,
            "front_direction": "+z",
            "no_delete": args.no_delete,
            "dry_run": args.dry_run,
        },
        "summary": {
            "total_cases": len(case_names),
            "log_failed_cases": len(log_failed_cases),
            "nipple_uncovered_cases": len(nipple_uncovered_cases),
            "nipple_check_error_cases": len(nipple_error_cases),
            "final_failed_cases": len(final_failed_cases),
            "passed_cases": len(passed_cases),
        },
        "case_names": case_names,
        "log_failed_cases": log_failed_cases,
        "log_failures_by_case": {case: log_failures[case] for case in log_failed_cases},
        "nipple_uncovered_cases": nipple_uncovered_cases,
        "nipple_check_error_cases": nipple_error_cases,
        "nipple_checks": nipple_checks,
        "final_failed_cases": final_failed_cases,
        "passed_cases": passed_cases,
        "final_fail_reasons": final_fail_reasons,
    }


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except Exception as exc:
        print(f"filter_fail.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    deletion = apply_deletion(report, args)
    report["deletion"] = deletion

    summary = report["summary"]
    print("\nSummary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("Deletion")
    print(f"  enabled: {deletion['enabled']}")
    print(f"  dry_run: {deletion['dry_run']}")
    print(f"  planned_paths_count: {deletion.get('planned_paths_count', 0)}")
    print(f"  deleted_paths_count: {deletion.get('deleted_paths_count', 0)}")
    print(f"  error_count: {deletion.get('error_count', 0)}")

    if not args.no_write:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else Path(report["body_root"]).resolve()
        )
        write_report(output_dir, report)
        print(f"\nWrote report to: {output_dir}")
    return 2 if deletion.get("error_count", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
