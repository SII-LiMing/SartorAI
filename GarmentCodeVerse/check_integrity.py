#!/usr/bin/env python3
"""Check integrity of one GarmentCode batch after filtering."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GCD_DIR = SCRIPT_DIR / "garmentcode_v2_100"
BODY_TYPES = ("default_body", "random_body")
DEFAULT_REQUIRED_SUFFIXES = (
    "specification.json",
    "design_params.yaml",
    "body_measurements.yaml",
    "boxmesh.obj",
    "boxmesh.ply",
    "sim.obj",
    "sim.ply",
    "sim_segmentation.txt",
    "texture.png",
    "material.mtl",
    "part_seg.json",
    "vertex_labels.yaml",
    "pattern.svg",
    "pattern.png",
)


def is_case_name(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("rand_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check case counts and required simulation outputs for one batch."
    )
    parser.add_argument("--gcd_dir", default=str(DEFAULT_GCD_DIR), help="Batch directory.")
    parser.add_argument("--body_type", default="default_body", choices=BODY_TYPES)
    parser.add_argument(
        "--expected_count",
        type=int,
        default=0,
        help="Expected original sample count for this batch. 0 means infer from metadata.",
    )
    parser.add_argument(
        "--dataset_properties",
        default=None,
        help="Optional explicit dataset_properties_*.yaml path.",
    )
    parser.add_argument(
        "--filter_report",
        default=None,
        help="Optional explicit filter_fail_report.yaml path.",
    )
    parser.add_argument(
        "--require_filter_report",
        action="store_true",
        help="Fail if filter_fail_report.yaml is missing.",
    )
    parser.add_argument(
        "--required_suffix",
        action="append",
        default=[],
        help="Required case file suffix. Can be passed multiple times. Defaults to the built-in list.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional report directory. Defaults to the selected body directory.",
    )
    parser.add_argument("--no_write", action="store_true", help="Do not write integrity_report.yaml.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    return data if isinstance(data, dict) else {}


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


def default_filter_report_path(gcd_dir: Path, body_type: str) -> Path:
    return gcd_dir / body_type / "filter_fail_report.yaml"


def collect_sample_cases(props: dict[str, Any], body_root: Path) -> tuple[set[str], str]:
    case_names: set[str] = set()
    generator_stats = props.get("generator", {}).get("stats", {})
    if isinstance(generator_stats, dict):
        panel_count = generator_stats.get("panel_count")
        if isinstance(panel_count, dict):
            case_names.update(str(key) for key in panel_count if is_case_name(key))
            if case_names:
                return case_names, "generator.stats.panel_count"

    data_dir = body_root / "data"
    if data_dir.is_dir():
        case_names.update(path.name for path in data_dir.iterdir() if path.is_dir() and is_case_name(path.name))
        if case_names:
            return case_names, "data directories"

    return case_names, "none"


def collect_failures_from_dataset_props(props: dict[str, Any]) -> dict[str, list[str]]:
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


def collect_failures_from_filter_report(report: dict[str, Any]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = defaultdict(list)

    log_failures = report.get("log_failures_by_case")
    if isinstance(log_failures, dict):
        for case_name, reasons in log_failures.items():
            if not is_case_name(case_name):
                continue
            if isinstance(reasons, list):
                failures[case_name].extend(str(reason) for reason in reasons)
            else:
                failures[case_name].append("filter_report.log_failures_by_case")

    for key, reason in (
        ("log_failed_cases", "filter_report.log_failed_cases"),
        ("nipple_uncovered_cases", "filter_report.nipple_uncovered_cases"),
    ):
        cases = report.get(key)
        if isinstance(cases, list):
            for case_name in cases:
                if is_case_name(case_name):
                    failures[case_name].append(reason)

    return {case: sorted(set(reasons)) for case, reasons in failures.items()}


def list_direct_case_dirs(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {child.name for child in path.iterdir() if child.is_dir() and is_case_name(child.name)}


def check_case_files(case_dir: Path, case_name: str, required_suffixes: list[str]) -> list[str]:
    missing: list[str] = []
    for suffix in required_suffixes:
        candidate = case_dir / f"{case_name}_{suffix}"
        if not candidate.is_file():
            missing.append(f"{case_name}_{suffix}")
        elif candidate.stat().st_size <= 0:
            missing.append(f"{case_name}_{suffix}:empty")
    return missing


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    gcd_dir = Path(args.gcd_dir).expanduser().resolve()
    body_root = gcd_dir / args.body_type
    data_root = body_root / "data"
    required_suffixes = list(args.required_suffix or DEFAULT_REQUIRED_SUFFIXES)

    if not body_root.is_dir():
        raise FileNotFoundError(body_root)

    dataset_props_path = (
        Path(args.dataset_properties).expanduser().resolve()
        if args.dataset_properties
        else default_dataset_properties_path(gcd_dir, args.body_type)
    )
    if not dataset_props_path.exists():
        raise FileNotFoundError(dataset_props_path)

    filter_report_path = (
        Path(args.filter_report).expanduser().resolve()
        if args.filter_report
        else default_filter_report_path(gcd_dir, args.body_type)
    )
    if args.require_filter_report and not filter_report_path.exists():
        raise FileNotFoundError(filter_report_path)

    props = load_yaml(dataset_props_path)
    sample_cases, sample_cases_source = collect_sample_cases(props, body_root)

    if filter_report_path.exists():
        failure_source = "filter_fail_report.yaml"
        failures_by_case = collect_failures_from_filter_report(load_yaml(filter_report_path))
    else:
        failure_source = "dataset_properties"
        failures_by_case = collect_failures_from_dataset_props(props)

    failed_cases = set(failures_by_case)
    product_cases = list_direct_case_dirs(body_root)
    data_cases = list_direct_case_dirs(data_root)

    if not sample_cases and args.expected_count:
        sample_cases_source = "expected_count"

    if sample_cases:
        failed_in_sample = failed_cases & sample_cases
        expected_remaining_cases = sample_cases - failed_in_sample
        failed_not_in_sample = sorted(failed_cases - sample_cases)
        missing_product_cases = sorted(expected_remaining_cases - product_cases)
        unexpected_product_cases = sorted(product_cases - expected_remaining_cases)
        missing_data_cases = sorted(expected_remaining_cases - data_cases)
        unexpected_data_cases = sorted(data_cases - expected_remaining_cases)
        original_sample_count = len(sample_cases)
        failed_case_count = len(failed_in_sample)
    else:
        failed_not_in_sample = []
        missing_product_cases = []
        unexpected_product_cases = []
        missing_data_cases = []
        unexpected_data_cases = []
        original_sample_count = int(args.expected_count)
        failed_case_count = len(failed_cases)
        expected_remaining_cases = set(product_cases)

    if args.expected_count and sample_cases and len(sample_cases) != args.expected_count:
        expected_count_mismatch = {
            "expected_count": args.expected_count,
            "sample_cases": len(sample_cases),
        }
    else:
        expected_count_mismatch = None

    expected_remaining_count = original_sample_count - failed_case_count
    failed_product_cases_remaining = sorted(product_cases & failed_cases)
    incomplete_cases: dict[str, list[str]] = {}
    for case_name in sorted(product_cases):
        missing = check_case_files(body_root / case_name, case_name, required_suffixes)
        if missing:
            incomplete_cases[case_name] = missing

    issues: list[str] = []
    if expected_count_mismatch:
        issues.append("expected_count_mismatch")
    if len(product_cases) != expected_remaining_count:
        issues.append("product_case_count_mismatch")
    if len(data_cases) != expected_remaining_count:
        issues.append("data_case_count_mismatch")
    if failed_not_in_sample:
        issues.append("failed_cases_not_in_sample_cases")
    if missing_product_cases:
        issues.append("missing_product_cases")
    if unexpected_product_cases:
        issues.append("unexpected_product_cases")
    if missing_data_cases:
        issues.append("missing_data_cases")
    if unexpected_data_cases:
        issues.append("unexpected_data_cases")
    if failed_product_cases_remaining:
        issues.append("failed_product_cases_remaining")
    if incomplete_cases:
        issues.append("incomplete_product_cases")

    status = "pass" if not issues else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "gcd_dir": str(gcd_dir),
        "body_type": args.body_type,
        "body_root": str(body_root),
        "dataset_properties": str(dataset_props_path),
        "filter_report": str(filter_report_path) if filter_report_path.exists() else None,
        "sample_cases_source": sample_cases_source,
        "failure_source": failure_source,
        "required_suffixes": required_suffixes,
        "summary": {
            "original_sample_cases": original_sample_count,
            "failed_cases": failed_case_count,
            "expected_remaining_cases": expected_remaining_count,
            "product_case_dirs": len(product_cases),
            "data_case_dirs": len(data_cases),
            "incomplete_product_cases": len(incomplete_cases),
            "issues": len(issues),
        },
        "issues": issues,
        "expected_count_mismatch": expected_count_mismatch,
        "failed_cases_not_in_sample_cases": failed_not_in_sample,
        "missing_product_cases": missing_product_cases,
        "unexpected_product_cases": unexpected_product_cases,
        "missing_data_cases": missing_data_cases,
        "unexpected_data_cases": unexpected_data_cases,
        "failed_product_cases_remaining": failed_product_cases_remaining,
        "incomplete_product_cases": incomplete_cases,
        "failures_by_case": failures_by_case,
    }


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "integrity_report.yaml"
    report_path.write_text(
        yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except Exception as exc:
        print(f"check_integrity.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Integrity")
    print(f"  status: {report['status']}")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    if report["issues"]:
        print("  issue_names: " + ", ".join(report["issues"]))

    if not args.no_write:
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(report["body_root"])
        report_path = write_report(output_dir, report)
        print(f"  report: {report_path}")

    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
