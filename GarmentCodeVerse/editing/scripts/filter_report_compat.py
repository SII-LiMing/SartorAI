#!/usr/bin/env python3
"""Convert the current compact filter report for semantic-edit projection."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import yaml


def mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: filter_report_compat.py WORKSPACE")
    workspace = Path(sys.argv[1]).expanduser().resolve()
    body_root = workspace / "random_body"
    manifest_path = workspace / "simulation_manifest.json"
    compact_path = body_root / "filter_fail_report.yaml"
    manifest = mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        "simulation manifest",
    )
    compact = mapping(
        yaml.safe_load(compact_path.read_text(encoding="utf-8")),
        "compact filter report",
    )
    candidate_ids = manifest.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError("simulation manifest has no candidate_ids")
    if len(candidate_ids) != len(set(candidate_ids)) or not all(
        isinstance(item, str) and item for item in candidate_ids
    ):
        raise ValueError("simulation manifest candidate_ids are invalid")

    candidates = set(candidate_ids)
    failure_reasons = mapping(
        compact.get("log_failures_by_case", {}),
        "filter failure reasons",
    )
    failed = set(failure_reasons)
    if failed - candidates:
        raise ValueError(f"filter report contains unknown candidates: {sorted(failed - candidates)}")

    full_report = {
        "report_type": "filter_fail_report_full",
        "simulation_id": manifest["simulation_id"],
        "passed_cases": sorted(candidates - failed),
        "final_failed_cases": sorted(failed),
        "final_fail_reasons": {
            candidate_id: failure_reasons[candidate_id]
            for candidate_id in sorted(failed)
        },
    }
    output_path = body_root / "filter_fail_report_full.yaml"
    output_path.write_text(
        yaml.safe_dump(full_report, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"full filter report: passed={len(full_report['passed_cases'])} "
        f"failed={len(full_report['final_failed_cases'])} "
        f"path={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
