"""Command line entry point for the semantic-edit pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate exactly one semantic edit per source, simulate it, and "
            "publish final pairs only after physical acceptance."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Plan balanced single-operation candidates for one source batch.",
    )
    plan_parser.add_argument("--input-root", type=Path, required=True)
    plan_parser.add_argument("--batch-id", required=True)
    plan_parser.add_argument("--analysis-root", type=Path, required=True)
    plan_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        required=True,
    )
    plan_parser.add_argument("--prompt-builder", type=Path, required=True)
    plan_parser.add_argument("--seed", default="42")

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate/resume candidates and prepare the active simulation workspace.",
    )
    generate_parser.add_argument("--input-root", type=Path, required=True)
    generate_parser.add_argument("--analysis-root", type=Path, required=True)
    generate_parser.add_argument("--output-root", type=Path, required=True)
    generate_parser.add_argument("--seed", default="42")
    generate_parser.add_argument("--split-path", type=Path)
    generate_parser.add_argument("--split-name")

    project_parser = subparsers.add_parser(
        "project",
        help="Finalize the active workspace from GarmentCode's filter report.",
    )
    project_parser.add_argument("--output-root", type=Path, required=True)
    project_parser.add_argument("--prompt-builder", type=Path, required=True)

    status_parser = subparsers.add_parser(
        "status",
        help="Summarize accepted, pending, and unresolved sources.",
    )
    status_parser.add_argument("--output-root", type=Path, required=True)
    status_parser.add_argument("--split-path", type=Path)
    status_parser.add_argument("--split-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        from .prompt import PromptError
        from .planning import SourcePlanError, plan

        execute = lambda: plan(
            input_root=args.input_root,
            batch_id=args.batch_id,
            analysis_root=args.analysis_root,
            config_path=args.config_path,
            prompt_builder=args.prompt_builder,
            seed=args.seed,
        )
        command_errors = (
            PromptError,
            SourcePlanError,
        )
    elif args.command == "generate":
        from editing.simulation_workspace import SimulationWorkspaceError

        from .generation import generate
        from .store import SemanticEditError, SimpleEditStoreError

        execute = lambda: generate(
            input_root=args.input_root,
            analysis_root=args.analysis_root,
            output_root=args.output_root,
            seed=args.seed,
            split_path=args.split_path,
            split_name=args.split_name,
        )
        command_errors = (
            SemanticEditError,
            SimpleEditStoreError,
            SimulationWorkspaceError,
        )
    elif args.command == "project":
        from editing.simulation_workspace import SimulationWorkspaceError

        from .projection import project
        from .prompt import PromptError
        from .store import SemanticEditError, SimpleEditStoreError

        execute = lambda: project(
            output_root=args.output_root,
            prompt_builder=args.prompt_builder,
        )
        command_errors = (
            PromptError,
            SemanticEditError,
            SimpleEditStoreError,
            SimulationWorkspaceError,
        )
    else:
        from editing.simulation_workspace import SimulationWorkspaceError

        from .store import SemanticEditError, SimpleEditStoreError, status

        execute = lambda: status(
            args.output_root,
            split_path=args.split_path,
            split_name=args.split_name,
        )
        command_errors = (
            SemanticEditError,
            SimpleEditStoreError,
            SimulationWorkspaceError,
        )
    caught_errors = (OSError, ValueError) + command_errors
    try:
        result = execute()
    except caught_errors as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
