"""Match each GCD garment panel with ten score/coverage-balanced images."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


MATCH_COUNT = 10
RANK_COVERAGE_TARGET = 0.70
SCORE_BATCH_SIZE = 16384
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        "--gcd_root",
        dest="gcd_root",
        required=True,
        help="Root containing GCD case prompt and feature files.",
    )
    parser.add_argument("--openvton_root", required=True)
    parser.add_argument(
        "--batch_list",
        type=parse_batch_list,
        default=None,
        help="Optional batch selection, for example 0-12 or 0,2,5-7.",
    )
    parser.add_argument("--log", required=True)
    return parser.parse_args()


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


def load_feature(path: Path) -> np.ndarray:
    feature = np.asarray(np.load(path), dtype="float32").reshape(-1)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature /= norm
    return feature


def load_image_bank(root: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    feature_paths = sorted(root.rglob("source_rgba_feat.npy"))

    image_paths: list[Path] = []
    image_types: list[str] = []
    features: list[np.ndarray] = []
    for feature_path in tqdm(
        feature_paths,
        desc="Loading OpenVTON features",
        unit="image",
    ):
        image_path = feature_path.parent / "source_rgba.png"
        image_paths.append(image_path.resolve())
        image_types.append(feature_path.parent.parent.name)
        features.append(load_feature(feature_path))

    return image_paths, np.asarray(image_types, dtype=object), np.stack(features)


def batch_id_for_prompt(prompt_path: Path, root: Path) -> int | None:
    if root.name.startswith("batch_") and root.name[6:].isdigit():
        return int(root.name[6:])
    for part in prompt_path.relative_to(root).parts:
        if part.startswith("batch_") and part[6:].isdigit():
            return int(part[6:])
    return None


def load_queries(
    root: Path, batch_ids: set[int] | None = None
) -> tuple[list[dict], list[dict]]:
    prompt_paths = sorted(root.rglob("*_prompts.json"))

    cases: list[dict] = []
    records: list[dict] = []
    for prompt_path in tqdm(prompt_paths, desc="Loading GCD queries", unit="case"):
        prompt_batch_id = batch_id_for_prompt(prompt_path, root)
        if batch_ids is not None and prompt_batch_id not in batch_ids:
            continue
        case_id = prompt_path.parent.name
        data = json.loads(prompt_path.read_text(encoding="utf-8"))
        parts = data["parts"]
        panel_names = ["upper", "bottom"] if "upper" in parts else ["whole"]
        case = {
            "case_id": case_id,
            "case_dir": prompt_path.parent,
            "record_indices": [],
        }
        for panel_name in panel_names:
            feature_path = prompt_path.parent / f"{case_id}_{panel_name}_prompt_feat.npy"
            part_type = parts[panel_name]["part_type"]
            record_index = len(records)
            records.append(
                {
                    "feature": load_feature(feature_path),
                    "allowed_key": (part_type,),
                    "panel_name": panel_name,
                }
            )
            case["record_indices"].append(record_index)
        cases.append(case)
    return cases, records


def rank_capacity(
    row_count: int,
    column_count: int,
) -> tuple[int, int]:
    """Choose a per-rank image capacity that guarantees the coverage target."""
    desired_coverage = min(
        row_count,
        column_count,
        max(1, int(np.ceil(column_count * RANK_COVERAGE_TARGET))),
    )
    minimum_capacity = (row_count + column_count - 1) // column_count
    if desired_coverage == 1:
        return max(1, minimum_capacity), desired_coverage

    maximum_capacity = (row_count - 1) // (desired_coverage - 1)
    return max(minimum_capacity, maximum_capacity), desired_coverage


def pick_type_matches(
    queries: np.ndarray,
    candidate_indices: np.ndarray,
    image_features: np.ndarray,
    image_paths: list[Path],
    garment_type: str,
    random_state: np.random.Generator,
    progress: tqdm,
) -> tuple[list[list[str]], dict[str, object]]:
    """Pick every rank under its own coverage capacity and CLIP scores."""
    row_count = len(queries)
    column_count = len(candidate_indices)
    if column_count < MATCH_COUNT:
        raise ValueError(
            f"{garment_type} has only {column_count} OpenVTON images; "
            f"at least {MATCH_COUNT} are required"
        )

    capacity, desired_coverage = rank_capacity(row_count, column_count)
    candidate_features = image_features[candidate_indices]
    rank_usage = np.zeros((MATCH_COUNT, column_count), dtype=np.int64)
    row_order = random_state.permutation(row_count)
    matches: list[list[str] | None] = [None] * row_count
    rank_score_sums = np.zeros(MATCH_COUNT, dtype=np.float64)

    for start in range(0, row_count, SCORE_BATCH_SIZE):
        rows = row_order[start : start + SCORE_BATCH_SIZE]
        scores = queries[rows] @ candidate_features.T

        for batch_row, row_index in enumerate(rows):
            selected_columns: list[int] = []
            for rank in range(MATCH_COUNT):
                available_mask = rank_usage[rank] < capacity
                if selected_columns:
                    available_mask[selected_columns] = False
                available = np.flatnonzero(available_mask)

                if not len(available):
                    covered = int(np.count_nonzero(rank_usage[rank]))
                    if covered < desired_coverage:
                        raise RuntimeError(
                            f"Cannot satisfy rank {rank + 1} coverage for {garment_type}"
                        )
                    available_mask = np.ones(column_count, dtype=bool)
                    available_mask[selected_columns] = False
                    available = np.flatnonzero(available_mask)

                available_scores = scores[batch_row, available]
                column = int(available[int(np.argmax(available_scores))])
                rank_usage[rank, column] += 1
                rank_score_sums[rank] += float(scores[batch_row, column])
                selected_columns.append(column)

            matches[int(row_index)] = [
                str(image_paths[int(candidate_indices[column])])
                for column in selected_columns
            ]

        rank_coverage = np.count_nonzero(rank_usage, axis=1)
        progress.set_postfix_str(
            f"type={garment_type}, rank-cover={rank_coverage.min()}/{column_count}",
            refresh=False,
        )
        progress.update(len(rows))

    if any(match is None for match in matches):
        raise RuntimeError(f"Failed to match one or more {garment_type} queries")

    rank_coverage = np.count_nonzero(rank_usage, axis=1)
    if np.any(rank_coverage < desired_coverage):
        raise RuntimeError(
            f"One or more ranks did not reach the coverage target for {garment_type}"
        )

    return [match for match in matches if match is not None], {
        "images": column_count,
        "queries": row_count,
        "rank_capacity": capacity,
        "rank_desired_coverage": desired_coverage,
        "rank_coverage": rank_coverage.tolist(),
        "rank_score_means": (rank_score_sums / row_count).tolist(),
    }


def build_matches(
    records: list[dict],
    image_types: np.ndarray,
    image_features: np.ndarray,
    image_paths: list[Path],
    logger: logging.Logger,
) -> list[list[str]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record_index, record in enumerate(records):
        grouped[record["allowed_key"][0]].append(record_index)

    matches: list[list[str] | None] = [None] * len(records)
    random_state = np.random.default_rng(RANDOM_SEED)
    with tqdm(total=len(records), desc="Matching GCD garments", unit="query") as progress:
        for garment_type in sorted(grouped):
            record_indices = grouped[garment_type]
            candidate_indices = np.flatnonzero(image_types == garment_type)
            queries = np.stack([records[index]["feature"] for index in record_indices])
            type_matches, stats = pick_type_matches(
                queries,
                candidate_indices,
                image_features,
                image_paths,
                garment_type,
                random_state,
                progress,
            )
            for record_index, selected in zip(record_indices, type_matches):
                matches[record_index] = selected

            for rank, (covered, score_mean) in enumerate(
                zip(stats["rank_coverage"], stats["rank_score_means"]),
                start=1,
            ):
                logger.info(
                    "%s rank=%d: queries=%d, images=%d, covered=%d (%.2f%%), "
                    "capacity=%d, score=%.6f",
                    garment_type,
                    rank,
                    stats["queries"],
                    stats["images"],
                    covered,
                    100.0 * covered / stats["images"],
                    stats["rank_capacity"],
                    score_mean,
                )

    if any(match is None for match in matches):
        raise RuntimeError("Failed to match one or more prompt features")
    return [match for match in matches if match is not None]


def main() -> None:
    args = parse_args()
    gcd_root = Path(args.gcd_root).resolve()
    openvton_root = Path(args.openvton_root).resolve()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)

    image_paths, image_types, image_features = load_image_bank(openvton_root)
    cases, records = load_queries(gcd_root, args.batch_list)
    matches = build_matches(
        records,
        image_types,
        image_features,
        image_paths,
        logger,
    )

    for case in tqdm(cases, desc="Writing match files", unit="case"):
        result = {
            records[index]["panel_name"]: matches[index]
            for index in case["record_indices"]
        }
        output_path = case["case_dir"] / f"{case['case_id']}_matched_openvton.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        panel_counts = ", ".join(
            f"{name}={len(result[name])}" for name in result
        )
        logger.info("%s: %s -> %s", case["case_id"], panel_counts, output_path)

    logger.info(
        "matched cases: %d, panels: %d, images: %d",
        len(cases),
        len(records),
        len(image_paths),
    )


if __name__ == "__main__":
    main()
