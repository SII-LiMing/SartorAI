"""Encode prompts produced by ``build_prompts.py`` into per-case CLIP features."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", required=True, help="Directory containing case prompt JSON files."
    )
    parser.add_argument("--log", required=True)
    parser.add_argument("--model_dir", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_text_features(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        return model.get_text_features(**inputs)


def encode_texts(
    model,
    processor,
    texts: list[str],
    device: torch.device,
) -> np.ndarray:
    features = []
    with torch.no_grad():
        for batch in list(chunks(texts, 256)):
            inputs = processor(
                text=batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features.append(
                F.normalize(get_text_features(model, inputs), dim=-1).cpu().numpy()
            )
    return np.concatenate(features).astype("float32")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    logging.basicConfig(
        filename=args.log, filemode="w", level=logging.INFO, format="%(message)s"
    )
    logger = logging.getLogger(__name__)
    
    prompt_files = sorted(root.rglob("*_prompts.json"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.model_dir)
    model = CLIPModel.from_pretrained(args.model_dir).to(device).eval()

    records: list[tuple[Path, str]] = []
    for path in prompt_files:
        case_id = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8"))
        parts = data.get("parts", {})
        part_names = ["upper", "bottom"] if "upper" in parts else ["whole"]
        for part_name in part_names:
            prompt = parts[part_name]["part_prompt"]
            records.append(
                (path.parent / f"{case_id}_{part_name}_prompt_feat.npy", str(prompt))
            )

    features = encode_texts(model, processor, [text for _, text in records], device)
    for (out, _), feature in zip(records, features):
        np.save(out, feature)
    logger.info("wrote prompt features: %d", len(records))


if __name__ == "__main__":
    main()
