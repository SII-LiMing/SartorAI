"""Encode every recursively discovered OpenVTON ``source_rgba.png`` image."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_rgb(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_image_features(model, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        feature = model.get_image_features(**inputs)
    if torch.is_tensor(feature):
        return feature
    if hasattr(feature, "pooler_output"):
        return feature.pooler_output
    if isinstance(feature, (tuple, list)) and len(feature) > 1:
        return feature[1]
    raise TypeError(f"Unsupported get_image_features result: {type(feature)!r}")


def encode_images(model, processor, paths: list[Path], device: torch.device) -> np.ndarray:
    features = []
    with torch.no_grad():
        for batch in tqdm(list(chunks(paths, 512)), desc="OpenVTON images"):
            images = [load_rgb(path) for path in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features.append(
                F.normalize(get_image_features(model, inputs), dim=-1).cpu().numpy()
            )
    return np.concatenate(features).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--model_dir", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    root = Path(args.root)

    logging.basicConfig(
        filename=args.log, filemode="w", level=logging.INFO, format="%(message)s"
    )
    logger = logging.getLogger(__name__)
    paths = sorted(root.rglob("source_rgba.png"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.model_dir)
    model = CLIPModel.from_pretrained(args.model_dir).to(device).eval()
    features = encode_images(model, processor, paths, device)
    for path, feature in zip(paths, features):
        out = path.parent / "source_rgba_feat.npy"
        np.save(out, feature)
        logger.info("%s -> %s", path, out)
    logger.info("wrote image features: %d", len(paths))


if __name__ == "__main__":
    main()
