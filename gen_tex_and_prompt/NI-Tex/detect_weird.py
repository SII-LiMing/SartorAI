"""Detect cases with two large, different color blocks in generated textures."""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

def parse_obj_uv(obj_path):
    mesh = trimesh.load(obj_path, force="mesh", process=False, maintain_order=True)
    return np.asarray(mesh.visual.uv, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def rasterize_uv(vts, faces, size):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    x = np.clip(vts[:, 0], 0, 1) * (size - 1)
    y = (1.0 - np.clip(vts[:, 1], 0, 1)) * (size - 1)
    for a, b, c in faces:
        draw.polygon([(x[a], y[a]), (x[b], y[b]), (x[c], y[c])], fill=255)
    return np.asarray(img) > 0


def merged_components(mask, min_area):
    labels, n = ndimage.label(mask)
    blocks = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        comp = labels[sl] == i
        area = int(comp.sum())
        if area < min_area:
            continue
        full = np.zeros(mask.shape, dtype=bool)
        full[sl][comp] = True
        blocks.append({"area": area, "mask": full})
    return blocks


def quantize_rgb(rgb, bin_size):
    q = np.round(rgb.astype(np.float64) / bin_size).astype(np.int32)
    return q


def abnormal_blocks(rgb, used, bin_size, min_area, color_dist_thresh):
    q = quantize_rgb(rgb, bin_size)
    used_q = q[used]
    bins, counts = np.unique(used_q, axis=0, return_counts=True)
    normal_bin = bins[np.argmax(counts)]
    dist = np.linalg.norm((q - normal_bin).astype(np.float64) * bin_size, axis=2)
    abnormal = used & (dist >= color_dist_thresh)
    abnormal = ndimage.binary_opening(abnormal, iterations=1)

    blocks = merged_components(abnormal, min_area)
    blocks.sort(key=lambda b: b["area"], reverse=True)

    normal_frac = float((used & (dist < color_dist_thresh)).sum()) / int(used.sum())
    return blocks, normal_bin.astype(np.float64) * bin_size, normal_frac


def analyze_part(part_dir, args):
    part_dir = Path(part_dir)
    obj_path = part_dir / "results.obj"
    tex_path = part_dir / "results.jpg"
    if not obj_path.exists() or not tex_path.exists():
        return None

    vts, faces = parse_obj_uv(obj_path)
    used = rasterize_uv(vts, faces, args.size)
    used = ndimage.binary_erosion(used, iterations=2)

    img = Image.open(tex_path)
    img.draft("RGB", (args.size, args.size))
    img = img.convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    rgb = np.asarray(img)

    used_area = int(used.sum())
    min_area = max(1, int(args.min_block_frac * used_area))
    blocks, normal_color, normal_frac = abnormal_blocks(
        rgb, used, args.bin_size, min_area, args.color_dist_thresh)

    fail_mask = np.zeros(used.shape, dtype=bool)
    color_dist = 0.0
    abnormal_frac = 0.0
    score = 0.0
    if blocks:
        abnormal_area = sum(block["area"] for block in blocks)
        abnormal_frac = abnormal_area / used_area
    if blocks and abnormal_frac <= args.max_abnormal_frac:
        for block in blocks:
            fail_mask |= block["mask"]
        score = blocks[0]["area"] / used_area
        abnormal_color = rgb[blocks[0]["mask"]].mean(axis=0)
        color_dist = float(np.linalg.norm(abnormal_color - normal_color))

    row = {
        "flag": int(score > 0),
        "score": round(float(score), 5),
        "normal_frac": round(float(normal_frac), 5),
        "abnormal_frac": round(float(abnormal_frac), 5),
        "color_dist": round(float(color_dist), 2),
        "n_blocks": len(blocks),
        "used_frac": round(float(used.mean()), 5),
    }
    return row, fail_mask, used, rgb


def save_montage(rgb, mask, used, out_path, label):
    shown = rgb.copy()
    shown[~used] = 0
    edge = mask ^ ndimage.binary_erosion(mask, iterations=2)
    shown[edge] = [255, 0, 0]

    size = rgb.shape[0]
    canvas = Image.new("RGB", (size, size + 26), (255, 255, 255))
    canvas.paste(Image.fromarray(shown), (0, 26))
    ImageDraw.Draw(canvas).text((4, 8), label, fill=(200, 0, 0))
    canvas.save(out_path)


def find_cases(root):
    root = Path(root)
    cases = sorted(root.glob("rand_*"))
    if cases:
        return cases
    if (root / "random_body").is_dir():
        return sorted((root / "random_body").glob("rand_*"))

    cases = []
    for batch in sorted(root.glob("batch_*")):
        random_body = batch / "random_body"
        if random_body.is_dir():
            cases.extend(sorted(random_body.glob("rand_*")))
    return cases


def iter_parts(root, parts, texture_subdir):
    for case in find_cases(root):
        out_dir = case / texture_subdir
        if not out_dir.is_dir():
            continue
        for part in parts:
            part_dir = out_dir / part
            if part_dir.is_dir():
                yield case, part, part_dir


def main():
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--texture_subdir", default="NI-Tex_output")
    parser.add_argument("--vis_flag", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--bin_size", type=float, default=24.0)
    parser.add_argument("--min_block_frac", type=float, default=0.03)
    parser.add_argument("--color_dist_thresh", type=float, default=60.0)
    parser.add_argument("--max_abnormal_frac", type=float, default=0.55)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.vis_flag:
        flagged_dir = out_dir / "flagged"
        flagged_dir.mkdir(parents=True, exist_ok=True)

    parts = ["upper","bottom", "whole"]
    tasks = list(iter_parts(args.root, parts, args.texture_subdir))
    # tasks = tasks[:100]

    total = 0
    failed = []
    for case, part, part_dir in tqdm(tasks):
        out = analyze_part(part_dir, args)
        if out is None:
            continue

        total += 1
        row, mask, used, rgb = out
        if row["flag"]:
            failed.append((row["score"], case.name, part, str(case)))
            if args.vis_flag:
                label = f'{case.name}/{part} score={row["score"]:.3f}'
                name = f'{row["score"]:.3f}_{case.name}_{part}.png'
                save_montage(rgb, mask, used, flagged_dir / name, label)

    failed.sort(reverse=True)
    fail_txt = out_dir / "failed_cases.txt"
    with open(fail_txt, "w") as f:
        for score, case_name, part, case_dir in failed:
            f.write(f"{score:.5f}\t{case_name}\t{part}\t{case_dir}\n")

    print(f"analyzed: {total}")
    print(f"fail: {len(failed)}")

if __name__ == "__main__":
    main()
