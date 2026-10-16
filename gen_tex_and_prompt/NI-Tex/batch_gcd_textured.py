import os, sys, time, statistics, argparse, traceback, json, shutil, warnings, gc
import logging
from pathlib import Path
import numpy as np
import torch, trimesh
from PIL import Image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from diffusers.utils import logging as diffusers_logging

warnings.filterwarnings("ignore")
diffusers_logging.set_verbosity(50)
torch.backends.cudnn.benchmark = True

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DifferentiableRenderer.MeshRender import MeshRender
from hunyuanpaintpbr.unet.model import HunyuanPaint
from utils.image_super_utils import imageSuperNet
from utils.pipeline_utils import ViewProcessor
from utils.uvwrap_utils import mesh_uv_wrap

try:
    from utils.torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    pass

VALID_PARTS = {"upper", "bottom", "whole"}
REQUIRED_RESULT_FILES = (
    "results.obj",
    "results.mtl",
    "results.jpg",
    "results_metallic.jpg",
    "results_roughness.jpg",
)


def job_is_complete(output_dir):
    output_dir = Path(output_dir)
    return all((output_dir / filename).is_file() for filename in REQUIRED_RESULT_FILES)


def load_image(pil_img, color, image_size=512):
    if isinstance(pil_img, (str, Path)):
        pil_img = Image.open(str(pil_img))

    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")

    pil_img = pil_img.resize((image_size, image_size))
    image = np.asarray(pil_img, dtype=np.float32) / 255.0

    if image.shape[2] == 3:
        image = image[:, :, :3]
        alpha = np.ones_like(image)
    else:
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
    alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
    return image, alpha


def multiview_images_to_tensor(images, expected_views=10, view_size=512):
    tensors = []
    for img in images[:expected_views]:
        img = img.convert("RGB")
        img_t = TF.to_tensor(img)
        img_t = TF.resize(img_t, [view_size, view_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        tensors.append(img_t)

    if not tensors:
        raise ValueError("No valid multiview images")

    return torch.stack(tensors).unsqueeze(0)


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view: int = 10, resolution: int = 512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resolution = resolution
        self.max_selected_view_num = max_num_view
        self.orth_scale = 1.35
        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.bake_exp = 4
        self.merge_method = "fast"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"
        self.candidate_camera_azims = [0, 180, 90, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1.0, 0.5, 0.1, 0.1, 0.05, 0.05]

        for azim in [45, 135]:
            self.candidate_camera_azims.extend([azim, -azim])
            self.candidate_camera_elevs.extend([0, 0])
            self.candidate_view_weights.extend([0.01, 0.01])


def get_view_params(conf):
    return {
        "elevs": conf.candidate_camera_elevs[:conf.max_selected_view_num],
        "azims": conf.candidate_camera_azims[:conf.max_selected_view_num],
        "weights": conf.candidate_view_weights[:conf.max_selected_view_num]
    }

def run_stage1_georender(mesh, conf: Hunyuan3DPaintConfig, view_params):
    print("\n--- [STAGE 1] Geometry Rendering ---")

    elevs, azims, weights = view_params["elevs"], view_params["azims"], view_params["weights"]
    render = MeshRender(
        default_resolution=conf.render_size,
        texture_size=conf.texture_size,
        bake_mode=conf.bake_mode,
        raster_mode=conf.raster_mode,
        orth_scale=conf.orth_scale
    )
    view_processor = ViewProcessor(conf, render)

    render.load_mesh(mesh=mesh)

    normal_maps = view_processor.render_normal_multiview(elevs, azims, use_abs_coor=False)
    position_maps = view_processor.render_position_multiview(elevs, azims)

    print("Geometry maps rendered in memory")

    del render, view_processor, mesh
    return normal_maps, position_maps, elevs, azims, weights

def run_stage2_diffusion(image_prompt_path, normal_maps, position_maps, output_dir, device, model):
    output_dir = Path(output_dir)
    model.logdir = str(output_dir)
    os.makedirs(os.path.join(model.logdir, "textures"), exist_ok=True)
    
    resolution = model.view_size
    N_views = model.num_view
    
    print("Loading image and geometry tensors into VRAM...")
    
    bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ref_tensor, _ = load_image(image_prompt_path, color=bg_color, image_size=resolution)
    ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(0).to(device)

    normal_tensor = multiview_images_to_tensor(normal_maps, N_views, resolution).to(device)
    if normal_tensor.dim() == 4:
        normal_tensor = normal_tensor.unsqueeze(0)
        
    position_tensor = multiview_images_to_tensor(position_maps, N_views, resolution).to(device)
    if position_tensor.dim() == 4:
        position_tensor = position_tensor.unsqueeze(0)

    dummy_target = torch.zeros((1, N_views, 3, resolution, resolution), device=device)

    # Construct batch mapping
    batch = {
        "images_cond": ref_tensor,
        "images_normal": normal_tensor,
        "images_position": position_tensor,
        "images_ref_paths": [[str(image_prompt_path)]], 
        "name": ["demo_output"] 
    }
    for pbr_token in model.pbr_settings:
        batch[f"images_{pbr_token}"] = dummy_target

    print("Running Diffusion Pipeline (Denosing via validation_step)...")
    
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.validation_step(batch, batch_idx=0)
    finally:
        validation_outputs = getattr(model, "validation_step_outputs", None)
        if validation_outputs is not None:
            validation_outputs.clear()
        del normal_tensor, position_tensor, ref_tensor, dummy_target, batch
    # Organize generated assets
    print("\nReorganizing generated textures to the target format...")

    base_name = "demo_output"
    suffix = os.path.splitext(os.path.basename(image_prompt_path))[0].split('_')[0]
    temp_output_path = output_dir / "images_val" / "images" / f"{base_name}_{suffix}"
    
    pbr_types = ["albedo", "mr"]
    for group_idx in range(N_views):
        for view_idx, pbr_name in enumerate(pbr_types):
            src_file = temp_output_path / f"{group_idx+1}_{view_idx+1}.png"
            dst_folder = output_dir / "textures" / pbr_name
            dst_folder.mkdir(parents=True, exist_ok=True)
            
            dst_file = dst_folder / f"view_{group_idx:03d}.png"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))
                
    # Cleanup validation temp directory
    val_dir_to_remove = output_dir / "images_val"
    if val_dir_to_remove.exists():
        shutil.rmtree(val_dir_to_remove)
        
    print(f"Extraction complete! Final assets are properly sorted in: {output_dir}")
    return output_dir / "textures"


def load_diffusion_model(config_path, device):
    """Load diffusion model for stage2"""
    print("\n--- Load diffusion model ---")
    
    config = OmegaConf.load(config_path)
    ckpt_path = config.get("resume_from") 
    
    print("Extracting model parameters from config...")
    init_kwargs = OmegaConf.to_container(config.model.params, resolve=True)

    print("Instantiating base model...")
    model = HunyuanPaint(**init_kwargs)

    # Patch UNet conv_in to accept the required number of channels
    nc = init_kwargs.get("noise_in_channels")
    if nc is not None:
        print(f"Modifying UNet conv_in to accept {nc} channels...")
        model_unet = model.unet.unet if hasattr(model.unet, "unet") else model.unet
        
        with torch.no_grad():
            old_conv = model_unet.conv_in
            new_conv = torch.nn.Conv2d(
                nc, old_conv.out_channels, old_conv.kernel_size, 
                old_conv.stride, old_conv.padding
            )
            new_conv.weight.zero_()
            new_conv.bias.zero_()
            model_unet.conv_in = new_conv

    print(f"Loading weights from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Load state dict with strict=False to tolerate missing DINO features
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device, dtype=torch.bfloat16)
    
    if hasattr(model, 'pipeline'):
        model.pipeline.to(device, dtype=torch.bfloat16)
    
    return model

def run_stage3_baking(mesh_path: Path, texture_dir: Path, output_dir: Path, elevs, azims, weights, conf: Hunyuan3DPaintConfig, super_model, mesh=None):
    print(f"\n--- [STAGE 3] Texture Baking & Inpainting ---")

    print("Loading generated PBR maps from disk...")
    albedo_imgs = []
    mr_imgs = []

    for i in range(conf.max_selected_view_num):
        albedo_path = texture_dir / "albedo" / f"view_{i:03d}.png"
        mr_path = texture_dir / "mr" / f"view_{i:03d}.png"

        albedo_imgs.append(Image.open(albedo_path).convert("RGBA"))
        if mr_path.exists():
            mr_imgs.append(Image.open(mr_path).convert("RGBA"))

    print("Enhancing Albedo maps...")
    for i in range(len(albedo_imgs)):
        albedo_imgs[i] = super_model(albedo_imgs[i])

    if mr_imgs:
        print("Enhancing Metallic-Roughness (MR) maps...")
        for i in range(len(mr_imgs)):
            mr_imgs[i] = super_model(mr_imgs[i])


    for i in range(len(albedo_imgs)):
        albedo_imgs[i] = albedo_imgs[i].resize((conf.render_size, conf.render_size))
    for i in range(len(mr_imgs)):
        mr_imgs[i] = mr_imgs[i].resize((conf.render_size, conf.render_size))

    print("Initializing Mesh Renderer & UV Wrapper...")
    render = MeshRender(
        default_resolution=conf.render_size,
        texture_size=conf.texture_size,
        bake_mode=conf.bake_mode,
        raster_mode=conf.raster_mode,
        orth_scale=conf.orth_scale
    )
    view_processor = ViewProcessor(conf, render)

    if mesh is None:
        mesh = trimesh.load(str(mesh_path), process=False)
        mesh = mesh_uv_wrap(mesh)
    render.load_mesh(mesh=mesh)

    print("Baking Albedo texture to UV map...")
    texture_albedo, mask_albedo = view_processor.bake_from_multiview(albedo_imgs, elevs, azims, weights)
    mask_albedo_np = (mask_albedo.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

    texture_albedo = view_processor.texture_inpaint(texture_albedo, mask_albedo_np)
    render.set_texture(texture_albedo, force_set=True)

    if mr_imgs:
        print("Baking Metallic-Roughness texture to UV map...")
        texture_mr, mask_mr = view_processor.bake_from_multiview(mr_imgs, elevs, azims, weights)
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_mr = view_processor.texture_inpaint(texture_mr, mask_mr_np)
        render.set_texture_mr(texture_mr)

    print("Exporting final 3D models...")
    obj_path = output_dir / "results.obj"

    render.save_mesh(str(obj_path), downsample=True)

    print(f"\n✅ Pipeline Complete! Fully textured 3D asset ready at: {obj_path}")

    del render, view_processor, mesh

def fmt(seconds):
    m, s = divmod(int(seconds), 60); h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def create_timing_logger(root, out_subdir):
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_name = Path(root).resolve().stem
    safe_out_subdir = str(out_subdir).strip().replace(os.sep, "_") or "output"
    log_path = log_dir / f"{batch_name}_{safe_out_subdir}.log"

    logger = logging.getLogger("batch_gcd_textured.timing")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger, log_path


def _failed_cases_file(weird_log_dir, root):
    """Locate the failed_cases.txt belonging to one batch.

    The detector currently writes either <log_root>/<batch_id>/failed_cases.txt
    or <log_root>/batch_<batch_id>/failed_cases.txt.  The direct-file fallback
    also makes a single-batch detector output directory usable.
    """
    if not weird_log_dir:
        return None

    log_root = Path(weird_log_dir)
    if not log_root.is_dir():
        return None

    batch_name = Path(root).name
    batch_id = batch_name[6:] if batch_name.startswith("batch_") else batch_name
    candidates = (
        log_root / batch_id / "failed_cases.txt",
        log_root / batch_name / "failed_cases.txt",
        log_root / "failed_cases.txt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_failed_pairs(failed_path):
    """Read (case_name, part) pairs from detector output."""
    failed_pairs = set()
    try:
        lines = Path(failed_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print("[weird-filter] cannot read " + str(failed_path) + ": " + str(exc), flush=True)
        return failed_pairs

    for line in lines:
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        case_name = fields[1].strip()
        part = fields[2].strip().lower()
        if case_name and part in VALID_PARTS:
            failed_pairs.add((case_name, part))
    return failed_pairs


def discover_jobs(root, out_subdir, rank, weird_log_dir=""):
    root = Path(root)
    body_root = root if root.name == "random_body" else root / "random_body"
    jobs = []

    failed_pairs = None
    weird_log_dir = str(weird_log_dir or "").strip()
    if weird_log_dir:
        log_root = Path(weird_log_dir)
        if not log_root.is_dir():
            print("[weird-filter] directory not found; using all matched cases: " + str(log_root), flush=True)
        else:
            failed_path = _failed_cases_file(log_root, root)
            if failed_path is None:
                failed_pairs = set()
                print("[weird-filter] no failed_cases.txt for " + root.name + "; skipping batch", flush=True)
            else:
                failed_pairs = _load_failed_pairs(failed_path)
                print(
                    "[weird-filter] " + str(failed_path) + ": "
                    + str(len(failed_pairs)) + " failed case/part pair(s)",
                    flush=True,
                )

    for match_path in sorted(body_root.rglob("*_matched_openvton.json")):
        rid = match_path.parent.name
        try:
            matches = json.loads(match_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print("[skip-invalid] " + str(match_path) + ": " + str(exc), flush=True)
            continue

        if not isinstance(matches, dict):
            print("[skip-invalid] " + str(match_path) + ": expected a JSON object", flush=True)
            continue

        for part, image_paths in matches.items():
            part = str(part).strip().lower()
            if part not in VALID_PARTS:
                continue
            if failed_pairs is not None and (rid, part) not in failed_pairs:
                continue
            if not isinstance(image_paths, list) or rank >= len(image_paths):
                print(
                    "[skip-missing] " + rid + "/" + part + ": rank " + str(rank),
                    flush=True,
                )
                continue

            if part == "whole":
                mesh = match_path.parent / (rid + "_sim.obj")
            else:
                mesh = match_path.parent / (rid + "_" + part + ".obj")
            img_path = Path(str(image_paths[rank]))
            out = match_path.parent / out_subdir / part
            if mesh.exists() and img_path.exists():
                jobs.append((rid + "/" + part, mesh, img_path, out))
            else:
                miss = []
                if not mesh.exists(): miss.append("mesh:" + mesh.name)
                if not img_path.exists(): miss.append("img:" + str(img_path))
                print("[skip-missing] " + rid + "/" + part + ": " + ", ".join(miss), flush=True)
    jobs.sort(key=lambda job: (job[0], str(job[1]), str(job[2]), str(job[3])))
    return jobs


def process_one(mesh_path, image_path, output_dir, conf, device, diffusion_model, super_model, view_params):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    texture_dir = output_dir / "textures"
    wrapped = normal_maps = position_maps = None
    try:
        wrapped = trimesh.load(str(mesh_path), process=False)
        wrapped = mesh_uv_wrap(wrapped)
        normal_maps, position_maps, elevs, azims, weights = run_stage1_georender(
            mesh=wrapped, conf=conf, view_params=view_params)
        texture_dir = run_stage2_diffusion(
            image_prompt_path=Path(image_path),
            normal_maps=normal_maps, position_maps=position_maps, output_dir=output_dir, device=device, model=diffusion_model)
        normal_maps = position_maps = None
        run_stage3_baking(
            mesh_path=Path(mesh_path), texture_dir=texture_dir, output_dir=output_dir,
            elevs=elevs, azims=azims, weights=weights, conf=conf, super_model=super_model, mesh=wrapped)
    finally:
        for temporary_dir in (output_dir / "geometry", texture_dir, output_dir / "images_val"):
            shutil.rmtree(temporary_dir, ignore_errors=True)
        validation_outputs = getattr(diffusion_model, "validation_step_outputs", None)
        if validation_outputs is not None:
            validation_outputs.clear()
        wrapped = normal_maps = position_maps = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def emit_ready_marker():
    ready_file = os.environ.get("NI_TEX_READY_FILE")
    if not ready_file:
        return

    ready_path = Path(ready_file)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_ready_path = ready_path.with_name(
        ready_path.name + "." + str(os.getpid()) + ".tmp"
    )
    temporary_ready_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    os.replace(temporary_ready_path, ready_path)
    print("[model-ready] " + str(ready_path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Zero-based match rank to use from each *_matched_openvton.json (0-9).",
    )
    ap.add_argument("--base", default="cfgs/inference_100k.yml")
    ap.add_argument("--out_subdir", default="NI-Tex_output")
    ap.add_argument(
        "--weird_log_dir",
        default="",
        help="Optional detector output root; process only entries in failed_cases.txt.",
    )
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.rank < 10:
        ap.error("--rank must be between 0 and 9")

    cpu_threads_raw = os.environ.get("NI_TEX_CPU_THREADS")
    if cpu_threads_raw:
        cpu_threads = int(cpu_threads_raw)
        if cpu_threads < 1:
            raise ValueError("NI_TEX_CPU_THREADS must be positive")
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Some imported libraries may initialize the inter-op pool first.
            pass
        print(
            "CPU threads: intra_op=" + str(torch.get_num_threads())
            + ", inter_op=" + str(torch.get_num_interop_threads()),
            flush=True,
        )
    
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    weird_filter_active = bool(
        str(args.weird_log_dir).strip() and Path(args.weird_log_dir).is_dir()
    )
    jobs = discover_jobs(args.root, args.out_subdir, args.rank, args.weird_log_dir)

    if not jobs:
        if weird_filter_active:
            print("No failed cases selected; skip batch.", flush=True)
            emit_ready_marker()
            return
        print("No jobs found.", flush=True); sys.exit(1)
    print(
        "Total jobs: " + str(len(jobs)) + "  (rank=" + str(args.rank)
        + ", out_subdir=" + args.out_subdir
        + ", weird_log_dir=" + (args.weird_log_dir or "<disabled>") + ")",
        flush=True,
    )

    conf = Hunyuan3DPaintConfig(max_num_view=10, resolution=512)
    timing_logger, timing_path = create_timing_logger(args.root, args.out_subdir)
    timing_logger.info(
        "run_start jobs=%d rank=%d out_subdir=%s root=%s",
        len(jobs),
        args.rank,
        args.out_subdir,
        Path(args.root).resolve(),
    )
    print("Timing log: " + str(timing_path), flush=True)

    diffusion_model = load_diffusion_model(args.base, device)
    super_model = imageSuperNet(conf)
    view_params = get_view_params(conf)

    emit_ready_marker()

    ok, failed, durations = 0, [], []
    run_t0 = time.time()
    for i, (name, mesh, img, out) in enumerate(jobs, 1):
        print("\n========== [" + str(i) + "/" + str(len(jobs)) + "] " + name + " ==========", flush=True)
        print("  mesh: " + str(mesh), flush=True)
        print("  img : " + str(img), flush=True)
        print("  out : " + str(out), flush=True)
        if args.skip_existing and job_is_complete(out):
            print("  skip (complete): " + str(out), flush=True)
            timing_logger.info(
                "job idx=%d name=%s status=skip seconds=0.0 output=%s",
                i,
                name,
                out,
            )
            ok += 1; continue
        t0 = time.time()
        try:
            process_one(mesh, img, out, conf, device, diffusion_model, super_model, view_params)
            dt = time.time() - t0; durations.append(dt)
            avg = statistics.mean(durations); eta = avg * (len(jobs) - i)
            print("  done " + name + ": " + fmt(dt) + " (" + format(dt, ".1f") + "s)   |   avg " + fmt(avg) + "/job, ETA ~" + fmt(eta), flush=True)
            timing_logger.info(
                "job idx=%d name=%s status=ok seconds=%.1f output=%s",
                i,
                name,
                dt,
                out,
            )
            ok += 1
        except Exception as e:
            dt = time.time() - t0
            print("[FAIL] " + name + ": " + str(e) + "  (after " + fmt(dt) + ")", flush=True)
            traceback.print_exc()
            timing_logger.exception(
                "job idx=%d name=%s status=fail seconds=%.1f error=%s",
                i,
                name,
                dt,
                e,
            )
            failed.append(name)

    total = time.time() - run_t0
    print("\n===== done: " + str(ok) + " ok, " + str(len(failed)) + " failed | wall " + fmt(total) + " =====", flush=True)
    timing_logger.info(
        "run_end ok=%d failed=%d wall_seconds=%.1f",
        ok,
        len(failed),
        total,
    )
    if durations:
        print("per-job (n=" + str(len(durations)) + "): mean " + fmt(statistics.mean(durations)) + ", median " + fmt(statistics.median(durations)) + ", min " + fmt(min(durations)) + ", max " + fmt(max(durations)), flush=True)
        timing_logger.info(
            "duration_stats count=%d mean=%.1f median=%.1f min=%.1f max=%.1f",
            len(durations),
            statistics.mean(durations),
            statistics.median(durations),
            min(durations),
            max(durations),
        )
    print("timing log: " + str(timing_path), flush=True)
    if failed:
        print("failed: " + ", ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
