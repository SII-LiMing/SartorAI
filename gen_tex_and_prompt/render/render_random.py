import argparse
import logging
import math
import random
import sys
import time
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render import (
    LIGHT_FILL,
    LIGHT_KEY,
    LIGHT_RIM,
    OUT_SUBDIR,
    RESOLUTION,
    assign_material,
    clear_scene,
    configure_render,
    fix_fabric_material,
    import_obj,
    launch_blender,
    look_at,
    make_skin_material,
    scene_bounds,
    smooth_shade,
)


OUTPUT_SUFFIX = "_render.png"
LOG_DIR = SCRIPT_DIR / "log"
CAMERA_AZIMUTH_JITTER_DEGREES = 15.0
CAMERA_ELEVATION_JITTER_DEGREES = 8.0
CAMERA_DISTANCE_JITTER = 0.12
CAMERA_TARGET_JITTER = 0.025
CAMERA_SCALE_JITTER = 0.04
CAMERA_MARGIN = 1.15
LIGHT_ENERGY_JITTER = 0.25
LIGHT_POSITION_JITTER = 0.30
LIGHT_SIZE_JITTER = 0.15
WORLD_STRENGTH_RANGE = (0.20, 0.40)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one randomized front view for each garment case."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--body_mesh_dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=list(RESOLUTION))
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional base seed. The effective seed is stable per case name.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a case when <case_id>_render.png already exists.",
    )
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return parser.parse_args(argv)


def create_timing_logger(base_dir):
    scope = base_dir.parent.name if base_dir.name == "random_body" else base_dir.name
    safe_scope = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in scope
    ) or "random_body"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (safe_scope + "_render_random.log")

    logger = logging.getLogger("render_random.timing")
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


def read_body_sample(yaml_path):
    body_indent = None
    for line_number, raw_line in enumerate(
        yaml_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        content = stripped.split("#", 1)[0].rstrip()
        if body_indent is None:
            if indent == 0 and content == "body:":
                body_indent = indent
            continue
        if indent <= body_indent:
            break

        key, separator, value = content.partition(":")
        if separator and key.strip() == "body_sample":
            body_sample = value.strip()
            if (
                len(body_sample) >= 2
                and body_sample[0] == body_sample[-1]
                and body_sample[0] in {"'", '"'}
            ):
                body_sample = body_sample[1:-1].strip()
            if not body_sample:
                raise ValueError(
                    "Empty body.body_sample at {}:{}".format(
                        yaml_path,
                        line_number,
                    )
                )
            return body_sample

    raise ValueError("Missing body.body_sample in " + str(yaml_path))


def resolve_body_mesh(rand_dir, body_mesh_dir):
    rand_name = rand_dir.name
    local_body = rand_dir / (rand_name + "_bodymesh.obj")
    if local_body.is_file():
        return local_body

    measurements = rand_dir / (rand_name + "_body_measurements.yaml")
    if not measurements.is_file():
        raise FileNotFoundError("Body measurements not found: " + str(measurements))

    body_sample = read_body_sample(measurements)
    if Path(body_sample).name != body_sample or body_sample in {".", ".."}:
        raise ValueError("Invalid body_sample in {}: {}".format(measurements, body_sample))

    dataset_body = body_mesh_dir / (body_sample + ".obj")
    if not dataset_body.is_file():
        raise FileNotFoundError(
            "Body mesh not found for {}: {}".format(body_sample, dataset_body)
        )
    print("    body fallback: {} -> {}".format(body_sample, dataset_body))
    return dataset_body


def setup_random_camera(objects, resolution, rng):
    min_c, max_c = scene_bounds(objects)
    center = (min_c + max_c) * 0.5
    extents = max_c - min_c

    azimuth = -90.0 + rng.uniform(
        -CAMERA_AZIMUTH_JITTER_DEGREES,
        CAMERA_AZIMUTH_JITTER_DEGREES,
    )
    elevation = rng.uniform(
        -CAMERA_ELEVATION_JITTER_DEGREES,
        CAMERA_ELEVATION_JITTER_DEGREES,
    )
    distance_scale = 1.0 + rng.uniform(
        -CAMERA_DISTANCE_JITTER,
        CAMERA_DISTANCE_JITTER,
    )
    framing_scale = 1.0 + rng.uniform(
        -CAMERA_SCALE_JITTER,
        CAMERA_SCALE_JITTER,
    )
    target_offset = Vector((
        rng.uniform(-extents.x, extents.x) * CAMERA_TARGET_JITTER,
        rng.uniform(-extents.y, extents.y) * CAMERA_TARGET_JITTER,
        rng.uniform(-extents.z, extents.z) * CAMERA_TARGET_JITTER,
    ))

    azimuth_radians = math.radians(azimuth)
    elevation_radians = math.radians(elevation)
    distance = max(extents.x * 4.0, extents.y * 4.0, 3.0) * distance_scale
    camera_offset = Vector((
        math.cos(azimuth_radians) * math.cos(elevation_radians) * distance,
        math.sin(azimuth_radians) * math.cos(elevation_radians) * distance,
        math.sin(elevation_radians) * distance,
    ))

    camera_data = bpy.data.cameras.new("random_front_camera")
    camera = bpy.data.objects.new("random_front_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + camera_offset
    look_at(camera, center + target_offset)
    camera_data.type = "ORTHO"
    horizontal_extent = max(extents.x, extents.y)
    camera_data.ortho_scale = max(
        extents.z,
        horizontal_extent * resolution[1] / resolution[0],
    ) * CAMERA_MARGIN * framing_scale
    bpy.context.scene.camera = camera

    print(
        "    camera: azimuth={:.2f}, elevation={:.2f}, "
        "distance_scale={:.3f}, framing_scale={:.3f}, "
        "target_offset=({:.4f}, {:.4f}, {:.4f})".format(
            azimuth,
            elevation,
            distance_scale,
            framing_scale,
            target_offset.x,
            target_offset.y,
            target_offset.z,
        )
    )


def setup_random_lights(objects, rng):
    min_c, max_c = scene_bounds(objects)
    center = (min_c + max_c) * 0.5
    diagonal = (max_c - min_c).length

    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    world_strength = rng.uniform(*WORLD_STRENGTH_RANGE)
    if background:
        background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        background.inputs["Strength"].default_value = world_strength

    base_lights = (
        ((0.0, -2.5, 2.0), LIGHT_KEY),
        ((-2.0, -1.0, 1.5), LIGHT_FILL),
        ((2.0, -1.0, 1.5), LIGHT_FILL),
        ((0.0, 2.0, 1.5), LIGHT_RIM),
    )
    light_scale = max(diagonal * 0.45, 1.0)
    energies = []
    for index, (base_position, base_energy) in enumerate(base_lights):
        position = Vector(tuple(
            coordinate + rng.uniform(-LIGHT_POSITION_JITTER, LIGHT_POSITION_JITTER)
            for coordinate in base_position
        ))
        energy = base_energy * (
            1.0 + rng.uniform(-LIGHT_ENERGY_JITTER, LIGHT_ENERGY_JITTER)
        )
        size = max(diagonal * 0.8, 2.0) * (
            1.0 + rng.uniform(-LIGHT_SIZE_JITTER, LIGHT_SIZE_JITTER)
        )

        light_data = bpy.data.lights.new("random_light_" + str(index), type="AREA")
        light_data.energy = energy
        light_data.size = size
        light = bpy.data.objects.new("random_light_" + str(index), light_data)
        bpy.context.collection.objects.link(light)
        light.location = center + position * light_scale
        look_at(light, center)
        energies.append(energy)

    print(
        "    lighting: world={:.3f}, energies={}".format(
            world_strength,
            ", ".join("{:.1f}".format(energy) for energy in energies),
        )
    )


def render_one_sample(rand_dir, resolution, rng, body_mesh_dir):
    rand_name = rand_dir.name
    out_root = rand_dir / OUT_SUBDIR
    upper_obj = out_root / "upper" / "results.obj"
    bottom_obj = out_root / "bottom" / "results.obj"
    whole_obj = out_root / "whole" / "results.obj"
    output_path = rand_dir / (rand_name + OUTPUT_SUFFIX)

    is_split = upper_obj.is_file() and bottom_obj.is_file()
    is_whole = whole_obj.is_file()
    if not is_split and not is_whole:
        print("    Missing garment results.obj: " + rand_name)
        return False
    body_path = resolve_body_mesh(rand_dir, body_mesh_dir)

    clear_scene()
    configure_render(resolution, transparent=True)
    scene = bpy.context.scene
    if hasattr(scene.cycles, "seed"):
        scene.cycles.seed = rng.randrange(0, 2**31)

    body_meshes = import_obj(body_path)
    assign_material(body_meshes, make_skin_material("skin"))

    garment_meshes = []
    if is_split:
        upper_meshes = import_obj(upper_obj)
        bottom_meshes = import_obj(bottom_obj)
        fix_fabric_material(upper_meshes)
        fix_fabric_material(bottom_meshes)
        garment_meshes.extend(upper_meshes)
        garment_meshes.extend(bottom_meshes)
        print("    split: body + upper + bottom")
    else:
        whole_meshes = import_obj(whole_obj)
        fix_fabric_material(whole_meshes)
        garment_meshes.extend(whole_meshes)
        print("    whole: body + whole")

    all_meshes = body_meshes + garment_meshes
    smooth_shade(all_meshes)
    setup_random_lights(all_meshes, rng)
    setup_random_camera(all_meshes, resolution, rng)

    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    print("    saved: " + str(output_path))
    return True





def main():
    args = parse_args()
    body_mesh_dir = args.body_mesh_dir.expanduser().resolve()
    resolution = tuple(args.resolution)
    if any(value <= 0 for value in resolution):
        raise ValueError("Resolution values must be positive")

    base_dir = Path(args.root).expanduser().resolve()
    if base_dir.name != "random_body":
        base_dir = base_dir / "random_body"
    if not base_dir.is_dir():
        raise FileNotFoundError("random_body directory not found: " + str(base_dir))

    base_seed = args.seed
    if base_seed is None:
        base_seed = random.SystemRandom().randrange(0, 2**63)
    print("seed=" + str(base_seed))

    rand_dirs = sorted(
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.startswith("rand_")
    )
    print("samples=" + str(len(rand_dirs)))
    timing_logger, log_path = create_timing_logger(base_dir)
    print("timing_log=" + str(log_path))
    timing_logger.info(
        "run_start root=%s cases=%d seed=%d resolution=%dx%d",
        base_dir,
        len(rand_dirs),
        base_seed,
        resolution[0],
        resolution[1],
    )

    success = 0
    skipped = 0
    failed = []
    run_started = time.perf_counter()
    for index, rand_dir in enumerate(rand_dirs, 1):
        rand_name = rand_dir.name
        output_path = rand_dir / (rand_name + OUTPUT_SUFFIX)
        print("[" + str(index) + "/" + str(len(rand_dirs)) + "] " + rand_name)
        case_started = time.perf_counter()
        status = "failed"
        error = ""
        rng = random.Random(str(base_seed) + ":" + rand_name)
        try:
            if args.skip_existing and output_path.is_file():
                print("    skip existing: " + str(output_path))
                skipped += 1
                status = "skipped"
                continue

            if render_one_sample(rand_dir, resolution, rng, body_mesh_dir):
                success += 1
                status = "success"
            else:
                failed.append(rand_name)
                error = "render_one_sample returned False"
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failed.append(rand_name)
            status = "error"
            error = "{}: {}".format(type(exc).__name__, exc)
        finally:
            elapsed = time.perf_counter() - case_started
            message = (
                "case_id={} status={} seconds={:.3f} output={}".format(
                    rand_name,
                    status,
                    elapsed,
                    output_path,
                )
            )
            if error:
                message += " error=" + repr(error)
            timing_logger.info(message)

    run_elapsed = time.perf_counter() - run_started
    print(
        "done: success={}, skipped={}, failed={}, total={}, seconds={:.3f}".format(
            success,
            skipped,
            len(failed),
            len(rand_dirs),
            run_elapsed,
        )
    )
    timing_logger.info(
        "run_end success=%d skipped=%d failed=%d total=%d seconds=%.3f",
        success,
        skipped,
        len(failed),
        len(rand_dirs),
        run_elapsed,
    )
    for name in failed[:30]:
        print("    fail " + name)


if __name__ == "__main__":
    if bpy is None:
        raise SystemExit(
            launch_blender(__file__, sys.argv[1:], python_exit_code=1)
        )
    main()
