import subprocess
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None
    Vector = None


BLENDER_EXECUTABLE = Path(
    "/inspire/hdd/global_user/liming-253108120187/softwares/"
    "blender-5.1.2-linux-x64/blender"
)
OUT_SUBDIR = "textured"
SKIN_COLOR = (0.90, 0.68, 0.52, 1.0)
RESOLUTION = (1024, 1024)
LIGHT_KEY = 75
LIGHT_FILL = 40
LIGHT_RIM = 30
FABRIC_METALLIC = 0.0
FABRIC_ROUGHNESS = 0.7


def launch_blender(script_path, argv, python_exit_code=None):
    if not BLENDER_EXECUTABLE.is_file():
        raise FileNotFoundError(
            "Blender executable not found: " + str(BLENDER_EXECUTABLE)
        )

    command = [str(BLENDER_EXECUTABLE), "--background"]
    if python_exit_code is not None:
        command.extend(("--python-exit-code", str(python_exit_code)))
    command.extend(("--python", str(Path(script_path).resolve()), "--", *argv))
    return subprocess.run(command, check=False).returncode


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    blocks = (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.textures,
        bpy.data.cameras,
        bpy.data.lights,
    )
    for block in blocks:
        for item in list(block):
            if not item.users:
                block.remove(item)


def import_obj(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh imported from " + str(path))
    return meshes


def make_skin_material(name, color=SKIN_COLOR):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Metallic"].default_value = 0.0
    try:
        bsdf.inputs["Subsurface Weight"].default_value = 0.05
    except KeyError:
        pass
    return material


def fix_fabric_material(objects):
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or not material.use_nodes:
                continue
            node_tree = material.node_tree
            bsdf = node_tree.nodes.get("Principled BSDF")
            if bsdf is None:
                continue
            metallic = bsdf.inputs.get("Metallic")
            if metallic and metallic.is_linked:
                for link in list(metallic.links):
                    node_tree.links.remove(link)
            if metallic:
                metallic.default_value = FABRIC_METALLIC
            roughness = bsdf.inputs.get("Roughness")
            if roughness and roughness.is_linked:
                for link in list(roughness.links):
                    node_tree.links.remove(link)
            if roughness:
                roughness.default_value = FABRIC_ROUGHNESS
            specular = bsdf.inputs.get("Specular IOR Level")
            if specular:
                specular.default_value = 0.3


def assign_material(objects, material):
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def smooth_shade(objects):
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
        obj.select_set(False)


def scene_bounds(objects):
    bpy.context.view_layer.update()
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            minimum.x = min(minimum.x, point.x)
            minimum.y = min(minimum.y, point.y)
            minimum.z = min(minimum.z, point.z)
            maximum.x = max(maximum.x, point.x)
            maximum.y = max(maximum.y, point.y)
            maximum.z = max(maximum.z, point.z)
    return minimum, maximum


def look_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def configure_cycles_gpu(scene):
    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if cycles_addon is None:
        raise RuntimeError("Cycles addon is not available")

    preferences = cycles_addon.preferences
    backend_errors = []
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            if hasattr(preferences, "refresh_devices"):
                preferences.refresh_devices()
            else:
                preferences.get_devices()
        except Exception as exc:
            backend_errors.append(backend + ": " + str(exc))
            continue

        gpu_devices = [
            device for device in preferences.devices if device.type == backend
        ]
        if not gpu_devices:
            backend_errors.append(backend + ": no devices found")
            continue

        for device in preferences.devices:
            device.use = device.type == backend
        scene.cycles.device = "GPU"
        if backend == "OPTIX":
            try:
                scene.cycles.denoiser = "OPTIX"
            except (AttributeError, TypeError):
                pass
        print(
            "Cycles GPU: backend={} devices={}".format(
                backend,
                ", ".join(device.name for device in gpu_devices),
            )
        )
        return

    raise RuntimeError(
        "No usable Cycles GPU device. " + "; ".join(backend_errors)
    )


def configure_render(resolution, transparent=True):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.film_transparent = transparent
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Contrast"
    scene.view_settings.exposure = -0.3
    scene.view_settings.gamma = 1.0
    configure_cycles_gpu(scene)
