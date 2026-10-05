# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import json
import sys
import time
import traceback
import platform
import multiprocessing
import signal

import numpy as np
import trimesh

# Warp
import warp as wp

# Custom code
from pygarment.meshgen.render.pythonrender import render_images
from pygarment.meshgen.render.texture_utils import save_obj, save_texture_mtl
from pygarment.meshgen.garment import Cloth
from pygarment.meshgen.sim_config import SimConfig, PathCofig

wp.init()

class SimulationError(BaseException):
    """To be rised when panel stitching cannot be executed correctly"""
    pass

class FrameTimeOutError(BaseException):
    """To be rised when frame takes too long to simulate"""
    pass

class SimTimeOutError(BaseException):
    """To be rised when simulation takes too long"""
    pass


MESH_EXPORT_FAIL_TYPE = 'mesh_export'


class PreStitchMappingError(ValueError):
    """Raised when a stitched mesh vertex has no unique pre-stitch representation."""
    pass


class MissingPerVertexUVError(ValueError):
    """Raised when a compressed mesh has no usable per-vertex UVs."""
    pass


def write_compressed_meshes(paths: PathCofig, overwrite=True):
    for src_path, dst_path in (
            (paths.g_box_mesh, paths.g_box_mesh_compressed),
            (paths.g_sim, paths.g_sim_compressed)):
        if not overwrite and dst_path.exists():
            continue
        mesh = trimesh.load(src_path)
        mesh.export(dst_path)


def optimize_garment_storage(paths: PathCofig):
    """Write compact PLY companions while keeping render-ready OBJ outputs."""
    try:
        write_compressed_meshes(paths, overwrite=False)
    except BaseException:
        pass


def rebuild_boxmesh(spec_path, resolution_scale):
    from pygarment.meshgen.boxmeshgen import BoxMesh

    boxmesh = BoxMesh(spec_path, resolution_scale)
    boxmesh.load()
    return boxmesh


def build_original_face_corner_metadata(boxmesh):
    face_metadata = []
    for panel_name in boxmesh.panelNames:
        panel = boxmesh.panels[panel_name]
        panel_vertices_3d = np.asarray(panel.rot_trans_panel(panel.panel_vertices), dtype=np.float64)
        for face in panel.panel_faces:
            f_glob_ids = boxmesh._get_glob_ids(panel, face)
            if f_glob_ids[0] == f_glob_ids[1] or f_glob_ids[1] == f_glob_ids[2] or f_glob_ids[0] == f_glob_ids[2]:
                continue
            corners = []
            for loc_id, glob_id in zip(face, f_glob_ids):
                loc_id = int(loc_id)
                corners.append((panel_name, loc_id, int(glob_id), panel_vertices_3d[loc_id]))
            face_metadata.append(corners)
    if len(face_metadata) != len(boxmesh.faces):
        raise RuntimeError(
            f"Panel face count mismatch: face_metadata={len(face_metadata)}, faces={len(boxmesh.faces)}"
        )
    return face_metadata


def build_pre_stitch_vertices(ply_faces, original_face_metadata, vertex_count, atol=1e-8):
    if len(ply_faces) != len(original_face_metadata):
        raise RuntimeError(
            f"Face count mismatch: ply_faces={len(ply_faces)}, "
            f"original_face_metadata={len(original_face_metadata)}"
        )

    vertices = np.full((vertex_count, 3), np.nan, dtype=np.float64)
    vertex_panel_names = [None] * vertex_count
    assigned_info = [None] * vertex_count
    conflicts = []
    panel_conflicts = []

    for face_id, ply_face in enumerate(ply_faces):
        corners = original_face_metadata[face_id]
        if len(ply_face) != len(corners):
            raise RuntimeError(f"Face corner mismatch at face {face_id}: ply={len(ply_face)}, metadata={len(corners)}")
        for corner_id, (vertex_id, corner) in enumerate(zip(ply_face, corners)):
            vertex_id = int(vertex_id)
            panel_name, loc_id, glob_id, pre_stitch_coord = corner
            if np.isnan(vertices[vertex_id, 0]):
                vertices[vertex_id] = pre_stitch_coord
                vertex_panel_names[vertex_id] = panel_name
                assigned_info[vertex_id] = (face_id, corner_id, panel_name, loc_id, glob_id)
                continue

            if vertex_panel_names[vertex_id] != panel_name:
                panel_conflicts.append(
                    (
                        vertex_id,
                        assigned_info[vertex_id],
                        (face_id, corner_id, panel_name, loc_id, glob_id),
                    )
                )
            if not np.allclose(vertices[vertex_id], pre_stitch_coord, atol=atol, rtol=0.0):
                conflicts.append(
                    (
                        vertex_id,
                        assigned_info[vertex_id],
                        (face_id, corner_id, panel_name, loc_id, glob_id),
                    )
                )

    if conflicts:
        vertex_id, old_info, new_info = conflicts[0]
        raise PreStitchMappingError(
            "Cannot build one-to-one pre-stitch boxmesh: "
            f"vertex {vertex_id} maps to different panel coordinates. "
            f"old={old_info}, new={new_info}, conflicts={len(conflicts)}"
        )

    if panel_conflicts:
        vertex_id, old_info, new_info = panel_conflicts[0]
        raise PreStitchMappingError(
            "Cannot assign a unique panel_name to every vertex: "
            f"vertex {vertex_id} is referenced by multiple panels. "
            f"old={old_info}, new={new_info}, conflicts={len(panel_conflicts)}"
        )

    unassigned = np.flatnonzero(np.isnan(vertices[:, 0]))
    if len(unassigned):
        sample = unassigned[:10].tolist()
        raise PreStitchMappingError(
            "Cannot build one-to-one pre-stitch boxmesh: "
            f"{len(unassigned)} PLY vertices are not referenced by any face; sample={sample}"
        )

    return vertices, vertex_panel_names


def build_part_seg(vertex_panel_names):
    parts = []
    part_to_id = {}
    vertex_part_ids = []
    for panel_name in vertex_panel_names:
        if panel_name not in part_to_id:
            part_to_id[panel_name] = len(parts)
            parts.append(panel_name)
        vertex_part_ids.append(part_to_id[panel_name])
    return {
        'parts': parts,
        'vertex_part_ids': vertex_part_ids,
    }


def remove_legacy_correspondence(paths: PathCofig):
    legacy_path = paths.out_el / f'{paths.boxmesh_tag}_boxmesh_sim_correspondence.json'
    legacy_path.unlink(missing_ok=True)


def read_obj_material_info(obj_path, default_mtl_name, default_mat_name='panels_texture'):
    mtl_name = default_mtl_name
    mat_name = default_mat_name
    with open(obj_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('mtllib '):
                mtl_name = line.split(maxsplit=1)[1].strip()
            elif line.startswith('usemtl '):
                mat_name = line.split(maxsplit=1)[1].strip()
    return mtl_name, mat_name


def load_textured_ply(path):
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh from {path}, got {type(mesh)}")
    uvs = getattr(mesh.visual, 'uv', None)
    if uvs is None or len(uvs) != len(mesh.vertices):
        raise MissingPerVertexUVError(f"PLY has no per-vertex UVs: {path}")
    return mesh, np.asarray(uvs, dtype=np.float64)


def export_render_ready_meshes(paths: PathCofig, resolution_scale, vertex_scale=0.01):
    boxmesh_ply, box_uvs = load_textured_ply(paths.g_box_mesh_compressed)
    sim_ply, sim_uvs = load_textured_ply(paths.g_sim_compressed)
    box_faces = np.asarray(boxmesh_ply.faces, dtype=np.int64)
    sim_faces = np.asarray(sim_ply.faces, dtype=np.int64)

    if len(boxmesh_ply.vertices) != len(sim_ply.vertices):
        raise ValueError(
            f"Vertex count mismatch: box={len(boxmesh_ply.vertices)}, sim={len(sim_ply.vertices)}"
        )
    if not np.array_equal(box_faces, sim_faces):
        raise ValueError("Face index order mismatch; direct same-index export is unsafe.")
    if not np.allclose(box_uvs, sim_uvs, atol=0.0, rtol=0.0):
        raise ValueError("BoxMesh and sim UVs differ; direct same-index textured export is unsafe.")

    rebuilt = rebuild_boxmesh(paths.in_g_spec, resolution_scale)
    original_faces = np.asarray(rebuilt.faces, dtype=np.int64)
    original_face_metadata = build_original_face_corner_metadata(rebuilt)
    if len(original_faces) != len(box_faces):
        raise ValueError(
            f"Rebuilt BoxMesh face count mismatch: rebuilt={len(original_faces)}, exported_ply={len(box_faces)}"
        )

    pre_stitch_box_vertices, vertex_panel_names = build_pre_stitch_vertices(
        box_faces,
        original_face_metadata,
        len(boxmesh_ply.vertices),
    )
    faces_with_texture = [
        [int(v0), int(v0), int(v1), int(v1), int(v2), int(v2)]
        for v0, v1, v2 in box_faces
    ]
    mtl_file_name, mat_name = read_obj_material_info(paths.g_box_mesh, paths.g_mtl.name)
    mtl_path = paths.g_box_mesh.parent / mtl_file_name
    save_texture_mtl(mtl_path, paths.g_texture.name, mat_name=mat_name)

    save_obj(
        paths.g_box_mesh,
        pre_stitch_box_vertices * vertex_scale,
        faces_with_texture,
        box_uvs,
        mtl_file_name=mtl_file_name,
        mat_name=mat_name,
    )
    save_obj(
        paths.g_sim,
        np.asarray(sim_ply.vertices, dtype=np.float64) * vertex_scale,
        faces_with_texture,
        sim_uvs,
        mtl_file_name=mtl_file_name,
        mat_name=mat_name,
    )

    remove_legacy_correspondence(paths)
    with open(paths.g_part_seg, 'w', encoding='utf-8') as file_obj:
        json.dump(build_part_seg(vertex_panel_names), file_obj, ensure_ascii=False, separators=(',', ':'))


def update_progress(progress, total):
    """Progress bar in console"""
    # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    amtDone = progress / total
    num_dash = int(amtDone * 50)
    sys.stdout.write('\rProgress: [{0:50s}] {1:.1f}%'.format('#' * num_dash + '-' * (50 - num_dash), amtDone * 100))
    sys.stdout.flush()


def sim_frame_sequence(garment, config, store_usd=False, verbose=False):

    frame_timeout_after = config.max_frame_time
    # Save initial state
    if store_usd:
        garment.render_usd_frame()

    start_time = time.time()
    for frame in range(0, config.max_sim_steps):
        
        if verbose:
            print(f'\n------ Frame {frame + 1} ------')
        else:
            update_progress(frame, config.max_sim_steps)

        garment.frame = frame 

        #Run frame and raise FrameTimeOutError if frame takes too long to simulate

        static = False
        if frame == 0:
            frame_timeout_after *= 2
        try:
            if platform.system() == "Windows":
                """https://stackoverflow.com/a/14920854"""

                if frame == 0: #only do it on first frame due to slowdown
                    p_frame = multiprocessing.Process(target=garment.run_frame(), name="FrameSimulation")
                    p_frame.start()

                    # Wait timeout_after seconds for garment.run_frame()
                    p_frame.join(frame_timeout_after)

                    # If thread is active
                    if p_frame.is_alive():
                        # Terminate the process
                        p_frame.terminate()
                        p_frame.join()
                        raise TimeoutError
                else:
                    garment.run_frame()

            elif platform.system() in ["Linux", "OSX"]:
                """https://code-maven.com/python-timeout"""

                def alarm_handler(signum, frame):
                    raise TimeoutError

                signal.signal(signal.SIGALRM, alarm_handler)
                signal.alarm(frame_timeout_after)
                s_time = time.time()
                try:
                    garment.run_frame()
                except TimeoutError as ex:
                    raise TimeoutError

                else:
                    e_time = time.time() - s_time
                    # print("No timeout error with time: ", e_time)
                    signal.alarm(0)

                pass
        except TimeoutError as e:
            raise FrameTimeOutError

        if verbose:
            num_cloth_cloth_contacts = garment.count_self_intersections()
            print(f'\nSelf-Intersection: {num_cloth_cloth_contacts}')

        if frame >= config.zero_gravity_steps and frame >= config.min_sim_steps:
            static, _ = garment.is_static()
        if static:
            break

        runtime = time.time() - start_time
        if runtime > config.max_sim_time:
            raise SimTimeOutError
        

def run_sim(
        cloth_name, props, paths: PathCofig, 
        save_v_norms=False, store_usd=False, 
        optimize_storage=False,
        verbose=False,
        mesh_data=None,
        write_outputs=True): 
    """Initialize and run the simulation
    !! Important !! 
        'store_usd' parameter slows down the simulation to CPU rates because of required CPU-GPU copies and file writes. Use only for debugging
    """
    sim_props = props['sim']
    render_props = props['render']

    start_time = time.time()

    config = SimConfig(sim_props['config'])   # Why separate class at all?
    garment = None

    try:
        garment = Cloth(cloth_name, config, paths, caching=store_usd, mesh_data=mesh_data)
        print("Simulation..")
        sim_frame_sequence(garment, config, store_usd, verbose=verbose)
    
    except FrameTimeOutError:
        print(f"FrameTimeOutError at frame {garment.frame}")
        props.add_fail('sim', 'frame_timeout', cloth_name)
    except SimTimeOutError:
        print("SimTimeOutError")
        props.add_fail('sim', 'simulation_timeout', cloth_name)
    except SimulationError:
        print("Simulation failed")
        props.add_fail('sim', 'gt_edges_creation', cloth_name)
    except BaseException as e:
        print(f'Sim::{cloth_name}::crashed with {e}')

        if isinstance(e, KeyboardInterrupt):
            # Allow to stop simulation loops by keyboard interrupt
            # It's not a real crash, so don't write down the failure
            sec = round(time.time() - start_time, 3)
            min = int(sec / 60)
            print(f"Simulation pipeline took: {min} m {sec - min * 60} s")
            raise e

        traceback.print_exc()
        if 'Invalid edge lengths' in str(e):
            props.add_fail('sim', 'gt_edges_creation', cloth_name)
        else:
            props.add_fail('sim', 'crashes', cloth_name)

        if garment is None:
            sec = round(time.time() - start_time, 3)
            min = int(sec / 60)
            print(f"\nSimulation pipeline took: {min} m {sec - min * 60} s")
            return None
    else:  # Other quality checks
        if garment.frame == config.max_sim_steps - 1:
            _, non_st_count = garment.is_static()
            print('\nFailed to achieve static equilibrium for {} with {} non-static vertices out of {}'.format(
                cloth_name, non_st_count, len(garment.current_verts)))
            props.add_fail('sim', 'static_equilibrium', cloth_name)

        if time.time() - start_time < 0.5:  # 0.5 sec  -- finished suspiciously fast
            props.add_fail('sim', 'fast_finish', cloth_name)

        # 3D penetrations
        num_body_collisions = garment.count_body_intersections()
        print("BODY CLOTH INTERSECTIONS: ", num_body_collisions)
        num_self_collisions = garment.count_self_intersections()

        sim_props['stats']['body_collisions'][cloth_name] = num_body_collisions
        sim_props['stats']['self_collisions'][cloth_name] = num_self_collisions

        if num_body_collisions > config.max_body_collisions:
            props.add_fail('sim', 'cloth_body_intersection', cloth_name)
        if num_self_collisions: 
            print(f'Self-Intersecting with {num_self_collisions}, '
                  f'is fail: {num_self_collisions > config.max_self_collisions}')
            if num_self_collisions > config.max_self_collisions:
                props.add_fail('sim', 'cloth_self_intersection', cloth_name)
        else:
            print('Not self-intersecting!!!')

    # ---- Postprocessing ----
    # NOTE: Attempt even on failures for accurate picture and post-analysis
    frame = garment.frame
    print(f"\nSimulation took #frames={frame + 1}")

    sim_props['stats']['sim_time'][cloth_name] = sim_time = time.time() - start_time
    sim_props['stats']['spf'][cloth_name] = sim_time / frame if frame else sim_time
    sim_props['stats']['fin_frame'][cloth_name] = frame

    if write_outputs:
        garment.save_frame(save_v_norms=save_v_norms) #saving after stats
        write_compressed_meshes(paths, overwrite=True)
        try:
            export_render_ready_meshes(paths, config.resolution_scale)
        except (PreStitchMappingError, MissingPerVertexUVError) as exc:
            print(
                f'MeshExport::{cloth_name}::ERROR::'
                f'{type(exc).__name__}: {exc}'
            )
            traceback.print_exc()
            props.add_fail('sim', MESH_EXPORT_FAIL_TYPE, cloth_name)

    # # Render images. Headless EGL/OSMesa support varies across machines, so a
    # # rendering failure should not discard an otherwise completed simulation.
    # s_time = time.time()
    # try:
    #     render_images(paths, garment.v_body, garment.f_body, render_props['config'])
    # except Exception as exc:
    #     render_image_time = time.time() - s_time
    #     render_props['stats']['render_time'][cloth_name] = render_image_time
    #     props.add_fail('render', 'render_crashes', cloth_name)
    #     print(f"Rendering {cloth_name} failed after {render_image_time}s: {exc}")
    # else:
    #     render_image_time = time.time() - s_time
    #     render_props['stats']['render_time'][cloth_name] = render_image_time
    #     print(f"Rendering {cloth_name} took {render_image_time}s")

    if write_outputs and optimize_storage:
        optimize_garment_storage(paths)

    # Final info output
    sec = round(time.time() - start_time, 3)
    min = int(sec / 60)
    print(f"\nSimulation pipeline took: {min} m {sec - min * 60} s")
    return garment
