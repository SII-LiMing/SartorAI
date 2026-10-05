"""Routines to run cloth simulation"""

# Basic
import copy
import json
import shutil
import time
import multiprocessing
import platform
import signal
from pathlib import Path

import numpy as np
import trimesh
import yaml

# BoxMeshGen
import pygarment.meshgen.boxmeshgen as bmg
from pygarment.meshgen.boxmeshgen import BoxMesh
from pygarment.meshgen.sim_config import PathCofig
from pygarment.meshgen.render.texture_utils import save_obj, texture_mesh_islands

# Warp simulation
from pygarment.meshgen.simulation import (
    MESH_EXPORT_FAIL_TYPE,
    build_part_seg,
    remove_legacy_correspondence,
    run_sim,
)


def batch_sim(data_path, output_path, dataset_props,
              run_default_body=False, num_samples=None, caching=False, force_restart=False):
    """
        Performs pattern simulation for each example in the dataset
        given by dataset_props.
        Batch processing is automatically resumed
        from the last unporcessed datapoint if restart is not forced. The last
        example on the processes list is assumed to cause the failure, so it can be later found in failure cases.

        Parameters:
            * data_path -- path to folder with patterns (for given body type)
            * output_path -- path to folder with the sumulated dataset
            * dataset_props -- dataset properties. Properties has to be of custom data_config.Properties() class and contain
                    * dataset folder (inside data_path)
                    * type of dataset structure (with/without subfolders for patterns)
                    * list of processed samples if processing of dataset was already attempted
                    * Simulation parameters
                    * Rendering parameters
                Other needed properties will be files with default values if the corresponding sections
                are not found in props object
            * run_default_body -- runs the dataset on the default body (disabled by default)
            * num_samples -- number of (unprocessed) samples from dataset to process with this run. If None, runs over all unprocessed samples
            * caching -- enables caching of every frame of simulation (disabled by default)
            * force_restart -- force restarting the batch processing even if resume conditions are met.

    """
    # ----- Init -----
    if 'frozen' in dataset_props and dataset_props['frozen']:
        # avoid accidential re-runs of data
        print('Warning: dataset is frozen, processing is skipped')
        return True

    resume = init_sim_props(dataset_props, batch_run=True, force_restart=force_restart)
    body_type = 'default_body' if run_default_body else 'random_body'
    data_props_file = output_path / f'dataset_properties_{body_type}.yaml'
    pattern_names = _get_pattern_names(data_path)
    print(
        f"[garmentcode-simulate] samples={len(pattern_names)}",
        flush=True,
    )

    # Simulate every template
    count = 0
    for progress_index, pattern_name in enumerate(pattern_names, start=1):
        progress = (
            "[garmentcode-simulate] "
            f"sample={progress_index}/{len(pattern_names)} id={pattern_name}"
        )
        # skip processed cases -- in case of resume. First condition needed to skip checking second one on False =)
        if resume and pattern_name in dataset_props['sim']['stats']['processed']:
            print(f'{progress} status=already_processed', flush=True)
            continue

        print(f'{progress} status=starting', flush=True)
        dataset_props['sim']['stats']['processed'].append(pattern_name)
        _serialize_props_with_sim_stats(dataset_props,
                                        data_props_file)  # save info of processed files before potential crash

        try:
            paths = PathCofig(
                in_element_path=data_path / pattern_name,
                out_path=output_path,
                in_name=pattern_name,
                body_name=dataset_props['body_default'],
                samples_name=dataset_props['body_samples'],
                default_body=run_default_body
            )
        except BaseException as e: 
            # Not all files available
            print("***Pattern loading failed (paths)***")
            dataset_props.add_fail('sim', 'crashes', pattern_name)
        else:
            template_simulation(paths, dataset_props, caching=caching)

        _serialize_props_with_sim_stats(dataset_props, data_props_file)
        print(f'{progress} status=attempt_finished', flush=True)

        count += 1  # count actively processed cases
        if num_samples is not None and count >= num_samples:  # only process requested number of samples
            break

    # Fin
    print(f'\nFinished batch of {data_path}')  
    try:
        if len(dataset_props['sim']['stats']['processed']) >= len(pattern_names):
            # processing successfully finished -- no need to resume later
            del dataset_props['sim']['stats']['processed']
            dataset_props['frozen'] = True
            process_finished = True
        else:
            process_finished = False
    except KeyError:
        print('KeyError -processed-')
        process_finished = True
        pass

    # Logs
    _serialize_props_with_sim_stats(dataset_props, data_props_file)

    return process_finished


def resim_fails(data_path, output_path, dataset_props,
              run_default_body=False, caching=False):
    """Resimulate failure cases -- maybe some of them would get fixed"""

    print('************** RESIMULATING FAILS ****************')

    sim_stats = dataset_props['sim']['stats']
    
    # Collect fails and remove them from fails list if any
    fails = sim_stats['fails']
    mesh_export_fails = set(fails.get(MESH_EXPORT_FAIL_TYPE, []))
    to_resim = set()
    non_resim_fail_types = {
        'cloth_body_intersection',
        'cloth_self_intersection',
        MESH_EXPORT_FAIL_TYPE,
    }
    for key, failed_samples in fails.items():
        if key in non_resim_fail_types:
            continue
        retryable_samples = set(failed_samples).difference(mesh_export_fails)
        to_resim.update(retryable_samples)
        # A mesh-export failure is deterministic for the sampled topology. Keep
        # all of its failure reasons recorded without spending another simulation.
        fails[key] = [
            sample for sample in failed_samples
            if sample in mesh_export_fails
        ]
    
    if not len(to_resim):
        # Return previous finished state
        return dataset_props['frozen'] if 'frozen' in dataset_props else False
    
    if 'processed' not in sim_stats:
        sim_stats['processed'] = _get_pattern_names(data_path)
    dataset_props['frozen'] = False

    # Remove fails from processed to trigger re-simulation
    for sample in to_resim:
        sim_stats['processed'].remove(sample)

    # Start simulation again
    finished = batch_sim(
        data_path, output_path, dataset_props, 
        run_default_body=run_default_body, 
        num_samples=len(to_resim)+1, 
        caching=caching, 
        force_restart=False
    )

    return finished

# ------- Utils -------
def init_sim_props(props, batch_run=False, force_restart=False):
    """
        Add default config values if not given in props & clean-up stats if not resuming previous processing
        Returns a flag whether current simulation is a resumed last one
    """
    if 'sim' not in props:
        props.set_section_config(
            'sim',
            max_sim_steps=1000, #affects speed
            max_meshgen_time=20, #in seconds, affects speed
            max_frame_time= 15, #in seconds, affects speed
            max_sim_time= 1500, #in seconds, affects speed
            zero_gravity_steps=10,  # 0.01  # depends on the units used, #affects speed
            static_threshold=0.03, #affects speed
            non_static_percent=1.5, #affects speed
            max_body_collisions=0,
            max_self_collisions=0,
            resolution_scale=1.0, #affects speed
            ground=False, # Do not add floor s.t. garment falls infinitely if falls
        )

    if 'material' not in props['sim']['config']:
        props['sim']['config']['material'] = {  
            'garment_tri_ka': 10000.0,  

            'garment_edge_ke': 1.0,  
            'garment_tri_ke': 10000.0,
            'spring_ke': 50000.0,  

            'garment_edge_kd': 10.0,
            'garment_tri_kd': 1.0,  
            'spring_kd': 10.0, 

            'fabric_density':  1.0,  
            'fabric_thickness': 0.1,
            'fabric_friction': 0.5

        }

    if 'options' not in props['sim']['config']:
        props['sim']['config']['options'] = {
            'enable_particle_particle_collisions': False,
            'enable_triangle_particle_collisions': True, 
            'enable_edge_edge_collisions': True,
            'enable_body_collision_filters': True,

            'enable_attachment_constraint': True,   
            'attachment_frames': 400, 
            'attachment_label_names': ['lower_interface'],
            'attachment_stiffness': [1000.],
            'attachment_damping': [10.], 

            'global_damping_factor': 0.25,
            'global_damping_effective_velocity': 0.0,
            'global_max_velocity': 25.0,

            'enable_global_collision_filter': True,
            'enable_cloth_reference_drag': False,    
            'cloth_reference_margin': 0.1,

            # FIXME Re-writes mesh references causing occasional CUDA errors when referencing meshes other than the body
            'enable_body_smoothing': False,  
            'smoothing_total_smoothing_factor': 1.0,
            'smoothing_recover_start_frame': 150,
            'smoothing_num_steps': 100,
            'smoothing_frame_gap_between_steps': 1,

            'body_collision_thickness': 0.25,
            'body_friction': 0.5
        }

    if 'render' not in props:
        # init with defaults
        props.set_section_config(
            'render',
            resolution=[800, 800],
            sides=['front','back'],
            front_camera_location=None,
            uv_texture={
                'seam_width': 0.5,
                'dpi': 1500,
            }
        )

    if batch_run and 'processed' in props['sim']['stats'] and not force_restart:
        # resuming existing batch processing -- do not clean stats
        # Assuming the last example processed example caused the failure
        last_processed = props['sim']['stats']['processed'][-1]

        if not any([(name in last_processed) or (last_processed in name) for name in
                    props['render']['stats']['render_time']]):
            # crash detected -- the last example does not appear in the stats
            if last_processed not in props['sim']['stats']['fails']['crashes']:
                # add to simulation failures
                # Remove last from processed if it did not crash
                if last_processed not in props['sim']['stats']['stop_over']:   
                    props['sim']['stats']['processed'].pop()
                else:
                    # Already passed here once -> add as crash
                    props['sim']['stats']['fails']['crashes'].append(last_processed)

        props['sim']['stats']['stop_over'].append(last_processed)  # indicate resuming dataset simulation


        return True

    # else new life
    # Prepare commulative stats
    props.set_section_stats('sim', 
                            fails={}, 
                            meshgen_time={}, 
                            sim_time={}, 
                            spf={}, 
                            fin_frame={}, 
                            face_count={},
                            body_collisions={}, 
                            self_collisions={})
    props['sim']['stats']['fails'] = {
        'crashes': [],
        'cloth_body_intersection': [],
        'cloth_self_intersection': [],
        'static_equilibrium': [],
        'fast_finish': [],
        'pattern_loading': [],
        'multi_stitching': [],
        'gt_edges_creation': [],
        MESH_EXPORT_FAIL_TYPE: [],

    }

    props.set_section_stats('render', render_time={})

    if batch_run:  # track batch processing
        props.set_section_stats('sim', processed=[], stop_over=[])

    return False


def _load_design_meta(paths: PathCofig):
    with open(paths.in_design_params, 'r') as file:
        design = yaml.load(file, Loader=yaml.SafeLoader)['design']
    return design['meta']


def _use_two_stage_no_wb(paths: PathCofig, props):
    options = props['sim']['config'].get('options', {})
    if not options.get('two_stage_no_wb_outfit', False):
        return False

    meta = _load_design_meta(paths)
    return (
        meta['upper']['v'] is not None
        and meta['bottom']['v'] is not None
        and meta['wb']['v'] is None
    )


def _load_pattern_spec(spec_path: Path):
    with open(spec_path, 'r') as file:
        return json.load(file)


def _split_pattern_spec(spec, keep_lower: bool):
    panels = spec['pattern']['panels']
    if keep_lower:
        keep_panels = {
            name for name, panel in panels.items()
            if panel.get('label') == 'leg'
        }
    else:
        keep_panels = {
            name for name, panel in panels.items()
            if panel.get('label') != 'leg'
        }

    if not keep_panels:
        split_name = 'lower' if keep_lower else 'upper'
        raise RuntimeError(f'Cannot build empty {split_name} split')

    split_spec = copy.deepcopy(spec)
    split_spec['pattern']['panels'] = {
        name: panel for name, panel in spec['pattern']['panels'].items()
        if name in keep_panels
    }
    split_spec['pattern']['stitches'] = [
        stitch for stitch in spec['pattern']['stitches']
        if all(
            edge['panel'] in keep_panels
            for edge in stitch
            if isinstance(edge, dict) and 'panel' in edge
        )
    ]
    if 'panel_order' in spec['pattern']:
        split_spec['pattern']['panel_order'] = [
            name for name in spec['pattern']['panel_order']
            if name in keep_panels
        ]

    return split_spec


def _stage_has_fail(stage_props, stage_name):
    fails = stage_props['sim']['stats'].get('fails', {})
    return any(stage_name in names for names in fails.values())


def _propagate_stage_fails(stage_props, props, stage_name, cloth_name):
    fails = stage_props['sim']['stats'].get('fails', {})
    for fail_name, names in fails.items():
        if stage_name in names:
            props.add_fail('sim', fail_name, cloth_name)


def _boxmesh_from_spec(stage_name, spec, res, props):
    stage_garment = BoxMesh(None, res)
    stage_garment.name = stage_name
    stage_garment.spec = copy.deepcopy(spec)
    stage_garment.pattern = stage_garment.spec['pattern']
    stage_garment.properties = stage_garment.spec['properties']
    stage_garment.panelNames = stage_garment.panel_order()

    meshgen_timeout = int(get_dict_default_value(props['sim']['config'], 'max_meshgen_time', 20))
    _load_boxmesh_timeout(stage_garment, meshgen_timeout)
    return stage_garment


def _segmentation_dict_from_rows(rows):
    seg_dict = {}
    for idx, row in enumerate(rows):
        entries = row if isinstance(row, list) else str(row).split(',')
        entry = "stitch" if entries[0].startswith('stitch') else entries[0]
        seg_dict.setdefault(entry, []).append(idx)
    return seg_dict


def _stage_mesh_data(stage_garment):
    vertex_labels = stage_garment.eval_vertex_labels()
    return {
        'vertices': np.asarray(stage_garment.vertices, dtype=np.float64),
        'faces': np.asarray(stage_garment.faces, dtype=np.int32),
        'faces_with_texture': np.asarray(stage_garment.faces_with_texture, dtype=np.int64),
        'vertex_texture': np.asarray(stage_garment.vertex_texture, dtype=np.float64),
        'segmentation_rows': list(stage_garment.stitch_segmentation),
        'segmentation_dict': _segmentation_dict_from_rows(stage_garment.stitch_segmentation),
        'orig_lens': stage_garment.orig_lens,
        'vertex_labels': vertex_labels,
        'pattern_spec': stage_garment.spec,
    }


def _stage_vertex_panel_names(stage_garment):
    vertex_panel_names = [None] * len(stage_garment.vertices)
    for panel_name in stage_garment.panelNames:
        panel = stage_garment.panels[panel_name]
        for face in panel.panel_faces:
            glob_ids = stage_garment._get_glob_ids(panel, face)
            if glob_ids[0] == glob_ids[1] or glob_ids[1] == glob_ids[2] or glob_ids[0] == glob_ids[2]:
                continue
            for idx in glob_ids:
                idx = int(idx)
                if vertex_panel_names[idx] is None:
                    vertex_panel_names[idx] = panel_name
    return [panel_name if panel_name is not None else 'None' for panel_name in vertex_panel_names]


def _stage_result(stage_garment, mesh_data, garment):
    sim_vertices = np.asarray(garment.current_verts, dtype=np.float64).copy()
    faces = np.asarray(mesh_data['faces'], dtype=np.int32)
    collision_mesh = trimesh.Trimesh(vertices=sim_vertices, faces=faces, process=False)
    return {
        'name': stage_garment.name,
        'box_vertices': np.asarray(mesh_data['vertices'], dtype=np.float64),
        'sim_vertices': sim_vertices,
        'faces': faces,
        'faces_with_texture': np.asarray(mesh_data['faces_with_texture'], dtype=np.int64),
        'vertex_texture': np.asarray(mesh_data['vertex_texture'], dtype=np.float64),
        'segmentation_rows': mesh_data['segmentation_rows'],
        'vertex_labels': mesh_data['vertex_labels'],
        'vertex_panel_names': _stage_vertex_panel_names(stage_garment),
        'collision_mesh': collision_mesh,
    }


def _simulate_stage(stage_name, spec, paths, props, res, save_v_norms, extra_collision_meshes=None):
    stage_garment = _boxmesh_from_spec(stage_name, spec, res, props)
    mesh_data = _stage_mesh_data(stage_garment)

    stage_props = copy.deepcopy(props)
    previous_collision_meshes = getattr(paths, 'extra_collision_meshes', None)
    if extra_collision_meshes is not None:
        paths.extra_collision_meshes = extra_collision_meshes
    elif hasattr(paths, 'extra_collision_meshes'):
        delattr(paths, 'extra_collision_meshes')

    try:
        garment = run_sim(
            stage_garment.name,
            stage_props,
            paths,
            save_v_norms=save_v_norms,
            store_usd=False,
            optimize_storage=False,
            verbose=False,
            mesh_data=mesh_data,
            write_outputs=False,
        )
    finally:
        if previous_collision_meshes is None:
            if hasattr(paths, 'extra_collision_meshes'):
                delattr(paths, 'extra_collision_meshes')
        else:
            paths.extra_collision_meshes = previous_collision_meshes

    if garment is None:
        if not _stage_has_fail(stage_props, stage_garment.name):
            stage_props.add_fail('sim', 'crashes', stage_garment.name)
        return stage_props, stage_garment.name, None

    return stage_props, stage_garment.name, _stage_result(stage_garment, mesh_data, garment)


def _collect_stage_geometry(stage_results):
    box_vertices = []
    sim_vertices = []
    faces = []
    faces_with_texture = []
    vertex_texture = []
    vertex_offset = 0
    texture_offset = 0

    for stage in stage_results:
        box_vertices.append(stage['box_vertices'])
        sim_vertices.append(stage['sim_vertices'])
        faces.append(stage['faces'] + vertex_offset)

        stage_faces_with_texture = stage['faces_with_texture'].copy()
        stage_faces_with_texture[:, [0, 2, 4]] += vertex_offset
        stage_faces_with_texture[:, [1, 3, 5]] += texture_offset
        faces_with_texture.append(stage_faces_with_texture)

        vertex_texture.append(stage['vertex_texture'])
        vertex_offset += len(stage['box_vertices'])
        texture_offset += len(stage['vertex_texture'])

    return {
        'box_vertices': np.concatenate(box_vertices, axis=0),
        'sim_vertices': np.concatenate(sim_vertices, axis=0),
        'faces': np.concatenate(faces, axis=0),
        'faces_with_texture': np.concatenate(faces_with_texture, axis=0),
        'vertex_texture': np.concatenate(vertex_texture, axis=0),
    }


def _write_mesh_ply(output_path, vertices, faces):
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(output_path)


def _write_combined_meshes(paths, stage_results, uv_config):
    geometry = _collect_stage_geometry(stage_results)
    uv_config = {
        'seam_width': 0.5,
        'dpi': 600,
        **(uv_config or {}),
    }
    uvs = texture_mesh_islands(
        texture_coords=geometry['vertex_texture'],
        face_texture_coords=geometry['faces_with_texture'][:, [1, 3, 5]],
        out_texture_image_path=paths.g_texture,
        out_mtl_file_path=paths.g_mtl,
        boundary_width=uv_config['seam_width'],
        dpi=uv_config['dpi'],
        mat_name='panels_texture',
        out_texture_image_path_replace=paths.g_texture_replace,
    )
    save_obj(
        paths.g_box_mesh,
        geometry['box_vertices'] * 0.01,
        geometry['faces_with_texture'],
        uvs,
        mtl_file_name=paths.g_mtl.name,
        mat_name='panels_texture',
    )
    save_obj(
        paths.g_sim,
        geometry['sim_vertices'] * 0.01,
        geometry['faces_with_texture'],
        uvs,
        mtl_file_name=paths.g_mtl.name,
        mat_name='panels_texture',
    )
    _write_mesh_ply(paths.g_box_mesh_compressed, geometry['box_vertices'], geometry['faces'])
    _write_mesh_ply(paths.g_sim_compressed, geometry['sim_vertices'], geometry['faces'])


def _write_combined_segmentation(output_path, stage_results):
    with open(output_path, 'w') as out_file:
        for stage in stage_results:
            for row in stage['segmentation_rows']:
                row_data = ','.join(row) if isinstance(row, list) else row
                out_file.write(row_data + '\n')


def _write_combined_vertex_labels(output_path, stage_results):
    combined = {}
    offset = 0
    for stage in stage_results:
        for label, ids in stage['vertex_labels'].items():
            combined.setdefault(label, []).extend([int(idx) + offset for idx in ids])
        offset += len(stage['box_vertices'])
    with open(output_path, 'w') as file:
        yaml.dump(combined, file, default_flow_style=False, sort_keys=False)


def _write_combined_part_seg(output_path, stage_results):
    vertex_panel_names = []
    for stage in stage_results:
        vertex_panel_names.extend(stage['vertex_panel_names'])
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(build_part_seg(vertex_panel_names), file, ensure_ascii=False, separators=(',', ':'))


def _write_two_stage_outputs(paths, stage_results, uv_config):
    _write_combined_meshes(paths, stage_results, uv_config)
    _write_combined_segmentation(paths.g_mesh_segmentation, stage_results)
    _write_combined_vertex_labels(paths.g_vert_labels, stage_results)
    remove_legacy_correspondence(paths)
    _write_combined_part_seg(paths.g_part_seg, stage_results)


def _copy_original_input_files(paths: PathCofig):
    shutil.copy2(paths.in_g_spec, paths.g_specs)
    if paths.in_body_mes.exists():
        shutil.copy2(paths.in_body_mes, paths.body_mes)
    if paths.in_design_params.exists():
        shutil.copy2(paths.in_design_params, paths.design_params)

    for suffix in ('pattern.svg', 'pattern.png'):
        src = paths.input / f'{paths.in_tag}_{suffix}'
        if src.exists():
            shutil.copy2(src, paths.out_el / src.name)


def _record_two_stage_stats(props, cloth_name, lower_props, lower_name, upper_props, upper_name):
    sim_stats = props['sim']['stats']
    lower_stats = lower_props['sim']['stats']
    upper_stats = upper_props['sim']['stats']

    sim_time = (
        lower_stats.get('sim_time', {}).get(lower_name, 0.0)
        + upper_stats.get('sim_time', {}).get(upper_name, 0.0)
    )
    lower_frame = lower_stats.get('fin_frame', {}).get(lower_name, 0)
    upper_frame = upper_stats.get('fin_frame', {}).get(upper_name, 0)
    frame_count = max(lower_frame + upper_frame, 1)

    sim_stats['sim_time'][cloth_name] = sim_time
    sim_stats['spf'][cloth_name] = sim_time / frame_count
    sim_stats['fin_frame'][cloth_name] = lower_frame + upper_frame
    sim_stats['body_collisions'][cloth_name] = (
        lower_stats.get('body_collisions', {}).get(lower_name, 0)
        + upper_stats.get('body_collisions', {}).get(upper_name, 0)
    )
    sim_stats['self_collisions'][cloth_name] = (
        lower_stats.get('self_collisions', {}).get(lower_name, 0)
        + upper_stats.get('self_collisions', {}).get(upper_name, 0)
    )


def two_stage_no_wb_simulation(
        cloth_name, props, paths: PathCofig, res, uv_config,
        caching=False, optimize_storage=False, save_v_norms=False, store_panels=False):
    print(f'Using two-stage no-waistband simulation for {cloth_name}')
    _ = caching, optimize_storage, store_panels
    shutil.rmtree(paths.out_el / '_two_stage', ignore_errors=True)
    for stale_texture in paths.out_el.glob(f'{cloth_name}_*_texture.png'):
        stale_texture.unlink()
    paths.g_texture.unlink(missing_ok=True)

    spec = _load_pattern_spec(paths.in_g_spec)
    lower_spec = _split_pattern_spec(spec, keep_lower=True)
    upper_spec = _split_pattern_spec(spec, keep_lower=False)

    lower_props, lower_name, lower_stage = _simulate_stage(
        f'{cloth_name}_lower',
        lower_spec,
        paths,
        props,
        res,
        save_v_norms,
    )
    _propagate_stage_fails(lower_props, props, lower_name, cloth_name)
    if lower_stage is None:
        print(f'{cloth_name}::WARNING::Lower-stage simulation crashed before producing a mesh; skipping this sample')
        return
    if _stage_has_fail(lower_props, lower_name):
        print(f'{cloth_name}::WARNING::Lower-stage simulation reported a failure; continuing with available mesh')

    upper_props, upper_name, upper_stage = _simulate_stage(
        f'{cloth_name}_upper',
        upper_spec,
        paths,
        props,
        res,
        save_v_norms,
        extra_collision_meshes=[lower_stage['collision_mesh']],
    )
    _propagate_stage_fails(upper_props, props, upper_name, cloth_name)
    if upper_stage is None:
        print(f'{cloth_name}::WARNING::Upper-stage simulation crashed before producing a mesh; skipping this sample')
        return

    _write_two_stage_outputs(paths, [lower_stage, upper_stage], uv_config)
    _copy_original_input_files(paths)
    _record_two_stage_stats(props, cloth_name, lower_props, lower_name, upper_props, upper_name)


def template_simulation(paths: PathCofig, props, caching=False):
    """
        Simulate given template within given scene & save log files
    """
    sim_props = props['sim']
    res = sim_props['config']['resolution_scale']

    garment = BoxMesh(paths.in_g_spec, res)

    print('\n-----------------------------'
          '\nLoading garment: ', garment.name)

    meshgen_start_time = time.time()
    timeout_after = int(get_dict_default_value(sim_props['config'], 'max_meshgen_time', 20))

    try:
        _load_boxmesh_timeout(garment, timeout_after)
    except TimeoutError as e:
        print(e)
        failure_case = 'meshgen-timeout'
        props.add_fail('sim', failure_case, garment.name)
    except bmg.PatternLoadingError as e:
        # record error and skip subequent processing
        print(e)
        failure_case = 'pattern_loading'
        props.add_fail('sim', failure_case, garment.name)
    except bmg.DegenerateTrianglesError as e:
        print(e)
        failure_case = 'degenerate_triangles'
        props.add_fail('sim', failure_case, garment.name)
    except bmg.MultiStitchingError as e:
        print(e)
        failure_case = 'multi_stitching'
        props.add_fail('sim', failure_case, garment.name)
    except bmg.NormError as e:
        print(e)
        failure_case = 'norm_error'
        props.add_fail('sim', failure_case, garment.name)
    except bmg.StitchingError as e:
        print(e)
        failure_case = 'stitching_error'
        props.add_fail('sim', failure_case, garment.name)
    except BaseException as e:   # Catch the rest of exceptions
        print("***Pattern loading failed due to unknown error***")
        print(e)
        failure_case = 'crashes'
        props.add_fail('sim', failure_case, garment.name)
    else:
        # garment.save_mesh(tag='stitched')  # Saving the geometry before eny forces were applied
        sim_props['stats']['meshgen_time'][garment.name] = time.time() - meshgen_start_time
        sim_props['stats']['face_count'][garment.name] = len(garment.faces)
        sim_props_option = sim_props['config']['options']
        
        vertex_normals = get_dict_default_value(sim_props_option,'store_vertex_normals',False)
        store_panels = get_dict_default_value(sim_props_option,'store_panels',False)
        if _use_two_stage_no_wb(paths, props):
            two_stage_no_wb_simulation(
                garment.name,
                props,
                paths,
                res,
                props['render']['config']['uv_texture'],
                caching=caching,
                optimize_storage=sim_props['config']['optimize_storage'],
                save_v_norms=vertex_normals,
                store_panels=store_panels,
            )
        else:
            garment.serialize(
                paths, 
                with_v_norms=vertex_normals, 
                store_panels=store_panels,
                uv_config=props['render']['config']['uv_texture']
            )

            run_sim(
                garment.name,  
                props, 
                paths,
                save_v_norms=vertex_normals,
                store_usd=caching,  # NOTE: False for fast simulation!, 
                optimize_storage=sim_props['config']['optimize_storage'],
                verbose=False
            )

def _load_boxmesh_timeout(garment, timeout_after):
    if platform.system() == "Windows":
        """https://stackoverflow.com/a/14920854"""
        p = multiprocessing.Process(target=garment.load(), name="GarmentGeneration")
        p.start()

        # Wait timeout_after seconds for garment.load()
        time.sleep(timeout_after)

        # If thread is active
        if p.is_alive():
            # Terminate the process
            p.terminate()
            p.join()
            raise TimeoutError

    elif platform.system() in ["Linux", "OSX"]:
        """https://code-maven.com/python-timeout"""
        def alarm_handler(signum, frame):
            raise TimeoutError

        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout_after)
        s_time = time.time()
        try:
            garment.load()
        except TimeoutError as ex:
            raise TimeoutError
        else:
            e_time = time.time() - s_time
            # print("No timeout error with time: ",e_time)
            signal.alarm(0)


def get_dict_default_value(props, name, default_value):
    if name in props:
        return props[name]
    return default_value

def _serialize_props_with_sim_stats(dataset_props, filename):
    """Compute data processing statistics and serialize props to file"""
    dataset_props.stats_summary()
    dataset_props.serialize(filename)


def _get_pattern_names(data_path: Path):
    names = []
    to_ignore = ['renders']  # special dirs not to include in the pattern list
    for el in data_path.iterdir():
        if el.is_dir() and el.stem not in to_ignore:
            names.append(el.stem)

    return names
