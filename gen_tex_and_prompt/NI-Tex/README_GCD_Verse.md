# GCD-Verse texture pipeline

`one_key.sh` runs generation and detection for fixed rounds 0-4, then gathers
the selected results into `textured/`.

## Run the new batches

```bash
bash one_key.sh --batch_list 148-171
```

The default selection is `148-171`. Lists and ranges can be mixed, e.g.
`--batch_list 148,150-153` or `--batch_list batch_148,batch_151`.

The workflow generates textures with match rank 0, detects abnormal parts, and
retries only the flagged case/part pairs with ranks 1-4. Each retry is followed
by detection. It stops early when no flagged parts remain. A nonzero
`flagged_parts` count at the last round means some parts still need review.
After the rounds finish, the script runs `gather_texture.py` for the selected
batches, collecting the latest complete result for each part into `textured/`.
After all copies succeed, it deletes the selected cases' `NI-Tex_output` and
`NI-Tex_output_regen*` directories.

## Rerunning

Generation always uses `--skip_existing`. Once gathering has cleaned up the
generation directories, rerunning the full workflow generates textures again.
A part is complete only when all five
result files listed below exist. Rerunning the same command skips complete
parts and retries incomplete outputs. Each run starts at round 0 and rewrites
detection reports for the selected batches. Rounds 1-4 use the preceding
round's reports to select the parts that need regeneration.

Generation exceptions stop the workflow before detection or a later round.
After fixing the error, rerun the command to finish that round. Abnormal-texture
regeneration instead selects another matched reference image for a successfully
generated but flagged part.

## Paths and dependencies

- The default data root is `../../../GCD-Verse/GCD-Verse`, relative to the script.
  In the current workspace this is
  `/inspire/hdd/global_user/liming-253108120187/SartorAI/GCD-Verse/GCD-Verse`.
  Override it with `--data_root /path/to/dataset`.
- Generation activates conda environment `NI-Tex`; detection and gathering activate
  `chatgarment`. `CONDA_SH` can override the conda initialization script.
- The default config is `cfgs/inference_100k.yml`. `--base` accepts an absolute
  path or a path relative to this NI-Tex directory.
- Local weights: `ckpt/step_100K.ckpt` and `ckpt/RealESRGAN_x4plus.pth`.
- Hugging Face models load from `HF_HOME` in offline mode. The default cache is
  `/inspire/hdd/global_user/liming-253108120187/Datasets/HF_HOUSE`; set `HF_HOME`
  before launching to use another cache. The inference YAML uses the model ID
  `sd2-community/stable-diffusion-2-1` instead of an absolute snapshot path.
- Requires Bash 5.1 or newer (`wait -n -p`).

Inputs are unchanged: each case supplies `*_matched_openvton.json`; `whole`
uses `*_sim.obj`, and separate pieces use `*_upper.obj` and `*_bottom.obj`.
The referenced OpenVTON image paths must exist.

## Parallelism

Defaults match the 8-GPU H200 deployment: GPUs 0-7, three generation processes
per GPU (up to 24 total), and up to 32 concurrent detection processes. Actual
concurrency is limited by the number of batches being processed; batches
148-171 supply 24 batch processes, initially distributed three per GPU for generation.
The equivalent explicit settings are:

```bash
bash one_key.sh --batch_list 148-171 \
  --gpu_ids 0-7 --processes_per_gpu 3 --detect_jobs 32
```

Generation retains the original model-loading waves: each wave launches at most
one new model per GPU and waits for readiness before the next wave. Available
GPU slots immediately receive the next batch. CPU threads are shared across
generation processes. Interrupting the script stops workers and removes its
temporary readiness markers.

## Outputs

| Round | Case texture directory | Detection report root |
| --- | --- | --- |
| 0 | `NI-Tex_output/<part>/` | `detect_weird_output/` |
| N | `NI-Tex_output_regenN/<part>/` | `detect_weird_output_after_regenN/` |

`<part>` is `upper`, `bottom`, or `whole`. Each complete result contains:

```text
results.obj
results.mtl
results.jpg
results_metallic.jpg
results_roughness.jpg
```

Detection writes `<report_root>/<batch_id>/failed_cases.txt`. Add `--vis_flag`
to save visualizations. Generation terminal logs remain under `log_terminal/`,
with timing logs under `log/`; detection logs are alongside each batch report.
Old reports for other batches are not used or removed.

`gather_texture.py` runs automatically at the end of the workflow. It also accepts
`--batch_list` for gathering previously generated batches separately:

```bash
python gather_texture.py --data_root ../../../GCD-Verse/GCD-Verse --batch_list 148-171
```

Gathering selects the latest complete files for each part; it does not filter
parts using the detector reports. Selected files are copied into `textured/`,
and all selected cases must finish copying successfully before cleanup starts.
Cleanup always deletes their `NI-Tex_output` and `NI-Tex_output_regen*`
directories; there is no option to disable it. If any copy fails, all intermediate
generation directories are retained. Cleanup failures are reported with a nonzero
exit status.
Rendering and VLM caption generation remain subsequent steps.
