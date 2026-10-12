# Batch Generation Guide

`one_key_batch.sh` generates GarmentCode data in separate batches and runs the following stages:

```text
sample -> simulate -> filter -> check
```

Each batch is stored under `<name>/batch_<index>/`. Results remain separate across batches.

## Requirements and paths

Run the examples from the `GarmentCodeVerse` directory. The script changes its working directory to its own location before running any stage.

- Use Bash with support for `wait -n -p`.
- Activate a Python environment with the [project dependencies](docs/Installation.md). The script uses `python` from the current environment and does not activate a conda environment.
- Configure `system.json`: `design_file` selects the design YAML; `bodies_default_path` selects the default body directory; `body_samples_path` and `body_samples` locate the random body dataset.
- For this wrapper, `datasets_path` and `datasets_sim` must resolve to the script directory, for example by setting both to `""` or `"."`. The wrapper locates batches under `<script_dir>/<name>` and does not follow other output roots.
- Simulation reads `sim_config_file` from `system.json` (currently `./assets/Sim_props/gcd_verse_sim_props.yaml`). Direct calls to `pattern_data_sim.py` can override it with `--config`, using a filename under `sim_configs_path` or an absolute path.

## Basic usage

Sample a small dataset:

```bash
bash one_key_batch.sh --name test_batch --size 30 --batch_size 10 --parallel_jobs 3 --op sample
```

Run a single stage on existing batches:

```bash
bash one_key_batch.sh --name test_batch --size 30 --batch_size 10 --op simulate
bash one_key_batch.sh --name test_batch --size 30 --batch_size 10 --op filter
bash one_key_batch.sh --name test_batch --size 30 --batch_size 10 --op check
```

Each command runs only the specified stage. Required outputs from earlier stages must already exist. Keep `--size`, `--batch_size`, and the selected body type consistent when processing an existing dataset.

Omit `--op` to run the full pipeline on a new dataset once all dependencies are available:

```bash
bash one_key_batch.sh --name full_pipeline_test --size 30 --batch_size 10 --parallel_jobs 3
```

Only one `--op` is accepted per invocation. Use `bash one_key_batch.sh --help` to view the command-line options.

## Options and defaults

Command-line values override the defaults in the script.

| Option | Default | Meaning |
| --- | --- | --- |
| `--name` | `garmentcode_144k` | Dataset directory relative to the script directory. |
| `--size` | `140000` | Target number of designs across all batches. |
| `--batch_size` | `1000` | Target number of designs per batch. |
| `--parallel_jobs` | `6` | Maximum number of concurrent batch workers. |
| `--max_hours` | `0` | Hours from command startup before pausing simulation; `0` disables the limit. Requires `--op simulate` when nonzero. |
| `--seed` | `42` | Global seed used to derive each batch's sampling seed. |
| `--batch_list` | `all` | Batch indices to process. |
| `--body_type` | See below | `random_body`, `default_body`, or `both`. |
| `--op` | Full pipeline | One of `sample`, `simulate`, `filter`, or `check`. |

`--name` must be a nonempty relative path without `..`. `--size`, `--batch_size`, and `--parallel_jobs` must be positive integers. The dataset name does not determine its size: the default name contains `144k`, but the default target is `140000`.

### Body selection

| `--body_type` | Sampling | Simulation, filtering, and checking |
| --- | --- | --- |
| Omitted | Both body types | `random_body` |
| `both` | Both body types | `random_body` |
| `random_body` | Random body only | `random_body` |
| `default_body` | Default body only | `default_body` |

Sampling loads, validates, and saves only the selected body types. With `both`, each accepted design has one version on each body type and must pass validation on both. `--size` counts designs, so selecting `both` does not double the number of unique designs.

Sample random bodies only:

```bash
bash one_key_batch.sh --name garmentcode_20k --size 20000 --batch_size 1000 --op sample --body_type random_body
```

Sample default bodies only:

```bash
bash one_key_batch.sh --name default_body_test --size 30 --batch_size 10 --op sample --body_type default_body
```

Pass `--body_type default_body` again when running later stages on default body samples. `both` does not run later stages twice.

### Dataset size

The number of batches is:

```text
num_batches = ceil(size / batch_size)
```

The final batch can be smaller than `batch_size`. These are sampling targets: a sample can be skipped after exhausting the sampler's retry limit, and filtering can further reduce the number of retained samples.

Selecting a subset with `--batch_list` does not redistribute `--size`. For example, `--size 20000 --batch_size 1000 --batch_list 0-3` samples four batches with a target of 1,000 designs each.

## Selecting batches

Batch indices start at zero and must be less than `num_batches`. Supported forms include:

```bash
--batch_list all
--batch_list "0,2,5"
--batch_list "batch_0,batch_2"
--batch_list "0-3"
```

Ranges include both endpoints. Duplicate indices are processed once. The selection applies to every stage in the invocation.

For example, check two existing batches after filtering:

```bash
bash one_key_batch.sh --name test_batch --size 30 --batch_size 10 --batch_list "0,2" --op check
```

## Parallel execution

`--parallel_jobs` limits concurrent batches and is capped at the number of selected batches. For a full pipeline run, all sampling workers finish before later stages start. Each batch then runs `simulate -> filter -> check` in order, while different batches can run concurrently.

A failure stops the remaining stages for that batch; other batches continue. Batches that fail sampling are skipped in later stages. The script exits with status `1` if any selected batch fails.

Choose concurrency according to available CPU, memory, and GPU resources; the script does not assign a separate GPU to each worker.

## Pausing and resuming simulation

For an instance with a 24-hour limit, reserve time for workers to finish their current sample. For example, on a newly started instance:

```bash
bash one_key_batch.sh --name garmentcode_4k_dress --size 4000 --batch_size 1000 \
  --op simulate --batch_list all --seed 43 --body_type random_body --max_hours 23
```

`--max_hours` accepts decimal hours and counts from command startup, not instance startup. Subtract time already spent on the instance and leave a margin. The deadline is shared by all batches, including queued batches. Once it is reached, workers finish their current sample, save progress, and exit; queued batches are left for the next invocation. This is a soft deadline: a slow or stuck sample can delay exit, so do not set it to the instance's exact remaining lifetime.

Wait until the wrapper prints `Simulation paused safely. All workers have exited` and returns to the shell before stopping the instance. A deliberate pause returns exit code `75`, completion returns `0`, and a batch failure returns `1`. Instance shutdown and restart are manual.

After restarting the instance and activating the environment, run the same command again with a suitable new time budget. Keep the dataset directory on persistent storage, including each body's `dataset_properties_<body_type>.yaml`, sampled inputs, and simulation outputs. Completed attempts are skipped, and the single failure-retry pass resumes from its saved position. Do not rerun `sample` or delete progress files to resume.

Progress files are written to a temporary file in the same directory, flushed, and atomically replaced. An interruption during saving therefore preserves the previous complete checkpoint. This does not save an individual sample's intermediate simulation frames: if the instance is forcibly stopped mid-sample, that unfinished sample is rerun.

## Sampling seeds

The default global seed is `42`. Each batch receives a string seed:

```text
<seed>::batch_<batch_index>
```

For example, batch 3 uses `42::batch_3`. The seed does not depend on the dataset name, timestamp, or number of parallel workers.

Reproducing samples requires the same seed, batch index, body selection, design configuration, body data, and code/environment. Changing only `--name` does not produce fresh samples. Different seeds or batch indices change the random sequence, but do not guarantee that all generated samples are distinct.

## Output layout

After sampling three batches with the default body selection:

```text
test_batch/
  logs/
    batch_0_2_summary.log
    batch_0_2_terminal.log
    batch_0_2_timing.log
  batch_0/
    dataset_properties.yaml
    logs/
    default_body/
      data/
      patterns_vis/
    random_body/
      data/
      patterns_vis/
  batch_1/
    ...
  batch_2/
    ...
```

Selecting one body type creates only its directory. Later stages place outputs under the body type they process:

```text
<name>/batch_<index>/<body_type>/dataset_properties_<body_type>.yaml
<name>/batch_<index>/<body_type>/rand_<id>/
<name>/batch_<index>/<body_type>/filter_fail_report.yaml
<name>/batch_<index>/<body_type>/integrity_report.yaml
<name>/batch_<index>/<body_type>/renders/
```

The simulation script gathers its preview images into `renders/` when the batch finishes.

## Logs

Dataset logs are appended under `<name>/logs/`, using the minimum and maximum selected batch indices:

```text
batch_<min_index>_<max_index>_summary.log
batch_<min_index>_<max_index>_terminal.log
batch_<min_index>_<max_index>_timing.log
```

For example, `--batch_list "4-7"` uses the prefix `batch_4_7`. Selections with the same endpoints share these files, including different subsets such as `0,2` and `0-2`. Repeated invocations and different stages append to the same files when their endpoints match.

Batch logs are stored under:

```text
<name>/batch_<index>/logs/<run_id>_terminal.log
<name>/batch_<index>/logs/<run_id>_timing.log
```

`run_id` has the format `YYYYMMDD_HHMMSS`. Sampling failures may occur before batch logs are copied into place; the dataset logs contain those worker logs and the failed batch list. No separate `failed_batches.txt` is created.

## Existing datasets

When sampling with the default `--batch_list all`, the dataset directory must not already exist. This also applies when running the full pipeline.

An explicit batch list or range allows an existing dataset directory, but each selected target `batch_<index>` must be absent. Existing sampler output candidates named `batch_<index>_*` also block sampling for that batch. This rule depends on using an explicit list or range, even if that range includes every batch.

Later stages require an existing dataset directory and the relevant batch outputs:

- `simulate`: batch-level `dataset_properties.yaml` and `<body_type>/data/`.
- `filter` and `check`: `<body_type>/dataset_properties_<body_type>.yaml`.
- `check`: also requires `<body_type>/filter_fail_report.yaml`.

These checks do not create missing inputs or run earlier stages automatically.

## Filtering and integrity checks

`filter` invokes `filter_fail_sim.py`, writes `filter_fail_report.yaml`, and deletes failed cases and their associated files in the selected body directory. It combines recorded simulation failures with detected nipple coverage failures. Coverage coordinates currently exist only for `mean_all.obj` and `mean_all_apart.obj`; unsupported bodies produce coverage check errors, which alone do not mark a case as failed.

`check` invokes `check_integrity.py` with the target size for that batch and requires the filter report. It checks that:

- The original sample count matches the requested batch target.
- Remaining sample and simulation case IDs match the original cases minus those recorded as failed in the filter report.
- Remaining cases have all required nonempty files, including `*_sim.obj`, `*_sim.ply`, `*_boxmesh.obj`, `*_texture.png`, and `*_part_seg.json`.

The report is saved to `<name>/batch_<index>/<body_type>/integrity_report.yaml`. An integrity failure marks the batch as failed.
