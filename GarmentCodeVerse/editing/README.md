# Semantic Garment Editing

`editing/` adds deterministic single-property edits to GarmentCodeRC. It uses
the original GarmentCode constructors, simulator, physical filter, and
integrity checker without modifying them.

The supported workflow is:

```text
plan -> generate -> simulate -> filter -> project -> status
```

Each accepted source/target pair keeps the same body, changes exactly one
semantic property, and contains a replayable instruction. Target prompts are
created only after physical simulation passes.

## Installation and dependencies

Copy `editing/` into the original `GarmentCodeRC/` directory:

```text
Reweaver2-Code-upload/
├── GarmentCodeRC/
│   ├── editing/
│   ├── assets/
│   ├── pygarment/
│   ├── pattern_data_sim.py
│   ├── filter_fail_sim.py
│   └── check_integrity.py
└── gen_garment_texture/
    └── build_prompt_split_match/build_prompts.py
```

Use the Python environment and machine-local `system.json` required by the
original GarmentCodeRC project. Run commands from the `GarmentCodeRC/`
directory. The only supported editing configuration is
`editing/configs/semantic_edit.yaml`; old run formats are not migrated, so use
new analysis and output directories when starting a run.

The prompt builder is an approved read-only input. `plan` uses it to classify
source prompt states, and `project` uses it to generate prompts for candidates
that pass simulation. `generate` does not load it.

## Input layout

`INPUT_ROOT` contains one or more GarmentCode batches:

```text
<input-root>/batch_N/random_body/data/<source-id>/
├── design_params.yaml
├── body_measurements.yaml
└── <source-id>_specification.json
```

Completed GarmentCode batches with the flattened simulator layout are also
accepted directly:

```text
<input-root>/batch_N/random_body/<source-id>/
├── <source-id>_design_params.yaml
├── <source-id>_body_measurements.yaml
└── <source-id>_specification.json
```

When both layouts exist for a batch, the canonical `random_body/data/` layout
takes precedence. Within a sample directory, each design/body YAML may use its
canonical or sample-prefixed name, but duplicate aliases are rejected.

Set the run paths once and keep the same input, analysis, output, and seed
throughout the run:

```bash
conda activate GCD

INPUT_ROOT=/path/to/source-dataset
ANALYSIS_ROOT=/path/to/new-analysis-directory
OUTPUT_ROOT=/path/to/new-editing-output
BATCH_ID=batch_N
PROMPT_BUILDER=/path/to/Reweaver2-Code-upload/gen_garment_texture/build_prompt_split_match/build_prompts.py
```

`ANALYSIS_ROOT` must not exist before `plan`. `OUTPUT_ROOT` must not exist
before the first `generate`; later `generate` calls resume that same run.

## 1. Plan

```bash
python -B -m editing.semantic_edit plan \
  --input-root "$INPUT_ROOT" --batch-id "$BATCH_ID" \
  --analysis-root "$ANALYSIS_ROOT" \
  --config editing/configs/semantic_edit.yaml \
  --prompt-builder "$PROMPT_BUILDER" --seed 42
```

Planning inspects each source with its fixed body, enumerates legal
single-operation candidates, balances the enabled rule distribution, and
writes `$ANALYSIS_ROOT/pilot_eligibility_report.json`.

## 2. Generate

```bash
python -B -m editing.semantic_edit generate \
  --input-root "$INPUT_ROOT" --analysis-root "$ANALYSIS_ROOT" \
  --output-root "$OUTPUT_ROOT" --seed 42
```

Generation selects one legal edit per unresolved source, invokes the original
GarmentCode constructor, validates the rule effect, renders the pattern, and
creates `$OUTPUT_ROOT/.state/simulation_workspace`. Each pending candidate has
five files: design YAML, unchanged body YAML, specification JSON, pattern PNG,
and pattern SVG. Generation failures try another legal edit and do not consume
a physical attempt.

Before validation and rendering, generation reconstructs the source from its
saved YAML once per source per round. Panels unchanged between this baseline
and the edited reconstruction are copied in full from the original source
specification, including vertices, curved edges, rotation, and translation.
Real edit dependencies (including placement changes) and added/removed panels
are retained. Incompatible source edge layouts are rejected so target stitch
indices cannot silently refer to different edges. The final specification is
validated again and is used by rendering, simulation, and publication.

This removes rounding drift in unaffected panels without changing the input
dataset. Target YAML alone does not reproduce the preserved source geometry;
replay also requires the paired source specification and this preservation
step. Metadata records the preserved panel names. Start a new output directory
for this pipeline; runs without its `panel_geometry_policy` are rejected.

Do not call `generate` again while an active simulation workspace exists.

## 3. Simulate and filter

Choose suitable GPUs for the machine, then run the original GarmentCode
scripts:

```bash
python -B pattern_data_sim.py \
  --data "$OUTPUT_ROOT/.state/simulation_workspace" \
  --config beautiful_sim_props.yaml

python -B filter_fail_sim.py \
  --gcd_dir "$OUTPUT_ROOT/.state/simulation_workspace" \
  --body_type random_body \
  --no-delete
```

Wait until simulation is frozen. `--no-delete` is required because `project`
reads the complete filter report and native simulation products.

## 4. Project

```bash
python -B -m editing.semantic_edit project \
  --output-root "$OUTPUT_ROOT" --prompt-builder "$PROMPT_BUILDER"
```

`project` consumes the frozen GarmentCode filter report. Passing candidates
receive their final prompt and metadata and are published once. Failed
candidates never receive a target prompt. If a round contains no passing
candidate, the prompt builder is not loaded.

## 5. Status and retries

```bash
python -B -m editing.semantic_edit status --output-root "$OUTPUT_ROOT"
```

`status` is read-only. While unresolved sources remain and no active workspace
exists, repeat:

```text
generate -> simulate -> filter --no-delete -> project -> status
```

After `simulation_failed` or `prompt_failed`, the next `generate` chooses a
different legal edit for the same source and body. Both outcomes count toward
the maximum of three physical attempts per source. A construction or local
generation failure does not. A source becomes terminal when it is accepted,
has no unused legal edit, or reaches the physical-attempt limit.

## Output layout

Every accepted target has exactly seven files in the flat data inventory:

```text
<output-root>/batch_N/random_body/data/<target-id>/
├── design_params.yaml
├── body_measurements.yaml
├── <target-id>_specification.json
├── <target-id>_pattern.png
├── <target-id>_pattern.svg
├── <target-id>_prompts.json
└── <target-id>_edit_metadata.json
```

The body YAML is copied unchanged from the paired source. Native GarmentCode
simulation products are published at
`batch_N/random_body/<target-id>/`. The batch also contains
`accepted_pairs.jsonl` and dataset properties. Active work, rejected
candidates, rejected simulation products, slot state, and archived filter
reports remain under `<output-root>/.state/`.

`accepted_pairs.jsonl` records the source reference, same-body pairing,
operation, instruction and replay provenance, semantic effect, simulation
decision, and target hashes.

## Final validation

After every source is terminal and no active workspace remains, run the
original final checks on the completed batch:

```bash
python -B filter_fail_sim.py \
  --gcd_dir "$OUTPUT_ROOT/$BATCH_ID" \
  --body_type random_body \
  --no-delete

python -B check_integrity.py \
  --gcd_dir "$OUTPUT_ROOT/$BATCH_ID" \
  --body_type random_body \
  --require_filter_report
```

Both commands must pass before publishing the dataset.

## Project structure

The production modules are organized by responsibility:

| Path | Responsibility |
| --- | --- |
| `models.py`, `config.py` | In-memory contracts and the supported rule configuration. |
| `canonical_io.py` | Canonical YAML/JSON I/O, dotted design paths, and identity hashes. |
| `pattern_geometry.py`, `pattern_compare.py`, `pattern_renderer.py` | Shared pattern measurement, semantic comparison, and rendering. |
| `publication.py`, `simulation_workspace.py` | Atomic publication and the active GarmentCode workspace. |
| `semantic_edit/planning.py` | Source analysis, legal capabilities, deterministic assignment, and candidate selection. |
| `semantic_edit/generation.py` | Run initialization, construction, rule validation, rendering, and five-file workspaces. |
| `semantic_edit/projection.py` | Filter projection, post-pass prompts, failure recording, and seven-file publication. |
| `semantic_edit/construction.py` | GarmentCode construction and preservation of unaffected source panels. |
| `semantic_edit/prompt.py`, `prompt_state.py` | Approved prompt-builder execution, active constructor paths, and shared prompt/neckline states. |
| `semantic_edit/registry.py`, `rules/` | Ordered rule catalog; each rule owns applicability, mutation, effect validation, and evidence. |
| `semantic_edit/instructions.py` | Deterministic instruction rendering and replay. |
| `semantic_edit/store.py` | Persisted run state and transitions. |
| `semantic_edit/pipeline.py`, `__main__.py` | Public Python exports and the four-command CLI. |

To tune an existing rule, edit only its values in
`editing/configs/semantic_edit.yaml` and run the affected behavior tests. To
add a rule id, add its config, self-contained rule implementation, registry
entry, instruction templates, and parameterized rule plus pipeline tests.
Extend `prompt_state.py` only when the new rule needs a shared prompt-derived
state.

## Tests

Run the complete editing suite from the GarmentCodeRC root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -p no:cacheprovider -q editing/tests

python -m compileall -q editing
find editing -type d -name __pycache__ -prune -exec rm -rf {} +
git diff --check
```
