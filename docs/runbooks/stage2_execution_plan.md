# Stage 2 Execution Plan

## Goal

Stage 2 should strengthen the repo as a post-training project instead of a phase-1 proof of concept.

The immediate priority is not to add more framework code. The immediate priority is to add a small, defensible experiment matrix on top of the existing reduced-schema QLoRA baseline.

## Current Position

Already completed in Stage 1:

- prompt-only baseline
- prompt-only plus repair
- full-schema QLoRA baseline
- reduced-schema QLoRA baseline
- reduced-schema H200-fast rerun
- schema-conditioned seen/unseen schema generalization
- field-level analysis and written findings

What is still missing for a stronger post-training story:

- controlled data-regime comparison
- controlled training-strategy comparison
- controlled LoRA-capacity ablation
- one clean review pass over all new results

## Recommended Stage 2 Order

Run the next experiments in this order.

### Batch A: Data Regime

Purpose:

- show whether a smaller but cleaner training regime can compete with the current full reduced-schema training set
- create the first direct data-strategy result without needing new external data

Notebook:

- `notebooks/07_stage2_experiment_runner.ipynb`

Preset to run:

- `data_small_600`

Expected outputs:

- `results/predictions/qwen25_3b_stage2_data_small_600_test.jsonl`
- `results/metrics/qwen25_3b_stage2_data_small_600_test_report.json`
- `results/metrics/qwen25_3b_stage2_data_small_600_field_analysis.json`
- repaired counterparts

Success criterion:

- the run finishes cleanly and gives a stable comparison against the existing reduced baseline

### Batch B: Curriculum

Purpose:

- test whether starting with easier records before full-data continuation helps semantic fields
- make the training-strategy section concrete

Notebook:

- `notebooks/07_stage2_experiment_runner.ipynb`

Preset to run:

- `curriculum_simple_medium_then_full`

Expected outputs:

- `results/predictions/qwen25_3b_stage2_curriculum_sm_then_full_test.jsonl`
- `results/metrics/qwen25_3b_stage2_curriculum_sm_then_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_curriculum_sm_then_full_field_analysis.json`
- repaired counterparts

Success criterion:

- compare against the reduced baseline and judge whether curriculum changes semantic-field accuracy, especially `action`, `component`, `category`, and `priority`

### Batch C: LoRA Capacity

Purpose:

- show you actually explored the fine-tuning design space
- keep the ablation narrow and tied to the structured-output task

Notebook:

- `notebooks/07_stage2_experiment_runner.ipynb`

Presets to run:

- `rank8_full`
- `rank16_full`
- `rank32_full`

Expected outputs:

- one prediction file per rank
- one report file per rank
- one field-analysis file per rank
- repaired counterparts

Success criterion:

- answer whether higher LoRA capacity helps the hardest semantic fields more than it helps already-stable structural fields

### Batch D: Review and Consolidation

Purpose:

- merge Stage 1 and Stage 2 into one result table
- decide whether constrained decoding is still worth adding next

Notebook:

- `notebooks/08_stage2_results_review.ipynb`

What to check:

- best raw run by end-to-end exact match
- best raw run by field exact match
- whether repair still adds near-zero value after post-training
- whether curriculum or rank changes the worst fields

### Batch E: Minimal Fine-Tuning Ablation

Purpose:

- add one compact, question-driven fine-tuning study without turning the project into a broad parameter sweep
- test whether additional training epochs mainly help the hardest semantic fields or only add redundant fitting

Notebook:

- `notebooks/07_stage2_experiment_runner.ipynb`

Presets to run:

- `epoch2_rank16_full`
- `epoch3_rank16_full`
- `epoch5_rank16_full`

Why this setup:

- fixes LoRA capacity at rank 16
- fixes learning rate and batch settings
- isolates the effect of training duration

Expected outputs:

- one prediction file per epoch setting
- one report file per epoch setting
- one field-analysis file per epoch setting
- repaired counterparts

Success criterion:

- determine whether longer training mostly improves `action`, `component`, `category`, and `priority`, or whether gains saturate after the current default

## What Not To Do Yet

Do not add large new branches before Stage 2 review is complete.

Defer these until after the Stage 2 results come back:

- broad hyperparameter sweeps
- multiple base models
- constrained decoding implementation
- synthetic-data generation pipeline

These may still be useful later, but they are not the shortest path to a stronger project story right now.

## Files To Sync Back After Online Runs

Please sync back these files after each completed preset:

- `results/predictions/<experiment_name>_test.jsonl`
- `results/predictions/<experiment_name>_test_repaired.jsonl`
- `results/metrics/<experiment_name>_test_report.json`
- `results/metrics/<experiment_name>_test_repaired_report.json`
- `results/metrics/<experiment_name>_field_analysis.json`
- `results/metrics/<experiment_name>_test_repaired_field_analysis.json`

If available, also sync:

- `results/checkpoints/<experiment_name>/trainer_state.json`
- `results/checkpoints/<experiment_name>/training_args.bin`

## How To Continue After Sync

Once the new files are back in the repo:

1. review all Stage 2 reports in `08_stage2_results_review.ipynb`
2. update the findings markdown
3. decide whether the next step is constrained decoding or final narrative consolidation
