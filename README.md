# structured-output-small-llm

Research-oriented small LLM post-training project for complex schema-based structured outputs.

## Project Goal

This repository studies a focused question:

How far can small models go on complex text-to-JSON structured output tasks with post-training alone, and what is the remaining boundary between training-time gains, decoding-time constraints, and repair?

The project is designed to demonstrate:

- SFT / LoRA / QLoRA post-training ability
- data construction, target design, and sampling strategy design
- training strategy and LoRA-capacity ablations
- structured-output evaluation and error analysis
- light but reusable engineering instead of notebook-only experiments

## Phase 1 Scope

Phase 1 focuses on one primary task:

- input: natural language text
- output: JSON object under a moderately complex schema
- emphasis: post-training first, decoding enhancement second

Recommended first task:

- email / ticket / task-description understanding to structured JSON

Why this task:

- realistic enterprise-style structured extraction
- easy to define nested and partially optional schemas
- supports both seen-schema and unseen-schema analysis later

## Main Experiment Line

1. Prompt-only baseline
2. Full-schema QLoRA baseline
3. Reduced-schema QLoRA baseline
4. Prompt-only / post-training with schema-aware validation / repair
5. Schema-conditioned seen/unseen generalization
6. Stage 2 post-training ablations:
   - small-data reduced-schema training
   - LoRA rank 8 / 16 / 32
   - curriculum simple/medium then full training
   - epoch-duration ablations at fixed rank 16
   - learning-rate ablations at fixed rank 16, epoch 5
   - structure-first then semantics-focused two-stage training
7. Error decomposition:
   - JSON format errors
   - schema compliance errors
   - field-level semantic errors

## Repository Layout

```text
structured-output-small-llm/
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- src/
|   |-- common/
|   |-- data/
|   |-- training/
|   |-- inference/
|   |-- evaluation/
|   |-- schemas/
|   `-- utils/
|-- configs/
|   |-- dataset/
|   |-- train/
|   `-- eval/
|-- scripts/
|-- notebooks/
|-- results/
|   |-- metrics/
|   `-- predictions/
`-- docs/
    `-- design/
```

## Key Documents

- [project_brief.md](d:/project/small-llm-structured-posttraining/docs/project_brief.md)
- [phase1_task.md](d:/project/small-llm-structured-posttraining/docs/design/phase1_task.md)
- [experiment_matrix.md](d:/project/small-llm-structured-posttraining/docs/design/experiment_matrix.md)
- [schema_variants.md](d:/project/small-llm-structured-posttraining/docs/design/schema_variants.md)
- [model_selection.md](d:/project/small-llm-structured-posttraining/docs/design/model_selection.md)
- [training_runbook.md](d:/project/small-llm-structured-posttraining/docs/runbooks/training_runbook.md)

## Development Principle

Keep reusable logic in `src/` and `scripts/`.

Use `notebooks/` only for:

- data exploration
- visualization
- error analysis
- result presentation

## Current Status

The repository currently contains:

- full phase-1 data pipeline
- prompt-only, repair, and QLoRA baselines
- reduced-schema ablation and H200-fast rerun
- Stage 2 data-regime, LoRA-rank, curriculum, epoch, and learning-rate ablations
- structure-first then semantics-focused two-stage training
- Stage 3/4/5 hard-example continuation and targeted refinement branches
- Stage 6 action-canonicalization experiments across single-stage and staged settings
- Stage 7 component-canonicalization follow-ups, including joint action+component target redesign
- Stage 8 deterministic postprocessing follow-ups on the Stage 7 best predictions
- Stage 9 lexical postprocessing follow-ups on top of the Stage 8 best predictions
- seen/unseen schema generalization results

For Stage 8 and Stage 9:

- the reusable experiment logic lives in `scripts/`
- the notebooks are thin launchers for Jupyter use
- the local verification runs were executed through the Python scripts directly, not through the Jupyter UI

Current strongest run:

- Stage 9 combined lexical postprocessing on top of the Stage 8 best run
- field exact match: `0.9470`
- end-to-end exact match: `0.7205`

Current high-level conclusions:

- prompt-only mainly fails on structure
- repair strongly helps prompt-only structure, but adds essentially no value once post-training already stabilizes output format
- reduced target design materially improves semantic learning
- epoch duration and learning rate both matter, but their gains saturate and are smaller than the gains from target design plus stronger training strategy
- broad hard-sample continuation and refinement do not beat the strongest staged baseline
- canonicalizing the `action` target is the first change that materially lifts end-to-end exact match beyond the Stage 2 ceiling
- `component` canonicalization alone is weak, but joint `action + component` canonicalization becomes effective when paired with staged training
- a final deterministic consistency pass on top of the strongest Stage 7 run lifts end-to-end exact match further without retraining
- a final high-precision lexical postprocess layer lifts `priority` and `blocking` further, pushing end-to-end exact match above `0.72`
- under mild schema shift, structure generalizes better than semantics

Recommended entry points for the current project state:

- [project_status.md](d:/project/small-llm-structured-posttraining/docs/project_status.md)
- [phase1_baseline_findings.md](d:/project/small-llm-structured-posttraining/docs/results/phase1_baseline_findings.md)
- [final_results_summary.md](d:/project/small-llm-structured-posttraining/docs/results/final_results_summary.md)
- [stage2_results_review.md](d:/project/small-llm-structured-posttraining/docs/results/stage2_results_review.md)
- [long_run_ablation_batch_summary.md](d:/project/small-llm-structured-posttraining/docs/results/long_run_ablation_batch_summary.md)
