# structured-output-small-llm

Research-oriented small LLM post-training project for complex schema-based structured outputs.

## At A Glance

- Task: complex text-to-JSON extraction under a moderately nested schema
- Core question: what small-model post-training can solve on its own, and where repair / decoding / postprocess still matter
- Strongest trained run:
  - `qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
  - `field_exact_match = 0.9402`
  - `end_to_end_exact_match = 0.6772`
- Strongest overall system run:
  - `qwen25_3b_stage9_lexical_combined`
  - `field_exact_match = 0.9470`
  - `end_to_end_exact_match = 0.7205`
- Strongest mapped external run:
  - `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
  - `field_exact_match = 0.7517`
  - `end_to_end_exact_match = 0.0636`

## Best Results

| Result Type | Run | Field Exact Match | End-to-End Exact Match | Meaning |
| --- | --- | ---: | ---: | --- |
| Best trained in-domain | `stage7_canonical_action_component_staged` | 0.9402 | 0.6772 | strongest model ability before no-train cleanup |
| Best system in-domain | `stage9_lexical_combined` | 0.9470 | 0.7205 | strongest end-to-end pipeline including deterministic and lexical cleanup |
| Best mapped external | `stage14_target_allcore_x1_epoch1_lr5e5` | 0.7517 | 0.0636 | strongest external adaptation result after completeness recovery |

## Key Findings

- Prompt-only mainly fails on required-field completeness and schema compliance.
- Repair is strong for prompt-only structure, but adds almost nothing once post-training already stabilizes output format.
- Reduced schema and target redesign matter more than raw model scaling for this task.
- `action` canonicalization is the first intervention that clearly breaks the Stage 2 end-to-end ceiling.
- `component` canonicalization only becomes useful when paired with staged training and the already-validated `action` redesign.
- Deterministic and lexical postprocess still add meaningful in-domain gains, but those gains do not transfer cleanly to the external mapped dataset.
- Same-family larger prompt-only models (`7B / 14B / 32B`) remain far below the post-trained 3B pipeline on exact schema-based evaluation.
- Modern decode-side control with vLLM structured outputs can force perfect schema compliance for prompt-only references, but still does not recover usable semantic exact match.
- External few-shot adaptation is a real positive result: it restores completeness and lifts mapped external end-to-end exact match above zero.
- External optimization now shows a clear plateau: after completeness is solved, the remaining bottleneck is semantic taxonomy alignment rather than structure.
- A narrow external `component` verifier does not break that plateau, which suggests the remaining external error is not solvable by simple component-only postprocess.

## Experiment Evolution

1. Prompt-only and repair established that the initial failure mode was structural incompleteness.
2. Reduced-schema post-training established that target design unlocks semantic learning.
3. Stage 2 ablations showed that rank, epoch, and learning rate matter, but do not match the gains from better target design and training structure.
4. Structure-then-semantics staged training became the strongest pre-canonicalization baseline.
5. Broad hard-sample continuation failed to beat that staged baseline, establishing a useful negative result.
6. `action` canonicalization created the first major breakthrough.
7. Joint `action + component` canonicalization plus staged training produced the strongest trained run.
8. Deterministic and lexical postprocess layers produced the strongest full system.
9. External zero-shot evaluation showed that trained small models still beat larger prompt-only models at field level, but completeness collapsed.
10. External few-shot adaptation restored completeness, and Stage 14 targeted continuation produced the best external result before the line plateaued.
11. A Stage 18 component-verifier follow-up confirmed that simple field-specific postprocess is still too weak to beat the Stage 14 external best.

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
- Stage 10 latent-action and Stage 11 semantic-core intermediate explorations
- Stage 12 semantic-slot auxiliary supervision experiments
- Stage 13 external few-shot adaptation experiments from the strongest Stage 7 checkpoint
- Stage 14 external targeted adaptation experiments on top of the strongest Stage 13 checkpoint
- Stage 16 external frontier experiments: larger-scale adaptation and retrieval-guided taxonomy postprocess
- Stage 17 external deeper-direction experiments: field-level target redesign and residual curriculum
- Stage 18 external component-verifier follow-ups on top of the strongest Stage 14 predictions
- Stage 19 vLLM structured-output reevaluation of same-family prompt-only references
- multi-model same-family prompt-only reference comparisons at `3B / 7B / 14B / 32B`
- external-dataset generalization references on a mapped customer-support ticket dataset
- seen/unseen schema generalization results

For Stage 8 and Stage 9:

- the reusable experiment logic lives in `scripts/`
- the notebooks are thin launchers for Jupyter use
- the local verification runs were executed through the Python scripts directly, not through the Jupyter UI

Current strongest run:

- Stage 9 combined lexical postprocessing on top of the Stage 8 best run
- field exact match: `0.9470`
- end-to-end exact match: `0.7205`

Current strongest trained run:

- Stage 7 joint canonical action+component redesign with staged training
- field exact match: `0.9402`
- end-to-end exact match: `0.6772`

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
- more algorithmic exploration lines were also tested: latent-action targets, semantic-core intermediates, and semantic-slot auxiliary supervision
- among those, Stage 12 semantic-slot supervision is the strongest exploration branch, but it still does not beat the Stage 7 canonicalized staged-training line
- same-family larger prompt-only models (`7B / 14B / 32B`) still fail to satisfy the target schema reliably, so raw scaling does not replace post-training and target redesign
- on a mapped out-of-domain customer-support eval set, the strongest trained 3B still beats `14B/32B` prompt-only references at field level, but schema completeness collapses and the Stage 8/9 postprocess layers do not transfer
- a heavier external few-shot adaptation branch from the Stage 7 checkpoint is effective: with enough external supervision, schema completeness recovers and external end-to-end exact match becomes non-zero again
- the best external adaptation run reaches `field_exact_match = 0.7512` and `end_to_end_exact_match = 0.0536`, showing that external adaptation mainly fixes completeness first, while taxonomy-level semantics remain the next bottleneck
- a follow-up external targeted adaptation branch improves the best external result further to `field_exact_match = 0.7517` and `end_to_end_exact_match = 0.0636`
- under the current external targeted recipe, a light low-learning-rate continuation over all core taxonomy fields works better than narrower field subsets or a higher learning-rate variant
- after Stage 14, the external line enters a semantic-taxonomy plateau: larger-scale adaptation, retrieval-guided postprocess, field-level target redesign, and residual curriculum do not beat the Stage 14 best
- a narrow Stage 18 component-verifier branch also fails to beat the Stage 14 external best, which reinforces that the remaining external issue is not a simple `component` postprocess problem
- a Stage 19 vLLM structured-output reevaluation shows that modern constrained decoding solves prompt-only structure cleanly, but still does not solve semantic correctness
- the strongest Stage 16/17 variants preserve perfect external schema completeness, which confirms that the remaining external bottleneck is no longer structure but `component / priority / category / action` alignment
- the internal/external mismatch audit shows that this external bottleneck is driven less by raw OOD label vocabulary and more by changed conditional mappings, especially weak external `summary -> category` and `name -> component` purity
- under mild schema shift, structure generalizes better than semantics

Recommended entry points for the current project state:

- [project_status.md](d:/project/small-llm-structured-posttraining/docs/project_status.md)
- [phase1_baseline_findings.md](d:/project/small-llm-structured-posttraining/docs/results/phase1_baseline_findings.md)
- [final_results_summary.md](d:/project/small-llm-structured-posttraining/docs/results/final_results_summary.md)
- [error_analysis_taxonomy.md](d:/project/small-llm-structured-posttraining/docs/results/error_analysis_taxonomy.md)
- [internal_external_mismatch_audit.md](d:/project/small-llm-structured-posttraining/docs/results/internal_external_mismatch_audit.md)
- [stage2_results_review.md](d:/project/small-llm-structured-posttraining/docs/results/stage2_results_review.md)
- [long_run_ablation_batch_summary.md](d:/project/small-llm-structured-posttraining/docs/results/long_run_ablation_batch_summary.md)
