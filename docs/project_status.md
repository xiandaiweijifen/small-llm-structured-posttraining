# Project Status

## Current Stage

The project is no longer in setup mode.

Current status:

- phase-1 data pipeline is complete
- core baselines have been run
- repair baseline has been implemented and evaluated
- reduced-schema ablation has been validated
- schema-generalization experiment has been run and evaluated
- Stage 2 data-regime, LoRA-rank, curriculum, epoch, and learning-rate ablations have been run and reviewed
- structure-then-semantics staged training has been run and reviewed
- Stage 3/4/5 hard-example continuation and targeted refinement branches have been run and reviewed
- Stage 6 action-canonicalization experiments have been run and reviewed
- Stage 7 component-canonicalization follow-ups have been run and reviewed
- Stage 8 deterministic postprocessing follow-ups have been run and reviewed

The project is currently in:

- **final result consolidation**
- **research-story finalization**

## Development History

### 1. Repository and Research Scoping

Completed:

- defined the project around small-model structured output post-training
- clarified the main question: what post-training solves vs what decoding/repair still must solve
- created modular repo structure with `src/`, `configs/`, `scripts/`, `docs/`, `notebooks/`, `results/`

### 2. Data Pipeline

Completed:

- fetched two external datasets
- mapped them into the project ticket schema
- added validation, profiling, and complexity relabeling
- built candidate data and stratified `train/val/test`
- prepared SFT-format data

Key outputs:

- `data/samples/phase1_train.jsonl`
- `data/samples/phase1_val.jsonl`
- `data/samples/phase1_test.jsonl`
- `data/processed/phase1_sft_train.jsonl`
- `data/processed/phase1_sft_val.jsonl`

### 3. Full-Schema Baselines

Completed:

- prompt-only baseline
- full-schema QLoRA baseline
- evaluation pipeline
- field-level error analysis

Main conclusion:

- prompt-only fails mainly on structure
- full-schema QLoRA fixes structure but still hallucinates field values

### 4. Reduced-Schema Ablation

Completed:

- built reduced schema removing noisy identity fields
- trained reduced-schema QLoRA baseline
- compared full vs reduced schema
- reran reduced-schema with larger H200-friendly batch settings

Main conclusion:

- noisy target fields materially hurt semantic learning
- reduced schema significantly improves end-to-end exact match

### 5. Repair Baseline

Completed:

- implemented schema-aware repair
- extended repair with conservative alias mapping and normalization
- evaluated prompt-only + repair
- evaluated QLoRA + repair

Main conclusion:

- repair strongly improves structural compliance for prompt-only outputs
- repair adds little once post-training already solves structure

### 6. Schema Generalization

Completed:

- added schema-conditioned prompt support
- added reduced unseen schema variant
- built seen/unseen schema evaluation datasets
- added grouped evaluation by `schema_seen_status`
- created dedicated schema-generalization notebook
- ran schema-conditioned reduced-schema QLoRA evaluation on combined seen/unseen schema test data

Key outputs:

- `data/generalization/phase1_test_seen_reduced.jsonl`
- `data/generalization/phase1_test_unseen_reduced.jsonl`
- `data/generalization/phase1_test_seen_unseen_reduced.jsonl`
- `data/processed/phase1_sft_train_reduced_schema_conditioned.jsonl`
- `data/processed/phase1_sft_val_reduced_schema_conditioned.jsonl`
- `notebooks/06_schema_generalization_qlora.ipynb`
- `results/metrics/qwen25_3b_schema_generalization_v1_test_report.json`
- `results/metrics/qwen25_3b_schema_generalization_v1_field_analysis.json`

### 7. Stage 2 Post-Training Ablations

Completed:

- ran a small-data reduced-schema training regime on 600 training samples
- ran LoRA rank ablations at 8, 16, and 32
- implemented and ran curriculum training from simple/medium buckets into full reduced-schema continuation
- ran rank-16 epoch ablations through epoch 9
- ran rank-16 learning-rate ablations at epoch 5
- implemented and ran structure-first then semantics-focused staged training
- ran Stage 3 hard-sample mining and continuation experiments on top of the strongest staged-training checkpoint
- ran Stage 4 targeted continuation experiments on narrower semantic subsets
- ran Stage 5 targeted refinement experiments around the best Stage 4 subset
- implemented and ran Stage 6 action-canonicalization experiments in both single-stage and staged forms
- implemented and ran Stage 7 component-canonicalization follow-ups, including joint `action + component` target redesign
- implemented and ran Stage 8 deterministic postprocessing variants on top of the Stage 7 best predictions
- exported a consolidated Stage 2 review summary
- exported a consolidated long-run batch summary
- exported a Stage 3 end-to-end optimization batch summary
- exported a Stage 6 action-canonicalization batch summary
- exported a Stage 7 component-canonicalization batch summary
- exported a Stage 8 deterministic postprocessing batch summary

Key outputs:

- `results/metrics/qwen25_3b_stage2_data_small_600_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank8_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank32_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_curriculum_sm_then_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_epoch9_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_lr2e4_epoch5_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_structure_then_semantics_v1_test_report.json`
- `docs/results/stage2_results_review.md`
- `docs/results/long_run_ablation_batch_summary.md`
- `results/metrics/qwen25_3b_stage3_sts_v2_full_plus_hard_x2_epoch2_lr1e4_test_report.json`
- `docs/results/end_to_end_optimization_batch_summary.md`
- `results/metrics/qwen25_3b_stage6_canonical_action_single_stage_epoch7_lr2e4_test_report.json`
- `docs/results/action_canonicalization_batch_summary.md`
- `results/metrics/qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9_test_report.json`
- `docs/results/component_canonicalization_batch_summary.md`
- `results/metrics/qwen25_3b_stage8_action_component_majority_test_report.json`
- `docs/results/deterministic_postprocess_batch_summary.md`

Main conclusions:

- small reduced-schema datasets are enough to preserve structure, but not enough to learn the hardest semantic fields
- rank 8 underfits, rank 16 is competitive, and rank 32 gives a modest additional gain
- epoch helps strongly up to about epoch 5, then mostly saturates
- `2e-4` is the best learning-rate balance among the tested single-stage runs
- broad hard-sample continuation and targeted refinement branches do not beat the strongest staged baseline
- action canonicalization is the first post-Stage-2 change that materially raises the end-to-end ceiling
- component canonicalization alone is weak, but joint `action + component` canonicalization becomes effective when paired with staged training
- deterministic postprocessing on top of the Stage 7 best run yields a further no-train gain
- the current best run is Stage 8 `action + component` deterministic postprocessing on top of the Stage 7 best model
- hard-sample continuation identifies a real semantic hard subset, but the current continuation recipes do not beat the strongest staged baseline
- repair still adds no measurable value once post-training has already stabilized structure

## Current Experimental Findings

### Prompt-only

- weak schema compliance
- often outputs partial or wrong-format JSON

### Prompt-only + Repair

- structure becomes mostly valid
- semantic extraction is still weak

### Full-Schema QLoRA

- structure is stable
- remaining bottleneck is value hallucination

### Reduced-Schema QLoRA

- best phase-1 result so far
- confirms the importance of target design and label quality

### Reduced-Schema QLoRA H200-Fast

- small improvement over the first reduced baseline
- useful as a stronger reproduction run, not a new qualitative breakthrough

### Schema-Conditioned Reduced QLoRA Generalization

- overall field exact match: `0.8764`
- overall end-to-end exact match: `0.4646`
- seen-schema field exact match: `0.8837`
- unseen-schema field exact match: `0.8691`
- seen-schema end-to-end exact match: `0.4764`
- unseen-schema end-to-end exact match: `0.4528`

Main conclusion:

- mild schema shift hurts semantics more than structure
- schema-conditioned post-training generalizes reasonably well to unseen schema variants
- remaining failures are concentrated in semantic fields such as `action`, `component`, `category`, and `priority`

### Stage 2 Reduced-Schema QLoRA, 600 Samples

- valid JSON and schema compliance stay perfect
- semantic performance drops sharply
- confirms that data scale and coverage are critical for post-training quality

### Stage 2 Reduced-Schema QLoRA Rank Ablation

- rank 8 is clearly weaker
- rank 16 recovers the earlier reduced baseline
- rank 32 becomes the strongest non-curriculum Stage 2 run

### Stage 2 Reduced-Schema Curriculum Training

- overall field exact match: `0.9037`
- overall end-to-end exact match: `0.5315`

Main conclusion:

- curriculum training is a strong improvement over one-shot reduced-schema training
- the gain is concentrated in semantic fields rather than structural repair

### Stage 2 Epoch and Learning-Rate Follow-Ups

- epoch 5 reaches `0.9145 / 0.5709`
- epoch 9 reaches `0.9166 / 0.5709`
- learning rate `1e-4` reaches `0.8901 / 0.4882`
- learning rate `2e-4` reaches `0.9141 / 0.5709`
- learning rate `4e-4` reaches `0.9173 / 0.5591`

Main conclusion:

- training duration matters mainly because it continues to improve semantic fields after structure is already solved
- learning rate changes semantic convergence quality more than structural stability
- the best single-stage balance in this set remains rank 16, epoch 5, learning rate `2e-4`

### Stage 2 Structure-Then-Semantics Training

- overall field exact match: `0.9245`
- overall end-to-end exact match: `0.5787`

Main conclusion:

- staged training now produces the strongest pre-canonicalization result in the repository
- explicitly separating structure-focused and semantics-focused phases improves the hardest semantic fields further

### Stage 3 Hard-Sample Continuation

- hard mining finds `561 / 1993` train samples with errors on the key semantic fields
- best continuation run: `0.9155 / 0.5433`
- strongest pre-canonicalization baseline remains `0.9245 / 0.5787`

Main conclusion:

- hard-sample mining confirms that the remaining bottleneck is concentrated in a real subset of difficult semantic examples
- however, direct hard-only continuation and heavy hard oversampling both hurt overall quality
- even the best mixed continuation remains below the strongest staged-training baseline
- the current hard-continuation recipe is therefore a useful negative result, not a new best model

### Stage 6 Action Canonicalization

- best run: `qwen25_3b_stage6_canonical_action_single_stage_epoch7_lr2e4`
- overall field exact match: `0.9341`
- overall end-to-end exact match: `0.6654`

Main conclusion:

- canonicalizing `actions_requested[0].action` is the first change that clearly breaks through the Stage 2 performance ceiling
- the main gain comes from target redesign rather than additional continuation or repair
- under the canonicalized target, a simpler single-stage run is slightly stronger than the more complex staged variants
- this result should be interpreted as a target-design improvement, not just a training-hyperparameter gain

### Stage 7 Component Canonicalization Follow-Up

- best run: `qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
- overall field exact match: `0.9402`
- overall end-to-end exact match: `0.6772`

Main conclusion:

- `component` canonicalization alone is not a strong direction
- however, joint `action + component` canonicalization does improve the best result when it is paired with staged training
- the main Stage 7 gain comes from a large improvement on `affected_systems[0].component`, which compensates for a slight drop in `action`
- this becomes the strongest trained run in the repository before any deterministic postprocessing

### Stage 8 Deterministic Postprocessing

- best run: `qwen25_3b_stage8_action_component_majority`
- overall field exact match: `0.9427`
- overall end-to-end exact match: `0.6929`

Main conclusion:

- a deterministic consistency pass can still extract additional gains after training saturates
- the useful rule is simple: refresh canonical `action` from predicted `category + summary`, and map `component` from predicted `name` using the train-set majority mapping
- almost all of the Stage 8 gain comes from the `component <- name` consistency rule; `action` refresh improves field exact match slightly but does not change end-to-end exact match by itself
- this is now the current strongest run in the repository

## What Is Still Missing

To reach the originally desired "more complete research project" level, the project still mainly needs:

- final README / summary cleanup for resume and interview use
- optionally, a cleaner constrained-decoding baseline if a better schema-compatible tool is used
- optionally, further target redesign or two-stage supervision for `category` and `priority`
- optionally, tighter deterministic handling for `category` and `priority` if a high-precision rule set can be found

## Current Next Step

Immediate next step:

- finalize top-level project narrative around the Stage 7 joint canonicalization result
- finalize top-level project narrative around the Stage 8 no-train postprocessing gain
- keep additional experimentation narrow unless it clearly improves the final research story

Expected outcome:

- one stable top-level summary of prompt-only, repair, reduced-schema target design, LoRA-rank ablations, epoch and learning-rate ablations, staged training, hard-example negative results, action canonicalization, component follow-ups, deterministic postprocessing, and seen/unseen schema generalization
- one clear statement that broad continuation did not beat the strongest staged baseline, while joint target redesign plus staged training did
- one clear statement of what post-training solves, what repair solves, what deterministic consistency can still clean up cheaply, and what still fails semantically

## Practical Rule

Current development priority:

- prioritize final narrative quality over adding broad new experiment branches
- do not spend major effort on broad hyperparameter tuning unless it directly supports a stronger final conclusion
- keep reusable logic in `src/` and `scripts/`, not notebooks
