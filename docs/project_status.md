# Project Status

## Current Stage

The project is no longer in setup mode.

Current status:

- phase-1 data pipeline is complete
- core baselines have been run
- repair baseline has been implemented and evaluated
- reduced-schema ablation has been validated
- schema-generalization experiment has been run and evaluated
- Stage 2 data-regime, LoRA-rank, and curriculum ablations have been run and reviewed

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
- exported a consolidated Stage 2 review summary

Key outputs:

- `results/metrics/qwen25_3b_stage2_data_small_600_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank8_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank32_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_curriculum_sm_then_full_test_report.json`
- `docs/results/stage2_results_review.md`

Main conclusions:

- small reduced-schema datasets are enough to preserve structure, but not enough to learn the hardest semantic fields
- rank 8 underfits, rank 16 is competitive, and rank 32 gives a modest additional gain
- curriculum training is the best run in the project so far
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

- curriculum training produces the strongest result in the repository
- the gain is concentrated in semantic fields rather than structural repair

## What Is Still Missing

To reach the originally desired "more complete research project" level, the project still mainly needs:

- optionally, a decoding-side constrained generation baseline beyond repair
- final README / summary cleanup for resume and interview use

## Current Next Step

Immediate next step:

- finalize top-level project narrative and README around Stage 2 findings

Expected outcome:

- one stable top-level summary of prompt-only, repair, reduced-schema target design, LoRA-rank ablations, curriculum training, and seen/unseen schema generalization
- one clear statement of what post-training solves, what repair solves, and what still fails semantically

## Practical Rule

Current development priority:

- prioritize final narrative quality over adding broad new experiment branches
- do not spend major effort on broad hyperparameter tuning unless it directly supports a stronger final conclusion
- keep reusable logic in `src/` and `scripts/`, not notebooks
