# Project Status

## Current Stage

The project is no longer in setup mode.

Current status:

- phase-1 data pipeline is complete
- core baselines have been run
- repair baseline has been implemented and evaluated
- reduced-schema ablation has been validated
- schema-generalization infrastructure has been added

The project is currently in:

- **phase-1 result consolidation**
- **schema generalization expansion**

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

### 6. Generalization Infrastructure

Completed:

- added schema-conditioned prompt support
- added reduced unseen schema variant
- built seen/unseen schema evaluation datasets
- added grouped evaluation by `schema_seen_status`
- created dedicated schema-generalization notebook

Key outputs:

- `data/generalization/phase1_test_seen_reduced.jsonl`
- `data/generalization/phase1_test_unseen_reduced.jsonl`
- `data/generalization/phase1_test_seen_unseen_reduced.jsonl`
- `data/processed/phase1_sft_train_reduced_schema_conditioned.jsonl`
- `data/processed/phase1_sft_val_reduced_schema_conditioned.jsonl`
- `notebooks/06_schema_generalization_qlora.ipynb`

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

## What Is Still Missing

To reach the originally desired “more complete research project” level, the project still needs:

- a real seen vs unseen schema experiment result
- a stronger schema generalization conclusion
- optionally, a decoding-side constrained generation baseline beyond repair
- final result tables and a concise project summary for resume/interview use

## Current Next Step

Immediate next step:

- run `notebooks/06_schema_generalization_qlora.ipynb`

Expected outcome:

- one schema-conditioned reduced-schema QLoRA model
- one combined seen/unseen prediction file
- one evaluation report split by `schema_seen_status`

## Practical Rule

Current development priority:

- prioritize experiments that deepen the research story
- do not spend major effort on broad hyperparameter tuning yet
- keep reusable logic in `src/` and `scripts/`, not notebooks
