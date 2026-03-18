# Minimal Viable Experiment Matrix

## Goal

Keep the first round small enough to finish, but rich enough to support a credible conclusion.

## Main Comparison

### Group A: No Post-Training

1. Prompt-only baseline

### Group B: Post-Training

2. SFT full fine-tuning or lightweight supervised baseline
3. LoRA or QLoRA baseline

### Group C: Decoding / Repair Enhancement

4. Prompt-only + validator repair
5. Best post-trained model + validator repair

This matrix is intentionally small. It isolates:

- what post-training fixes
- what repair still fixes after post-training

## Recommended First-Stage Variables

Control most variables and change only:

- training method
- decoding / repair strategy
- schema complexity bucket

Do not introduce many model families in the first stage.

## Suggested Evaluation Metrics

### Structure Metrics

- valid JSON rate
- schema compliance rate
- exact schema match rate

### Semantic Metrics

- field-level exact match
- macro field F1 for categorical / boolean fields
- object-array element match for list fields

### Aggregate

- end-to-end exact match
- complexity-bucket performance

## Error Taxonomy

Each failed sample should be assigned one primary error label:

1. invalid_json
2. schema_violation
3. missing_required_field
4. enum_error
5. type_error
6. value_hallucination
7. extraction_omission
8. cross_field_inconsistency

## First Conclusion You Want To Be Able To Write

An ideal phase-1 conclusion looks like:

Post-training substantially reduces semantic extraction errors and some schema violations, but decoding-time validation / repair remains necessary for the final layer of strict schema compliance under complex nested outputs.

## First Round Experiment Table

| Experiment ID | Model Variant | Training | Decoding | Repair | Purpose |
|---|---|---|---|---|---|
| E1 | base model | none | normal | no | prompt-only baseline |
| E2 | small model | SFT | normal | no | post-training effect |
| E3 | small model | LoRA/QLoRA | normal | no | efficient post-training effect |
| E4 | base model | none | normal | yes | repair-only contribution |
| E5 | best of E2/E3 | same as best | normal | yes | residual error after training |

## What Not To Do Yet

- do not compare too many base models
- do not add RLHF / DPO in phase 1
- do not build a large multi-task benchmark first
- do not spend early time on latency optimization
