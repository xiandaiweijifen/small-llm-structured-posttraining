# Schema Generalization

## Goal

Turn phase-1 from a single-schema extraction project into a real schema-generalization study.

## Why This Matters

If the prompt only contains a schema name, unseen-schema evaluation is not meaningful.

For unseen-schema generalization, the model must see:

- the task
- the input text
- the actual target schema definition

Otherwise a schema-name change mostly measures label-name memorization rather than schema transfer.

## Phase-1 Generalization Setup

### Seen Schema

- `ticket_schema_v1_reduced`

### Unseen Schema

- `ticket_schema_v1_reduced_1_1`

Differences:

- same task semantics
- same core fields
- adds optional `customer_impact`

This keeps the generalization shift controlled and interpretable.

## Dataset Outputs

Use:

- `scripts/build_schema_generalization_eval.py`

This creates:

- `data/generalization/phase1_test_seen_reduced.jsonl`
- `data/generalization/phase1_test_unseen_reduced.jsonl`
- `data/generalization/phase1_test_seen_unseen_reduced.jsonl`

Each record is tagged with:

- `schema_name`
- `schema_seen_status`

## Training Prompt Requirement

For future schema-generalization training runs, prepare SFT data with:

```bash
python scripts/prepare_sft_data.py --input ... --output ... --include-schema-definition
```

This embeds the actual JSON schema in the user prompt and makes unseen-schema evaluation valid.

## Evaluation

`scripts/run_eval.py` now supports grouped reporting by:

- `complexity_bucket`
- `schema_name`
- `schema_seen_status`

This makes it possible to compare:

- seen schema performance
- unseen schema performance
- structure vs semantic degradation under schema shift
