# Data Format

## Goal

Define one stable dataset and prediction format before data collection and experiments expand.

## Dataset Record Format

Each sample should be stored as one JSON object per line in `.jsonl`.

Required fields:

```json
{
  "sample_id": "string",
  "task_name": "ticket_structured_output",
  "schema_name": "ticket_schema_v1",
  "complexity_bucket": "simple|medium|complex",
  "input_text": "raw user text",
  "target_json": {},
  "metadata": {
    "source_type": "email|issue|notification|task",
    "split": "train|val|test",
    "is_synthetic": true
  }
}
```

## Prediction Record Format

Each prediction file should also use `.jsonl`.

```json
{
  "sample_id": "string",
  "prediction_text": "raw model output text",
  "prediction_json": {},
  "metadata": {
    "model_name": "string",
    "experiment_id": "string"
  }
}
```

## Recommendation

Store both `prediction_text` and `prediction_json`.

Reason:

- `prediction_text` is needed for debugging formatting failures
- `prediction_json` is needed for evaluation after parsing or repair

## Phase 1 Minimal Files

- `data/raw/external/<source_name>/`
- `data/raw/exports/<source_name>/`
- `data/samples/phase1_train.jsonl`
- `data/samples/phase1_val.jsonl`
- `data/samples/phase1_test.jsonl`
- `data/samples/phase1_seed.jsonl`
- `results/predictions/<experiment_id>.jsonl`

## Notes

- `sample_id` must be globally unique inside phase 1
- `schema_name` should stay explicit to support later schema variants
- `complexity_bucket` is mandatory for grouped analysis
- for tiny seed datasets, `metadata.split` can be pre-assigned manually to avoid empty validation splits
