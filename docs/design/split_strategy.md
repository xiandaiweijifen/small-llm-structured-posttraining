# Split Strategy

## Goal

Build phase-1 train / val / test splits from the candidate dataset with stable distribution.

## Principle

Do not split purely at random.

At minimum, preserve:

- `complexity_bucket`
- `raw_source`

Optional later refinement:

- `category`
- `schema_name`

## Current Strategy

Use stratified grouping by:

- `complexity_bucket`
- `metadata.raw_source`

This is a practical compromise:

- simple enough to implement and audit
- enough to reduce major distribution drift

## Outputs

- `data/samples/phase1_train.jsonl`
- `data/samples/phase1_val.jsonl`
- `data/samples/phase1_test.jsonl`
- `data/samples/phase1_split_summary.json`

## Validation

After splitting:

1. validate each split
2. inspect split summary
3. prepare SFT-formatted train and val files
