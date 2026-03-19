# Data Validation

## Goal

Provide a repeatable validation step for every generated dataset version.

## Why A Dedicated Validation Script

Even though validation logic already exists in Python modules, a standalone script is better because:

- easier to run in Jupyter terminals
- easier to add to preprocessing workflows
- reduces manual notebook code duplication

## Validation Checks

Current validation should cover:

- required top-level fields
- metadata field correctness
- unique `sample_id`
- valid `complexity_bucket`
- non-empty `input_text`
- `target_json` schema compliance

## Recommended Workflow

After generating or modifying any dataset:

1. run preprocessing
2. run dataset validation
3. inspect profile summary
4. only then prepare train splits or SFT data
