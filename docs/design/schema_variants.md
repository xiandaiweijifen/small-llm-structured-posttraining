# Schema Variants

## Goal

Support a small number of controlled schema variants for phase-1 generalization analysis.

## Principle

Do not introduce many unrelated schemas.

Use:

- one base schema for most training data
- one or two small variants for held-out evaluation

## Current Variants

### `ticket_schema_v1`

Base phase-1 schema.

### `ticket_schema_v1_1`

Light variant for unseen-schema evaluation.

Differences from `ticket_schema_v1`:

- adds optional `customer_impact` field
- keeps task semantics nearly unchanged
- useful for testing whether model behavior transfers to lightly modified schemas

## Evaluation Usage

Use grouped reporting by:

- `schema_name`
- `complexity_bucket`

This allows:

- base-schema performance analysis
- light unseen-schema generalization analysis
