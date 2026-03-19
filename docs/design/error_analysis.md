# Error Analysis

## Goal

Move beyond aggregate scores and inspect which fields fail most often.

## Why This Matters

Current phase-1 QLoRA results show:

- perfect JSON validity
- perfect schema compliance
- zero exact-match success
- universal semantic mismatch

This means the next useful question is:

- which fields are still wrong most often
- whether errors are concentrated in identity fields, semantic labels, or nested content

## Field Groups

Track at least:

- `ticket_id`
- `summary`
- `category`
- `priority`
- `requires_followup`
- `reporter.name`
- `reporter.team`
- `affected_systems[0].name`
- `affected_systems[0].component`
- `actions_requested[0].action`
- `constraints.environment`
- `constraints.blocking`

## Intended Use

Use field-level analysis to compare:

- prompt-only baseline
- QLoRA baseline
- repaired outputs
- future cleaner-data retraining runs
