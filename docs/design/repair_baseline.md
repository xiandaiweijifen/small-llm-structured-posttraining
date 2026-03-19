# Repair Baseline

## Goal

Quantify what schema-aware repair can fix without any additional training.

## Why This Matters

Current experiments show a clean separation:

- prompt-only fails mainly on structure
- QLoRA fixes structure but still suffers semantic hallucination

This makes repair a useful comparison point.

## Main Questions

- how much can repair improve prompt-only outputs?
- how much residual value does repair have after post-training?
- does repair mainly help structure, or can it also help field-level semantics?

## Recommended Comparisons

1. prompt-only
2. prompt-only + repair
3. QLoRA full schema
4. QLoRA full schema + repair
5. QLoRA reduced schema
6. QLoRA reduced schema + repair

## Current Local Tooling

Use:

- `scripts/run_repair.py`
- `scripts/evaluate_with_repair.py`

## Current Repair Scope

The repair baseline is intentionally lightweight and deterministic:

- remove unknown keys
- fill schema-required defaults when safe
- map obvious alias fields such as `subject -> summary`
- normalize numeric/string priority values into schema enums
- infer coarse `category` from `type` and text keywords
- map `created_by` and similar fields into `reporter.name`

It is not intended to fabricate rich semantic content. This keeps repair useful as a
comparison point against post-training rather than turning it into another extraction model.
