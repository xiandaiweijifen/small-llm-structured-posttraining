# Reduced Schema

## Goal

Create a lower-noise training target to test whether semantic extraction improves when high-noise identity fields are removed.

## Motivation

Current QLoRA results show:

- strong structure learning
- weak identity-field accuracy

Worst fields include:

- `ticket_id`
- `reporter.name`
- `reporter.team`

These are likely noisy under the current weak supervision pipeline.

## Reduced Schema Definition

Keep:

- `summary`
- `category`
- `priority`
- `requires_followup`
- `affected_systems`
- `actions_requested`
- `constraints`

Drop:

- `ticket_id`
- `reporter`

## Why This Variant Matters

This is not a final production schema.

It is an experiment to answer:

- does the model learn semantic extraction better when high-noise fields are removed?

## Experiment Use

Compare:

1. prompt-only on base schema
2. QLoRA on base schema
3. QLoRA on reduced schema

If reduced-schema training improves semantic match, that supports the hypothesis that weakly supervised identity fields are hurting post-training quality.
