# Model Selection

## Goal

Choose a first-round small instruct model that is realistic for Jupyter GPU training and aligned with structured-output tasks.

## Recommended First-Round Models

### Primary Recommendation

- `Qwen/Qwen2.5-3B-Instruct`

Why:

- strong instruction-following baseline
- explicitly positioned for structured data and JSON-friendly outputs
- still small enough for realistic LoRA / QLoRA experiments in school GPU environments

Reference:

- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

### Secondary Lightweight Fallback

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`

Why:

- smaller and cheaper to iterate on
- supports instruction tuning well
- useful fallback if 3B-scale runs are unstable in the available environment

Reference:

- https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct

## Recommended Phase-1 Order

1. prompt-only baseline with the chosen instruct model
2. QLoRA on `Qwen/Qwen2.5-3B-Instruct`
3. if compute is tight, run parallel fallback on `SmolLM2-1.7B-Instruct`

## What Not To Do First

- do not start with too many base models
- do not start with 7B+ if your Jupyter quota is uncertain
- do not start with full fine-tuning before LoRA / QLoRA feasibility is clear

## Working Recommendation

Use:

- main training track: `Qwen/Qwen2.5-3B-Instruct`
- fallback track: `HuggingFaceTB/SmolLM2-1.7B-Instruct`

This should be treated as a practical starting point, not a final benchmark roster.
