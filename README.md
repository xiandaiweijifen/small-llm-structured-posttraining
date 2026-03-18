# structured-output-small-llm

Research-oriented small LLM post-training project for complex structured outputs.

## Project Goal

This repository studies a focused question:

How far can small models go on complex text-to-JSON structured output tasks with post-training alone, and which failures still require constrained decoding or repair?

The project is designed to demonstrate:

- SFT / LoRA / QLoRA post-training ability
- data construction and sampling strategy design
- structured-output evaluation and error analysis
- light but reusable engineering instead of notebook-only experiments

## Phase 1 Scope

Phase 1 focuses on one primary task:

- input: natural language text
- output: JSON object under a moderately complex schema
- emphasis: post-training first, decoding enhancement second

Recommended first task:

- email / ticket / task-description understanding to structured JSON

Why this task:

- realistic enterprise-style structured extraction
- easy to define nested and partially optional schemas
- supports both seen-schema and unseen-schema analysis later

## Initial Experiment Line

1. Prompt-only baseline
2. SFT baseline
3. LoRA / QLoRA baseline
4. Same models with schema-aware validation / repair
5. Error decomposition:
   - JSON format errors
   - schema compliance errors
   - field-level semantic errors

## Repository Layout

```text
structured-output-small-llm/
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- src/
|   |-- common/
|   |-- data/
|   |-- training/
|   |-- inference/
|   |-- evaluation/
|   |-- schemas/
|   `-- utils/
|-- configs/
|   |-- dataset/
|   |-- train/
|   `-- eval/
|-- scripts/
|-- notebooks/
|-- results/
|   |-- metrics/
|   `-- predictions/
`-- docs/
    `-- design/
```

## Key Documents

- [project_brief.md](d:/project/small-llm-structured-posttraining/docs/project_brief.md)
- [phase1_task.md](d:/project/small-llm-structured-posttraining/docs/design/phase1_task.md)
- [experiment_matrix.md](d:/project/small-llm-structured-posttraining/docs/design/experiment_matrix.md)

## Development Principle

Keep reusable logic in `src/` and `scripts/`.

Use `notebooks/` only for:

- data exploration
- visualization
- error analysis
- result presentation

## Current Status

The repository currently contains:

- refined project scope
- phase-1 task and schema definition
- initial experiment matrix
- starter Python package skeleton

Implementation of dataset builders, training entrypoints, inference, and evaluation will follow this structure.
