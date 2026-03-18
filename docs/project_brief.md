# Project Brief

## One-Sentence Positioning

This project studies the capability boundary of small models on complex structured output tasks, with post-training as the main axis and constrained decoding / repair as supporting axes.

## Final Resume Narrative

Target the project toward a statement like:

Systematically studied small-model capability boundaries on complex schema-based structured output, and analyzed which failures can be resolved by post-training versus decoding-time constraint and repair mechanisms.

## Core Research Question

For complex schema structured output tasks, what are the dominant failure modes of small models, and which of them can be mitigated by SFT / LoRA / data strategy optimization versus decoding-time constraint or repair?

## Non-Goals

- not a generic LLM application demo
- not primarily an inference acceleration project
- not a benchmark-collection project with too many unrelated tasks
- not a notebook-only exploratory repo

## Why This Topic Works Well

### Hot

- structured outputs are now a production requirement
- function calling / JSON schema compliance is industry-relevant
- small-model post-training remains highly practical

### Deep

- separates structure compliance from semantic correctness
- lets you study training-time versus decoding-time responsibilities
- supports generalization analysis on seen versus unseen schemas

### Feasible

- can start from one task family
- can use SFT / LoRA without requiring huge compute
- can build synthetic and human-curated samples incrementally

## Phase 1 Deliverables

- one well-defined text-to-JSON task
- one moderately complex schema family
- prompt-only baseline
- SFT / LoRA baselines
- validation / repair baseline
- evaluation pipeline with error taxonomy
- result tables and qualitative error analysis

## Main Axes To Showcase

### Post-Training

- instruction data formatting
- SFT / LoRA setup
- training configuration choices

### Data

- schema-aware sample generation
- high-quality versus synthetic data comparison
- complexity-aware sample bucketing

### Analysis

- structure versus semantics error split
- simple versus complex schema buckets
- seen versus unseen schema generalization

### Engineering

- modular dataset / schema / eval code
- config-driven experiments
- scripts reusable in Jupyter and CLI
