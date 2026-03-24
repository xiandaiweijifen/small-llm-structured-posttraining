# Project Brief

## One-Sentence Positioning

This project studies the capability boundary of small models on complex structured output tasks, with post-training as the main axis and constrained decoding / repair as supporting axes.

## Final Resume Narrative

Target the project toward a statement like:

Systematically studied small-model capability boundaries on complex schema-based structured output, and analyzed which failures can be resolved by post-training versus decoding-time constraint and repair mechanisms.

Current strongest project-level statement:

Built a small-model structured-output post-training framework for complex text-to-JSON tasks, and showed that target design, data scale, LoRA capacity, epoch duration, learning rate, staged structure-then-semantics training, action-target canonicalization, joint action+component target redesign, deterministic consistency passes, and a final high-precision lexical postprocess layer all affect semantic accuracy differently, while repair mainly helps prompt-only structural failures. Same-family prompt-only scaling from `3B` to `7B / 14B / 32B` still remains far below the post-trained 3B system because schema completeness does not emerge reliably from prompt-only use alone; on a mapped external customer-support dataset, the trained 3B still keeps a field-level edge over larger prompt-only references, but the task-aware Stage 8/9 postprocess gains do not transfer cleanly.

Best-result split to preserve in interviews:

- best trained in-domain run:
  - `qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
  - `0.9402 / 0.6772`
- best full in-domain system:
  - `qwen25_3b_stage9_lexical_combined`
  - `0.9470 / 0.7205`
- best mapped external run:
  - `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
  - `0.7517 / 0.0636`

## Core Research Question

For complex schema structured output tasks, what are the dominant failure modes of small models, and which of them can be mitigated by SFT / LoRA / data strategy optimization versus decoding-time constraint or repair?

After Stage 9, the clearest refined question is:

Once target design removes noisy fields and structure is mostly solved, which lever matters most for semantic correctness: more data, more LoRA capacity, stronger optimization settings, better staged training, further target canonicalization, deterministic consistency, or a final high-precision lexical postprocess pass on top of the best trained model?

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
- LoRA rank ablations
- curriculum versus one-shot training
- epoch and learning-rate ablations
- structure-first then semantics-focused staged training
- action-target canonicalization as a target-design intervention
- component follow-up target redesign and interaction with staged training
- deterministic postprocessing as a low-cost final consistency lever
- lexical postprocessing as a final precision-focused semantic cleanup layer

### Data

- schema-aware sample generation
- reduced-schema target design
- data scale and coverage comparison
- complexity-aware sample bucketing

### Analysis

- structure versus semantics error split
- simple versus complex schema buckets
- seen versus unseen schema generalization
- repair delta after post-training
- negative-result analysis for hard-example continuation

### Engineering

- modular dataset / schema / eval code
- config-driven experiments
- scripts reusable in Jupyter and CLI
- exported markdown result summaries for notebook runs
