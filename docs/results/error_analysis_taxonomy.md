# Error Analysis Taxonomy

## Purpose

This document compresses the project's most important failure-analysis findings into one place.

It is meant to answer four questions:

- what prompt-only gets wrong
- what post-training fixes
- what still remains wrong in the best in-domain runs
- what changes under cross-dataset transfer and external adaptation

## Four Failure Regimes

### 1. Prompt-Only: Structure Failure Dominates

Representative run:

- `qwen25_14b_reference_canonical_prompt`
- `field_exact_match = 0.5777`
- `end_to_end_exact_match = 0.0000`

Primary pattern:

- the model often produces valid-looking JSON
- but required fields are missing almost everywhere
- schema completeness fails before semantic quality can even matter

Evidence:

- `schema_compliance_rate = 0.0000`
- `missing_required_field = 253 / 254`

Interpretation:

- larger prompt-only models improve field overlap somewhat
- but raw scaling does not make the model reliably enter the target output space
- this is why prompt-only and larger prompt-only references still lose badly to post-trained small models on this task

### 2. Strongest In-Domain Trained Run: Structure Solved, Semantics Remain

Representative run:

- `qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
- `field_exact_match = 0.9402`
- `end_to_end_exact_match = 0.6772`

At this point:

- `valid_json_rate = 1.0`
- `schema_compliance_rate = 1.0`
- repair adds essentially no value

The remaining errors are concentrated in a few semantic fields:

- `actions_requested[0].action = 0.8504`
- `category = 0.8622`
- `priority = 0.8701`
- `affected_systems[0].component = 0.9173`

Typical mistakes:

- wrong task type drives wrong action template
- category confusion such as `bug -> task` or `task -> question`
- severity over- or under-prediction
- component family confusion such as `software/account/security`

Interpretation:

- once structure is solved, end-to-end exact match is dominated by a very small semantic residual
- this is why target redesign and staged training matter more than repair at the top end

### 3. Strongest In-Domain System Run: Residual Error Becomes Even Narrower

Representative run:

- `qwen25_3b_stage9_lexical_combined`
- `field_exact_match = 0.9470`
- `end_to_end_exact_match = 0.7205`

Relative to Stage 7:

- `component` is improved further by deterministic consistency
- `priority` and `blocking` are improved further by high-precision lexical rules

Still-hard fields:

- `actions_requested[0].action = 0.8583`
- `category = 0.8622`
- `priority = 0.9016`
- `affected_systems[0].name = 0.9252`
- `affected_systems[0].component = 0.9370`

Interpretation:

- the best system no longer fails broadly
- it fails on a narrow band of coupled semantic decisions
- in-domain exact-match gains after Stage 7 come mostly from low-cost consistency cleanup, not from solving a new broad modeling problem

### 4. External Zero-Shot: Completeness Collapses Again

Representative run:

- `gorkemsevinc_cst_eval_trained_stage7_raw`
- `field_exact_match = 0.6040`
- `end_to_end_exact_match = 0.0000`

Primary error pattern:

- the trained 3B still beats larger prompt-only references at field level
- but required-field completeness collapses under dataset shift

Evidence:

- `schema_compliance_rate = 0.1504`
- `missing_required_field = 421 / 512`

Interpretation:

- the model has transferred some structure-and-semantics bias
- but it has not transferred enough output completeness for the mapped external taxonomy
- this is the key reason external zero-shot `end_to_end` remains zero

## External Adaptation Changes The Failure Regime

### Stage 13: Completeness Recovery

Representative run:

- `qwen25_3b_stage13_ext1024_epoch3_lr1e4`
- `field_exact_match = 0.7512`
- `end_to_end_exact_match = 0.0536`

What changes:

- `valid_json_rate = 1.0`
- `schema_compliance_rate = 1.0`
- `missing_required_field = 0`

Interpretation:

- few-shot external adaptation does not just nudge field averages
- it fundamentally changes the external failure mode
- after Stage 13, the problem is no longer completeness

### Stage 14: Best External Semantic Continuation

Representative run:

- `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- `field_exact_match = 0.7517`
- `end_to_end_exact_match = 0.0636`

Remaining external hardest fields:

- `affected_systems[0].component = 0.2091`
- `priority = 0.2803`
- `actions_requested[0].action = 0.3963`
- `category = 0.3963`

Interpretation:

- external adaptation solves structure first
- the remaining external problem is semantic taxonomy alignment
- `component`, `priority`, and the coupled `category/action` decision remain the dominant residuals

## External Plateau: What Later Runs Proved

Stage 16 and Stage 17 tested four follow-up directions:

- larger-scale external adaptation
- retrieval-guided taxonomy postprocess
- field-level target redesign
- residual curriculum

Best representatives:

- Stage 16 larger-scale adaptation:
  - `qwen25_3b_stage16_extfull_epoch2_lr5e5`
  - `0.7537 / 0.0542`
- Stage 16 retrieval:
  - `knn3_priority_majority`
  - `0.7501 / 0.0554`
- Stage 17 target redesign:
  - `qwen25_3b_stage17_redesignfull_c80_cat60_epoch2_lr5e5`
  - `0.7514 / 0.0530`
- Stage 17 residual curriculum:
  - `qwen25_3b_stage17_residual_component_focused_x2_epoch1_lr5e5`
  - `0.7479 / 0.0512`

None of them beat Stage 14.

Interpretation:

- external completeness is no longer the bottleneck
- more data, kNN transfer, redesign variants, and narrower continuation do not materially surpass the best Stage 14 continuation
- the external line has entered a semantic-taxonomy plateau

## Final Failure Taxonomy

The project now supports a clean four-level taxonomy:

1. `Structure failure`
   - missing required fields
   - invalid or incomplete schema output
   - dominant in prompt-only and external zero-shot

2. `Cheap consistency failure`
   - canonical action not aligned with predicted category
   - component inconsistent with predicted system name
   - high-severity cases not normalized into priority/blocking
   - addressed by Stage 8 and Stage 9 postprocess

3. `In-domain semantic residual`
   - category, action template, priority, component family confusion
   - dominant after structure is solved in the main trained pipeline

4. `Cross-dataset semantic taxonomy failure`
   - external label-space mismatch
   - component/category/priority/action do not align cleanly to the trained taxonomy
   - dominant after Stage 13 restores completeness

## Why This Matters For The Project Story

This error taxonomy explains the whole project at a glance:

- prompt-only mainly fails on structure
- repair mainly fixes structure
- post-training mainly fixes stable structured generation
- target redesign mainly lowers semantic target entropy
- deterministic and lexical postprocess clean up a narrow in-domain residual
- external few-shot adaptation fixes completeness
- external plateau results show that the last remaining problem is semantic taxonomy alignment, not formatting
