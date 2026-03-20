# Phase 1 Baseline Findings

## Full-Schema Prompt-Only

- valid JSON rate: `0.9646`
- schema compliance rate: `0.0000`
- field exact match: `0.2894`
- end-to-end exact match: `0.0000`

Main failure mode:

- mostly missing required schema fields

## Full-Schema Prompt-Only + Repair

- valid JSON rate: `0.9646`
- schema compliance rate: `0.9567`
- field exact match: `0.4685`
- end-to-end exact match: `0.0000`

Interpretation:

- lightweight repair is enough to fix most structural failures
- repair improves schema compliance substantially
- repair does not solve semantic extraction

## Full-Schema QLoRA

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7708`
- end-to-end exact match: `0.0000`

Main failure mode:

- all remaining failures are value hallucination

Interpretation:

- post-training teaches the model to emit schema-compliant JSON reliably
- structure is solved, semantics are not

## Full-Schema QLoRA + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7708`
- end-to-end exact match: `0.0000`

Interpretation:

- repair adds essentially no value after full-schema post-training
- once structure is already correct, repair cannot fix hallucinated content

## Reduced-Schema QLoRA

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8851`
- end-to-end exact match: `0.4882`

By complexity bucket:

- simple exact match: `0.3770`
- medium exact match: `0.5625`
- complex exact match: `0.1176`

Interpretation:

- removing noisy identity fields materially improves learning
- the main bottleneck is not only model size, but also target design and label quality

## Reduced-Schema QLoRA + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8851`
- end-to-end exact match: `0.4882`

Interpretation:

- reduced-schema QLoRA already solves the structure problem
- repair again adds almost no value

## Schema-Conditioned Reduced QLoRA Generalization

- overall valid JSON rate: `0.9980`
- overall schema compliance rate: `0.9980`
- overall field exact match: `0.8764`
- overall end-to-end exact match: `0.4646`

Seen vs unseen schema:

- seen field exact match: `0.8837`
- unseen field exact match: `0.8691`
- seen end-to-end exact match: `0.4764`
- unseen end-to-end exact match: `0.4528`

Interpretation:

- schema-conditioned post-training preserves structure under mild schema shift
- unseen schema causes a small but real semantic drop
- the main generalization bottleneck is still field semantics rather than JSON validity

## Stage 2 Data-Regime Ablation

### Reduced-Schema QLoRA on 600 Training Samples

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7645`
- end-to-end exact match: `0.0394`

Interpretation:

- a small reduced-schema subset is enough to preserve structural learning
- the run collapses mainly on semantic fields rather than formatting
- this is strong evidence that post-training quality depends heavily on data scale and coverage, not only on whether fine-tuning happened at all

Main degraded semantic fields:

- `actions_requested[0].action`: `0.1102`
- `category`: `0.5000`
- `affected_systems[0].component`: `0.5354`
- `priority`: `0.7047`

### Reduced-Schema QLoRA on 600 Training Samples + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.7645`
- end-to-end exact match: `0.0394`

Interpretation:

- repair adds nothing once the small-data model already emits structurally valid JSON
- the remaining failure mode is semantic weakness, not residual structure errors

## Stage 2 LoRA Rank Ablation

### Rank 8

- valid JSON rate: `0.9961`
- schema compliance rate: `0.9961`
- field exact match: `0.8604`
- end-to-end exact match: `0.4173`

Interpretation:

- rank 8 underfits the reduced-schema task relative to stronger baselines
- the hardest semantic fields remain noticeably weaker than the best rank 16 and rank 32 runs

### Rank 16

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8844`
- end-to-end exact match: `0.4843`

Interpretation:

- rank 16 is already enough to recover the original reduced-schema baseline level
- this is a good efficiency-quality tradeoff point if the goal is a practical default

### Rank 32

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.8912`
- end-to-end exact match: `0.5079`

Interpretation:

- higher LoRA capacity gives a small but real improvement over rank 16
- the gain shows up mostly in the hardest semantic fields and in end-to-end exact match
- rank 32 becomes the strongest non-curriculum Stage 2 run

Rank trend summary:

- rank 8 is clearly worse than rank 16 and rank 32
- rank 16 roughly matches the original reduced-schema baseline
- rank 32 improves semantic-field learning, especially `action`, `category`, and `priority`

## Stage 2 Curriculum Training

### Simple/Medium Then Full Continuation

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9037`
- end-to-end exact match: `0.5315`

Interpretation:

- curriculum training is the strongest run in the project so far
- it outperforms the previous H200-fast reduced baseline on end-to-end exact match
- the gain comes from better semantic learning rather than any structural repair effect

Main improved semantic fields:

- `actions_requested[0].action`: `0.6811`
- `affected_systems[0].component`: `0.7953`
- `category`: `0.8268`
- `priority`: `0.8307`

### Curriculum + Repair

- valid JSON rate: `1.0000`
- schema compliance rate: `1.0000`
- field exact match: `0.9037`
- end-to-end exact match: `0.5315`

Interpretation:

- repair again adds no measurable value once post-training has already solved structure
- the curriculum result strengthens the project's conclusion about training-vs-repair division of labor

## Current Project-Level Conclusion

The combined Stage 1 and Stage 2 experiments support a clear division of labor:

- prompt-only is weak on schema compliance
- repair is strong for structural normalization
- post-training is the main lever for stable structured generation
- noisy target fields can dominate failure modes and hide real extraction ability
- training strategy matters after target design is cleaned up: curriculum improves semantic learning beyond one-shot reduced-schema training
- LoRA capacity matters, but its gains are smaller than the gains from the best training strategy
- after structure is solved, the remaining problem is semantic accuracy rather than formatting
- mild schema shift mostly hurts semantic fields, not structural compliance
