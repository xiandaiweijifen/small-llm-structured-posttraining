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

## Current Project-Level Conclusion

The phase-1 experiments support a clear division of labor:

- prompt-only is weak on schema compliance
- repair is strong for structural normalization
- post-training is the main lever for stable structured generation
- noisy target fields can dominate failure modes and hide real extraction ability
- after structure is solved, the remaining problem is semantic accuracy rather than formatting
- mild schema shift mostly hurts semantic fields, not structural compliance
