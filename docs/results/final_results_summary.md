# Final Results Summary

## Project Question

This project studies a focused question:

- for complex schema text-to-JSON tasks, what can small-model post-training solve on its own
- what failures are mainly structural and can be handled by repair or decoding constraints
- what failures remain semantic even after structure is stabilized
- how much performance drops when the schema shifts from seen to unseen variants

## Main Experiment Table

| Setting | Valid JSON | Schema Compliance | Field Exact Match | End-to-End Exact Match | Main Takeaway |
| --- | ---: | ---: | ---: | ---: | --- |
| Prompt-only | 0.9646 | 0.0000 | 0.2894 | 0.0000 | Fails mainly on structure and required fields |
| Prompt-only + Repair | 0.9646 | 0.9567 | 0.4685 | 0.0000 | Repair fixes structure, not semantics |
| Full-Schema QLoRA | 1.0000 | 1.0000 | 0.7708 | 0.0000 | Post-training stabilizes schema-compliant JSON |
| Full-Schema QLoRA + Repair | 1.0000 | 1.0000 | 0.7708 | 0.0000 | Repair adds little once structure is already solved |
| Reduced-Schema QLoRA | 1.0000 | 1.0000 | 0.8851 | 0.4882 | Removing noisy fields unlocks semantic gains |
| Reduced-Schema QLoRA + Repair | 1.0000 | 1.0000 | 0.8851 | 0.4882 | Repair again adds almost no value |
| Reduced-Schema QLoRA H200-Fast | 1.0000 | 1.0000 | 0.8919 | 0.4961 | Larger batch slightly improves the reduced baseline |
| Stage 2 Reduced QLoRA, 600 Samples | 1.0000 | 1.0000 | 0.7645 | 0.0394 | Small data preserves structure but collapses on semantic fields |
| Stage 2 Reduced QLoRA, Rank 8 | 0.9961 | 0.9961 | 0.8604 | 0.4173 | Lower LoRA capacity underfits the hardest semantic fields |
| Stage 2 Reduced QLoRA, Rank 16 | 1.0000 | 1.0000 | 0.8844 | 0.4843 | Rank 16 roughly recovers the original reduced-schema baseline |
| Stage 2 Reduced QLoRA, Rank 32 | 1.0000 | 1.0000 | 0.8912 | 0.5079 | Higher LoRA capacity gives a small end-to-end gain |
| Stage 2 Reduced QLoRA, Curriculum | 1.0000 | 1.0000 | 0.9037 | 0.5315 | Strong training-strategy gain over one-shot reduced-schema training |
| Stage 2 Reduced QLoRA, Rank 16, Epoch 5 | 1.0000 | 1.0000 | 0.9145 | 0.5709 | Longer training mainly improves the hardest semantic fields |
| Stage 2 Reduced QLoRA, Rank 16, Epoch 9 | 1.0000 | 1.0000 | 0.9166 | 0.5709 | Longer training past epoch 5 gives only marginal additional gain |
| Stage 2 Reduced QLoRA, Rank 16, LR 1e-4, Epoch 5 | 1.0000 | 1.0000 | 0.8901 | 0.4882 | Lower learning rate under-trains semantic fields |
| Stage 2 Reduced QLoRA, Rank 16, LR 2e-4, Epoch 5 | 1.0000 | 1.0000 | 0.9141 | 0.5709 | Best learning-rate balance among the tested single-stage runs |
| Stage 2 Reduced QLoRA, Rank 16, LR 4e-4, Epoch 5 | 1.0000 | 1.0000 | 0.9173 | 0.5591 | Slightly higher field accuracy, but worse end-to-end stability than 2e-4 |
| Stage 2 Reduced QLoRA, Structure Then Semantics | 1.0000 | 1.0000 | 0.9245 | 0.5787 | Strongest pre-canonicalization baseline; staged training improves the hardest semantic fields further |
| Stage 3 Hard-Only Continuation, x4, Epoch 1, LR 5e-5 | 1.0000 | 1.0000 | 0.8726 | 0.3307 | Hard-only continuation badly hurts overall quality |
| Stage 3 Full + Hard Mix, x2, Epoch 2, LR 1e-4 | 1.0000 | 1.0000 | 0.9155 | 0.5433 | Best hard-continuation result, but still below the strongest staged baseline |
| Stage 3 Full + Hard Mix, x3, Epoch 2, LR 1e-4 | 1.0000 | 1.0000 | 0.9023 | 0.5039 | Heavier hard oversampling degrades both field and end-to-end accuracy |
| Stage 3 Full + Hard Mix, x2, Epoch 2, LR 5e-5 | 1.0000 | 1.0000 | 0.9109 | 0.5354 | Lower learning rate does not recover the strongest staged baseline |
| Stage 6 Canonical Action, Single-Stage, Epoch 7, LR 2e-4 | 1.0000 | 1.0000 | 0.9341 | 0.6654 | First major canonicalization breakthrough; sharply reduces `action` entropy and lifts exact match |
| Stage 6 Canonical Action, Staged, Stage 2 Epoch 9 | 1.0000 | 1.0000 | 0.9320 | 0.6654 | Best staged canonicalized run, but not stronger than the simpler single-stage epoch-7 variant |
| Stage 7 Canonical Action + Component, Staged, Stage 2 Epoch 9 | 1.0000 | 1.0000 | 0.9402 | 0.6772 | Best overall run; joint target redesign plus staged training improves `component` enough to set a new best |
| Schema-Conditioned Reduced QLoRA Generalization | 0.9980 | 0.9980 | 0.8764 | 0.4646 | Structure transfers well; semantics drop under schema shift |

## Stage 2 Takeaways

The Stage 2 through Stage 7 ablations clarify where the strongest gains come from:

- small reduced-schema training sets are enough for structure, but not enough for the hardest semantic fields
- LoRA rank matters: rank 8 is clearly weaker, rank 16 is already competitive, and rank 32 gives a modest additional gain
- epoch duration matters up to about epoch 5; after that, field gains are marginal and end-to-end exact match plateaus
- learning rate matters: `1e-4` is too conservative, `2e-4` is the best balance, and `4e-4` trades a bit of end-to-end stability for higher average field accuracy
- staged structure-then-semantics training becomes the strongest pre-canonicalization baseline
- hard-sample continuation does not beat the strongest staged baseline; direct hard-only continuation and heavy oversampling both degrade performance
- action canonicalization is the first post-Stage-2 intervention that materially breaks through the previous end-to-end ceiling
- component canonicalization alone is weak, but joint `action + component` canonicalization becomes effective when paired with staged training
- the current strongest run combines target redesign and staged semantic continuation rather than relying on either lever alone
- repair still adds essentially no value once post-training has stabilized structure

## Generalization Breakdown

Seen vs unseen schema results from the schema-conditioned reduced QLoRA run:

| Split | Valid JSON | Schema Compliance | Field Exact Match | End-to-End Exact Match |
| --- | ---: | ---: | ---: | ---: |
| Seen schema | 1.0000 | 1.0000 | 0.8837 | 0.4764 |
| Unseen schema | 0.9961 | 0.9961 | 0.8691 | 0.4528 |

Interpretation:

- mild schema shift causes a small but real semantic drop
- structure stays almost perfectly stable
- unseen schema hurts semantic fields more than JSON validity or schema matching

## Main Error Patterns

Across the strongest runs, the remaining bottlenecks are concentrated in semantic fields:

- `actions_requested[0].action`
- `affected_systems[0].component`
- `category`
- `priority`

Fields that become highly stable after post-training include:

- `summary`
- `requires_followup`
- `constraints.environment`
- `constraints.blocking`

The strongest pre-canonicalization staged training provides the strongest improvement on the hardest field:

- `actions_requested[0].action`: `0.7323` in the structure-then-semantics run vs `0.6457` in the H200-fast reduced baseline

Stage 6 and Stage 7 canonicalization change the picture further:

- `actions_requested[0].action`: `0.8622` in the canonical-action single-stage epoch-7 run
- this large gain is the main reason end-to-end exact match rises from the `0.57x` range into the `0.66x` range
- `affected_systems[0].component`: `0.9173` in the Stage 7 joint canonicalized staged run
- this `component` gain is what pushes the best end-to-end exact match from `0.6654` to `0.6772`

Stage 3 hard mining also shows that the remaining semantic bottleneck is concentrated in a real subset of difficult samples:

- `561 / 1993` training samples still contain at least one error among `action`, `component`, `category`, and `priority`
- however, broad hard-subset continuation does not convert this finding into a stronger model under the current recipe

## Project-Level Conclusions

The experiments support a clear division of labor:

- prompt-only is weak mainly because it does not reliably satisfy schema requirements
- repair is effective for structural normalization and schema cleanup
- post-training is the main lever for stable structured generation
- target design matters: noisy identity fields can dominate failure modes and hide the model's real extraction ability
- after target cleanup, training strategy matters more than repair; structure-first then semantics-focused training gives the strongest pre-canonicalization result
- LoRA capacity, epoch duration, and learning rate all help, but they are secondary levers compared with target design plus stronger staged training
- hard-example continuation is not automatically beneficial; if applied too broadly, it causes distribution drift and hurts end-to-end exact match
- further target redesign can matter even more than continuation; canonicalizing the hardest semantic field produces the strongest overall run in the repository
- additional target redesign can still help when it is precise; `component` canonicalization only becomes useful when paired with the already-validated `action` redesign and staged training
- once structure is solved, the remaining bottleneck is semantic accuracy
- under mild schema shift, structure generalizes better than field semantics

## Recommended Project Narrative

The most defensible summary of the project is:

- built a structured-output post-training and evaluation framework for small models on complex text-to-JSON tasks
- compared prompt-only, post-training, repair, reduced-schema target design, curriculum training, LoRA-rank ablations, epoch and learning-rate ablations, staged structure-then-semantics training, hard-example continuation, action canonicalization, component follow-up target redesign, and seen/unseen schema generalization
- found that repair mainly fixes structure, while semantic correctness depends more on post-training quality, target design, training strategy, and schema-conditioned generalization

## Key Result Files

- `results/metrics/qwen25_3b_prompt_only_test_report.json`
- `results/metrics/qwen25_3b_prompt_only_test_repaired_report.json`
- `results/metrics/qwen25_3b_phase1_qlora_v1_test_report.json`
- `results/metrics/qwen25_3b_phase1_qlora_reduced_v1_test_report.json`
- `results/metrics/qwen25_3b_phase1_qlora_reduced_h200fast_test_report.json`
- `results/metrics/qwen25_3b_stage2_data_small_600_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank8_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank32_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_curriculum_sm_then_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_epoch5_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_epoch9_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_lr1e4_epoch5_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_lr2e4_epoch5_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_lr4e4_epoch5_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_structure_then_semantics_v1_test_report.json`
- `results/metrics/qwen25_3b_stage3_sts_v2_hard_only_x4_epoch1_lr5e5_test_report.json`
- `results/metrics/qwen25_3b_stage3_sts_v2_full_plus_hard_x2_epoch2_lr1e4_test_report.json`
- `results/metrics/qwen25_3b_stage3_sts_v2_full_plus_hard_x3_epoch2_lr1e4_test_report.json`
- `results/metrics/qwen25_3b_stage3_sts_v2_full_plus_hard_x2_epoch2_lr5e5_test_report.json`
- `results/metrics/qwen25_3b_stage6_canonical_action_single_stage_epoch7_lr2e4_test_report.json`
- `results/metrics/qwen25_3b_stage6_canonical_action_structure_then_semantics_stage2_epoch9_test_report.json`
- `docs/results/action_canonicalization_batch_summary.md`
- `results/metrics/qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9_test_report.json`
- `docs/results/component_canonicalization_batch_summary.md`
- `results/metrics/qwen25_3b_schema_generalization_v1_test_report.json`
- `results/metrics/qwen25_3b_schema_generalization_v1_field_analysis.json`
