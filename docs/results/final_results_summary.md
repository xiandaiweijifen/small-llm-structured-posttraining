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
| Stage 2 Reduced QLoRA, Curriculum | 1.0000 | 1.0000 | 0.9037 | 0.5315 | Best overall run; curriculum improves semantic learning beyond one-shot training |
| Schema-Conditioned Reduced QLoRA Generalization | 0.9980 | 0.9980 | 0.8764 | 0.4646 | Structure transfers well; semantics drop under schema shift |

## Stage 2 Takeaways

The Stage 2 ablations clarify where the strongest gains come from:

- small reduced-schema training sets are enough for structure, but not enough for the hardest semantic fields
- LoRA rank matters: rank 8 is clearly weaker, rank 16 is already competitive, and rank 32 gives a modest additional gain
- curriculum training matters more than rank tuning in this project; it becomes the best run overall
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

Curriculum provides the strongest improvement on the hardest field:

- `actions_requested[0].action`: `0.6811` in the curriculum run vs `0.6457` in the H200-fast reduced baseline

## Project-Level Conclusions

The experiments support a clear division of labor:

- prompt-only is weak mainly because it does not reliably satisfy schema requirements
- repair is effective for structural normalization and schema cleanup
- post-training is the main lever for stable structured generation
- target design matters: noisy identity fields can dominate failure modes and hide the model's real extraction ability
- after target cleanup, training strategy matters more than repair; curriculum gives the strongest overall result
- LoRA capacity helps, but it is a secondary lever compared with target design and curriculum-style training
- once structure is solved, the remaining bottleneck is semantic accuracy
- under mild schema shift, structure generalizes better than field semantics

## Recommended Project Narrative

The most defensible summary of the project is:

- built a structured-output post-training and evaluation framework for small models on complex text-to-JSON tasks
- compared prompt-only, post-training, repair, reduced-schema target design, curriculum training, LoRA-rank ablations, and seen/unseen schema generalization
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
- `results/metrics/qwen25_3b_schema_generalization_v1_test_report.json`
- `results/metrics/qwen25_3b_schema_generalization_v1_field_analysis.json`
