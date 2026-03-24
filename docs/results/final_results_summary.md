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
| Stage 7 Canonical Action + Component, Staged, Stage 2 Epoch 9 | 1.0000 | 1.0000 | 0.9402 | 0.6772 | Best trained run; joint target redesign plus staged training improves `component` enough to set a new best before postprocessing |
| Stage 8 Deterministic Action + Component Postprocess | 1.0000 | 1.0000 | 0.9427 | 0.6929 | Best overall result; a no-train consistency pass on top of Stage 7 gives another exact-match gain |
| Stage 9 Combined Lexical Postprocess | 1.0000 | 1.0000 | 0.9470 | 0.7205 | New best overall result; a high-precision lexical layer further improves `priority` and `blocking` without retraining |
| Stage 11 Semantic-Core Intermediate, Staged, Stage 2 Epoch 9 | 1.0000 | 1.0000 | 0.9256 | 0.6299 | A more principled intermediate-representation branch than Stage 10, but still below the canonicalized main line |
| Stage 12 Semantic-Slot Supervision, Staged, Stage 2 Epoch 11 | 1.0000 | 1.0000 | 0.9298 | 0.6496 | Strongest algorithmic exploration branch after Stage 9, but still below the Stage 7 canonicalized staged-training best |
| Stage 13 External Adaptation, 1024 Samples, Epoch 3, LR 1e-4 | 1.0000 | 1.0000 | 0.7512 | 0.0536 | First external adaptation branch to restore completeness and lift mapped external end-to-end exact match above zero |
| Stage 14 External Targeted Adaptation, All-Core, x1, Epoch 1, LR 5e-5 | 1.0000 | 1.0000 | 0.7517 | 0.0636 | Best mapped external result so far; light low-LR continuation on all core taxonomy fields improves the Stage 13 adapter further |
| Stage 16 External Adaptation, Full Data, Epoch 2, LR 5e-5 | 1.0000 | 1.0000 | 0.7537 | 0.0542 | Larger-scale external adaptation slightly improves field-level accuracy, but does not beat the Stage 14 end-to-end best |
| Stage 16 Retrieval Postprocess, KNN-3 Priority Majority | 1.0000 | 1.0000 | 0.7501 | 0.0554 | Retrieval gives a weak useful signal for `priority`, but nearest-neighbor transfer remains too noisy for a new best |
| Stage 17 External Target Redesign, Full Data, c80/c60, Epoch 2, LR 5e-5 | 1.0000 | 1.0000 | 0.7514 | 0.0530 | Field-level target redesign is directionally useful, but still below the simpler Stage 14 all-core continuation |
| Stage 17 Residual Curriculum, Component-Focused, x2, Epoch 1, LR 5e-5 | 1.0000 | 1.0000 | 0.7479 | 0.0512 | Field-targeted residual continuation does not outperform the Stage 14 recipe under the current external setup |
| Stage 18 Component Verifier, Guarded Name Majority p80 | 1.0000 | 1.0000 | 0.7517 | 0.0636 | Best verifier only ties the Stage 14 external best; high-purity `name -> component` rules rarely fire usefully on the external train distribution |
| Stage 18 Component Verifier, NB Text+Name+Pred Features | 1.0000 | 1.0000 | 0.7510 | 0.0489 | Simple learned component-only verifier underperforms the Stage 14 external best |
| Stage 19 vLLM Raw Prompt-Only Reference, 32B | 1.0000 | 0.0000 | 0.5737 | 0.0000 | vLLM prompt-only matches the earlier prompt-only conclusion: larger models still miss exact schema-based task success |
| Stage 19 vLLM Structured JSON Reference, 32B | 1.0000 | 1.0000 | 0.3672 | 0.0000 | Modern structured decoding guarantees structure, but semantic value errors remain overwhelming |
| Qwen2.5-3B Prompt-Only Reference, Canonicalized Target | 0.9724 | 0.0000 | 0.4470 | 0.0000 | Raw prompt-only on the same base model still fails mainly on required fields |
| Qwen2.5-7B Prompt-Only Reference, Canonicalized Target | 1.0000 | 0.0000 | 0.5251 | 0.0000 | Larger prompt-only model improves field average, but still misses required fields systematically |
| Qwen2.5-14B Prompt-Only Reference, Canonicalized Target | 0.9961 | 0.0000 | 0.5777 | 0.0000 | Strongest raw prompt-only reference, but still structurally unusable under exact schema evaluation |
| Qwen2.5-32B Prompt-Only Reference, Canonicalized Target | 1.0000 | 0.0000 | 0.5634 | 0.0000 | More scale alone does not solve schema completeness |
| Schema-Conditioned Reduced QLoRA Generalization | 0.9980 | 0.9980 | 0.8764 | 0.4646 | Structure transfers well; semantics drop under schema shift |

Implementation note for Stage 8 and Stage 9:

- both are script-driven postprocessing/evaluation stages
- their notebooks are thin Jupyter launchers, not separate notebook-only implementations

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
- the current strongest trained run combines target redesign and staged semantic continuation
- a final deterministic consistency pass on top of that trained run is still able to push exact match further
- a final lexical postprocess layer can still push exact match further when it is restricted to a small set of high-precision severity cues
- Stage 10 latent-action targets are not a viable direction under the current formulation
- Stage 11 semantic-core intermediates are more reasonable than Stage 10, but still weaker than the canonicalized main line
- Stage 12 semantic-slot auxiliary supervision is a valid algorithmic exploration direction and beats Stage 11, but still does not overtake Stage 7
- repair still adds essentially no value once post-training has stabilized structure
- same-family prompt-only scaling helps field-level averages somewhat, but none of the `3B / 7B / 14B / 32B` prompt-only references achieve non-zero end-to-end exact match under the canonicalized reduced-schema evaluation target
- on a mapped external customer-support eval set, the strongest trained 3B still leads `14B/32B` prompt-only references at field level (`0.6040` vs `0.4201/0.4212`), but all methods fall to `0.0000` end-to-end exact match because required-field completeness does not transfer
- the Stage 8/9 deterministic and lexical postprocess layers do not add measurable gains on that external mapped dataset, indicating that those layers are task-aware rather than broadly transferable
- Stage 13 external few-shot adaptation changes that picture: with enough mapped external supervision, completeness recovers (`schema_compliance = 1.0`) and external end-to-end exact match rises above zero (`0.0536`)
- under the current external adaptation recipe, pure external supervision works slightly better than mixing in-domain data, which suggests that the main problem is adapting to the new taxonomy rather than preserving the old one
- Stage 14 external targeted continuation improves that adapted external best further (`0.0536 -> 0.0636`) without changing completeness, which indicates that once structure is restored the next effective lever is narrow semantic-taxonomy continuation
- under the current Stage 14 recipe, targeting all core external semantic fields works better than narrower subsets, and `5e-5` works better than `1e-4`
- Stage 16 and Stage 17 establish an external plateau: larger-scale adaptation, retrieval-guided postprocess, field-level target redesign, and residual curriculum all preserve perfect completeness but still fail to beat the Stage 14 end-to-end best
- Stage 18 reinforces that plateau: a narrow `component`-only verifier can at best tie Stage 14 and simple learned verifier variants degrade external end-to-end exact match
- Stage 19 adds a stronger decode-side control result: vLLM structured outputs can force perfect JSON/schema compliance for prompt-only references, but still fail semantically and keep end-to-end exact match at zero
- this means the mapped external bottleneck is now firmly a semantic-taxonomy problem, especially `component`, then `priority`, then the coupled `category / action` fields
- the internal/external mismatch audit sharpens that conclusion: `category` and `priority` mostly share the same coarse label vocabulary across domains, but external `summary -> category` mappings are dramatically less pure and external `name -> component` mappings are far less reusable
- this means the external plateau is not mainly a vocab-OOD problem; it is a conditional label-semantics problem

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
- Stage 8 deterministic postprocessing pushes `affected_systems[0].component` further to `0.9370`
- this low-cost consistency pass is what moves the best end-to-end exact match from `0.6772` to `0.6929`
- Stage 9 lexical postprocessing pushes `priority` to `0.9016` and `constraints.blocking` to `0.9724`
- this is what moves the best end-to-end exact match from `0.6929` to `0.7205`

Stage 3 hard mining also shows that the remaining semantic bottleneck is concentrated in a real subset of difficult samples:

- `561 / 1993` training samples still contain at least one error among `action`, `component`, `category`, and `priority`
- however, broad hard-subset continuation does not convert this finding into a stronger model under the current recipe

## Project-Level Conclusions

The experiments support a clear division of labor:

- prompt-only is weak mainly because it does not reliably satisfy schema requirements
- same-family larger prompt-only models remain structurally weak on this task; larger scale alone does not replace post-training
- cross-dataset mapped evaluation shows that post-training still transfers better than prompt-only scaling, but the strongest system-level gains are less transferable than the core trained model
- cross-dataset few-shot adaptation is effective: it restores completeness and recovers non-zero external end-to-end exact match, but semantic taxonomy alignment remains the dominant post-adaptation bottleneck
- post-Stage-14 external follow-ups confirm that this bottleneck is real: neither more external data, nor retrieval-style postprocess, nor narrower continuation variants materially surpass the best all-core external continuation
- a narrow component-only verifier also fails to beat the best all-core external continuation, which shows that the remaining external issue is not a simple local `component` correction problem
- a modern vLLM structured-output stack also does not change the core project conclusion: decode-side control can solve structure, but not the semantic alignment needed for this task
- the internal/external mismatch audit explains why: the mapped external dataset preserves much of the same coarse label space, but changes the conditional semantics of those labels enough that simple reuse of in-domain mappings breaks down
- repair is effective for structural normalization and schema cleanup
- post-training is the main lever for stable structured generation
- target design matters: noisy identity fields can dominate failure modes and hide the model's real extraction ability
- after target cleanup, training strategy matters more than repair; structure-first then semantics-focused training gives the strongest pre-canonicalization result
- LoRA capacity, epoch duration, and learning rate all help, but they are secondary levers compared with target design plus stronger staged training
- hard-example continuation is not automatically beneficial; if applied too broadly, it causes distribution drift and hurts end-to-end exact match
- further target redesign can matter even more than continuation; canonicalizing the hardest semantic field produces the strongest overall run in the repository
- additional target redesign can still help when it is precise; `component` canonicalization only becomes useful when paired with the already-validated `action` redesign and staged training
- more algorithmic supervision designs are not automatically stronger: in this project, semantic-slot auxiliary supervision helps but still remains below the canonicalized target-design line
- even after target redesign saturates, a narrow deterministic consistency layer can still recover a few remaining exact-match errors without retraining
- even after deterministic consistency saturates, a very small high-precision lexical rule layer can still recover additional exact-match errors without retraining
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
- `results/metrics/qwen25_3b_stage8_action_component_majority_test_report.json`
- `docs/results/deterministic_postprocess_batch_summary.md`
- `results/metrics/qwen25_3b_stage9_lexical_combined_test_report.json`
- `docs/results/lexical_postprocess_batch_summary.md`
- `docs/results/big_model_reference_summary.md`
- `docs/results/gorkemsevinc_cst_eval_generalization_reference_summary.md`
- `results/metrics/qwen25_3b_reference_canonical_prompt_test_report.json`
- `results/metrics/qwen25_7b_reference_canonical_prompt_test_report.json`
- `results/metrics/qwen25_14b_reference_canonical_prompt_test_report.json`
- `results/metrics/qwen25_32b_reference_canonical_prompt_test_report.json`
- `results/metrics/gorkemsevinc_cst_eval_trained_stage7_raw_test_report.json`
- `results/metrics/gorkemsevinc_cst_eval_trained_stage9_lexical_test_report.json`
- `results/metrics/gorkemsevinc_cst_eval_prompt_14b_raw_test_report.json`
- `results/metrics/gorkemsevinc_cst_eval_prompt_32b_raw_test_report.json`
- `results/metrics/qwen25_3b_stage13_ext1024_epoch3_lr1e4_test_report.json`
- `docs/results/external_adaptation_batch_summary.md`
- `results/metrics/qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5_test_report.json`
- `docs/results/external_targeted_adaptation_batch_summary.md`
- `results/metrics/qwen25_3b_schema_generalization_v1_test_report.json`
- `results/metrics/qwen25_3b_schema_generalization_v1_field_analysis.json`
