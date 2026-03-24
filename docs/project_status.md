# Project Status

## Current Stage

The project is no longer in setup mode.

Current status:

- phase-1 data pipeline is complete
- core baselines have been run
- repair baseline has been implemented and evaluated
- reduced-schema ablation has been validated
- schema-generalization experiment has been run and evaluated
- Stage 2 data-regime, LoRA-rank, curriculum, epoch, and learning-rate ablations have been run and reviewed
- structure-then-semantics staged training has been run and reviewed
- Stage 3/4/5 hard-example continuation and targeted refinement branches have been run and reviewed
- Stage 6 action-canonicalization experiments have been run and reviewed
- Stage 7 component-canonicalization follow-ups have been run and reviewed
- Stage 8 deterministic postprocessing follow-ups have been run and reviewed
- Stage 9 lexical postprocessing follow-ups have been run and reviewed
- same-family big-model prompt-only reference runs have been executed and reviewed
- external-dataset generalization references have been executed and reviewed
- Stage 10 latent-action, Stage 11 semantic-core intermediate, and Stage 12 semantic-slot auxiliary-supervision explorations have been executed and reviewed
- Stage 13 external few-shot adaptation experiments have been executed and reviewed
- Stage 14 external targeted adaptation experiments have been executed and reviewed
- Stage 16 external frontier experiments and Stage 17 deeper-direction external experiments have been executed and reviewed
- Stage 18 external component-verifier experiments have been executed and reviewed
- internal/external mismatch audit has been executed and reviewed

The project is currently in:

- **final result consolidation**
- **research-story finalization**

## Active Vs Retired Workflows

Active workflows:

- Stage 2 baseline and ablation execution / review
- long-run training ablations
- action and component canonicalization
- deterministic and lexical postprocessing
- same-family big-model prompt-only reference comparison
- external-dataset generalization comparison
- semantic-core intermediate exploration
- semantic-slot auxiliary-supervision exploration
- external few-shot adaptation from the strongest Stage 7 checkpoint
- external targeted adaptation on top of the strongest Stage 13 checkpoint
- external frontier scaling, retrieval postprocess, field-level target redesign, and residual-curriculum follow-ups
- external component-verifier follow-ups
- internal/external mismatch auditing for taxonomy-shift diagnosis

Retired workflows:

- constrained decoding
- broad hard-continuation batch branches
- targeted continuation / refinement branches
- latent-action template branch

These retired branches are kept only for historical traceability and negative-result references; they are no longer part of the active top-level workflow.

## Development History

### 1. Repository and Research Scoping

Completed:

- defined the project around small-model structured output post-training
- clarified the main question: what post-training solves vs what decoding/repair still must solve
- created modular repo structure with `src/`, `configs/`, `scripts/`, `docs/`, `notebooks/`, `results/`

### 2. Data Pipeline

Completed:

- fetched two external datasets
- mapped them into the project ticket schema
- added validation, profiling, and complexity relabeling
- built candidate data and stratified `train/val/test`
- prepared SFT-format data

Key outputs:

- `data/samples/phase1_train.jsonl`
- `data/samples/phase1_val.jsonl`
- `data/samples/phase1_test.jsonl`
- `data/processed/phase1_sft_train.jsonl`
- `data/processed/phase1_sft_val.jsonl`

### 3. Full-Schema Baselines

Completed:

- prompt-only baseline
- full-schema QLoRA baseline
- evaluation pipeline
- field-level error analysis

Main conclusion:

- prompt-only fails mainly on structure
- full-schema QLoRA fixes structure but still hallucinates field values

### 4. Reduced-Schema Ablation

Completed:

- built reduced schema removing noisy identity fields
- trained reduced-schema QLoRA baseline
- compared full vs reduced schema
- reran reduced-schema with larger H200-friendly batch settings

Main conclusion:

- noisy target fields materially hurt semantic learning
- reduced schema significantly improves end-to-end exact match

### 5. Repair Baseline

Completed:

- implemented schema-aware repair
- extended repair with conservative alias mapping and normalization
- evaluated prompt-only + repair
- evaluated QLoRA + repair

Main conclusion:

- repair strongly improves structural compliance for prompt-only outputs
- repair adds little once post-training already solves structure

### 6. Schema Generalization

Completed:

- added schema-conditioned prompt support
- added reduced unseen schema variant
- built seen/unseen schema evaluation datasets
- added grouped evaluation by `schema_seen_status`
- created dedicated schema-generalization notebook
- ran schema-conditioned reduced-schema QLoRA evaluation on combined seen/unseen schema test data

Key outputs:

- `data/generalization/phase1_test_seen_reduced.jsonl`
- `data/generalization/phase1_test_unseen_reduced.jsonl`
- `data/generalization/phase1_test_seen_unseen_reduced.jsonl`
- `data/processed/phase1_sft_train_reduced_schema_conditioned.jsonl`
- `data/processed/phase1_sft_val_reduced_schema_conditioned.jsonl`
- `notebooks/06_schema_generalization_qlora.ipynb`
- `results/metrics/qwen25_3b_schema_generalization_v1_test_report.json`
- `results/metrics/qwen25_3b_schema_generalization_v1_field_analysis.json`

### 7. Stage 2 Post-Training Ablations

Completed:

- ran a small-data reduced-schema training regime on 600 training samples
- ran LoRA rank ablations at 8, 16, and 32
- implemented and ran curriculum training from simple/medium buckets into full reduced-schema continuation
- ran rank-16 epoch ablations through epoch 9
- ran rank-16 learning-rate ablations at epoch 5
- implemented and ran structure-first then semantics-focused staged training
- ran Stage 3 hard-sample mining and continuation experiments on top of the strongest staged-training checkpoint
- ran Stage 4 targeted continuation experiments on narrower semantic subsets
- ran Stage 5 targeted refinement experiments around the best Stage 4 subset
- implemented and ran Stage 6 action-canonicalization experiments in both single-stage and staged forms
- implemented and ran Stage 7 component-canonicalization follow-ups, including joint `action + component` target redesign
- implemented and ran Stage 8 deterministic postprocessing variants on top of the Stage 7 best predictions
- implemented and ran Stage 9 lexical postprocessing variants on top of the Stage 8 best predictions
- kept Stage 8 and Stage 9 as script-driven postprocessing experiments with matching notebook launchers, rather than notebook-only workflows
- ran `3B / 7B / 14B / 32B` same-family prompt-only reference comparisons under the canonicalized reduced-schema evaluation target
- exported a big-model reference summary
- built a mapped reduced-schema eval set from `gorkemsevinc/customer_support_tickets`
- ran cross-dataset references comparing the strongest trained/system 3B lines against `14B` and `32B` prompt-only references
- exported an external generalization reference summary
- exported a consolidated Stage 2 review summary
- exported a consolidated long-run batch summary
- exported a Stage 3 end-to-end optimization batch summary
- exported a Stage 6 action-canonicalization batch summary
- exported a Stage 7 component-canonicalization batch summary
- exported a Stage 8 deterministic postprocessing batch summary
- exported a Stage 9 lexical postprocessing batch summary

Key outputs:

- `results/metrics/qwen25_3b_stage2_data_small_600_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank8_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_rank32_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_curriculum_sm_then_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_epoch9_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_lr2e4_epoch5_rank16_full_test_report.json`
- `results/metrics/qwen25_3b_stage2_structure_then_semantics_v1_test_report.json`
- `docs/results/stage2_results_review.md`
- `docs/results/long_run_ablation_batch_summary.md`
- `results/metrics/qwen25_3b_stage3_sts_v2_full_plus_hard_x2_epoch2_lr1e4_test_report.json`
- `docs/results/end_to_end_optimization_batch_summary.md`
- `results/metrics/qwen25_3b_stage6_canonical_action_single_stage_epoch7_lr2e4_test_report.json`
- `docs/results/action_canonicalization_batch_summary.md`
- `results/metrics/qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9_test_report.json`
- `docs/results/component_canonicalization_batch_summary.md`
- `results/metrics/qwen25_3b_stage8_action_component_majority_test_report.json`
- `docs/results/deterministic_postprocess_batch_summary.md`
- `results/metrics/qwen25_3b_stage9_lexical_combined_test_report.json`
- `docs/results/lexical_postprocess_batch_summary.md`

Main conclusions:

- small reduced-schema datasets are enough to preserve structure, but not enough to learn the hardest semantic fields
- rank 8 underfits, rank 16 is competitive, and rank 32 gives a modest additional gain
- epoch helps strongly up to about epoch 5, then mostly saturates
- `2e-4` is the best learning-rate balance among the tested single-stage runs
- broad hard-sample continuation and targeted refinement branches do not beat the strongest staged baseline
- action canonicalization is the first post-Stage-2 change that materially raises the end-to-end ceiling
- component canonicalization alone is weak, but joint `action + component` canonicalization becomes effective when paired with staged training
- deterministic postprocessing on top of the Stage 7 best run yields a further no-train gain
- lexical postprocessing on top of the Stage 8 best run yields another no-train gain
- the current best run is Stage 9 combined lexical postprocessing on top of the Stage 8 best model
- hard-sample continuation identifies a real semantic hard subset, but the current continuation recipes do not beat the strongest staged baseline
- same-family larger prompt-only models remain far below the trained 3B pipeline because they still fail on required-field completeness and schema compliance
- on the mapped customer-support eval set, the strongest trained 3B still outperforms `14B/32B` prompt-only references, but all runs fall to zero end-to-end exact match because schema completeness does not transfer cleanly
- Stage 8/9 postprocessing gains are strongly task-aware and do not meaningfully transfer to this external mapped dataset
- Stage 10 latent-action targets are a negative result
- Stage 11 semantic-core intermediate modeling is a more reasonable exploration, but still below the canonicalized main line
- Stage 12 semantic-slot auxiliary supervision is the strongest algorithmic exploration branch after Stage 9, but still does not beat the Stage 7 canonicalized staged-training baseline
- Stage 13 external few-shot adaptation is a real positive result: it restores schema completeness on the mapped external dataset and lifts external end-to-end exact match above zero
- Stage 14 external targeted adaptation is also a real positive result: it pushes the best mapped external end-to-end exact match further without reintroducing completeness failures
- Stage 16/17 show a clear external plateau: larger-scale adaptation, retrieval-guided postprocess, field-level target redesign, and residual curriculum all preserve completeness but still fail to beat the Stage 14 best
- Stage 18 reinforces that plateau: a narrow `component`-only verifier branch also fails to beat the Stage 14 best external run
- the internal/external mismatch audit explains that plateau more concretely: the main remaining issue is not label absence, but conditional semantic mismatch, especially weakly reusable `name -> component` mappings and highly ambiguous external `summary -> category` mappings
- repair still adds no measurable value once post-training has already stabilized structure

## Current Experimental Findings

### Prompt-only

- weak schema compliance
- often outputs partial or wrong-format JSON

### Prompt-only + Repair

- structure becomes mostly valid
- semantic extraction is still weak

### Full-Schema QLoRA

- structure is stable
- remaining bottleneck is value hallucination

### Reduced-Schema QLoRA

- best phase-1 result so far
- confirms the importance of target design and label quality

### Reduced-Schema QLoRA H200-Fast

- small improvement over the first reduced baseline
- useful as a stronger reproduction run, not a new qualitative breakthrough

### Schema-Conditioned Reduced QLoRA Generalization

- overall field exact match: `0.8764`
- overall end-to-end exact match: `0.4646`
- seen-schema field exact match: `0.8837`
- unseen-schema field exact match: `0.8691`
- seen-schema end-to-end exact match: `0.4764`
- unseen-schema end-to-end exact match: `0.4528`

Main conclusion:

- mild schema shift hurts semantics more than structure
- schema-conditioned post-training generalizes reasonably well to unseen schema variants
- remaining failures are concentrated in semantic fields such as `action`, `component`, `category`, and `priority`

### Stage 2 Reduced-Schema QLoRA, 600 Samples

- valid JSON and schema compliance stay perfect
- semantic performance drops sharply
- confirms that data scale and coverage are critical for post-training quality

### Stage 2 Reduced-Schema QLoRA Rank Ablation

- rank 8 is clearly weaker
- rank 16 recovers the earlier reduced baseline
- rank 32 becomes the strongest non-curriculum Stage 2 run

### Stage 2 Reduced-Schema Curriculum Training

- overall field exact match: `0.9037`
- overall end-to-end exact match: `0.5315`

Main conclusion:

- curriculum training is a strong improvement over one-shot reduced-schema training
- the gain is concentrated in semantic fields rather than structural repair

### Stage 2 Epoch and Learning-Rate Follow-Ups

- epoch 5 reaches `0.9145 / 0.5709`
- epoch 9 reaches `0.9166 / 0.5709`
- learning rate `1e-4` reaches `0.8901 / 0.4882`
- learning rate `2e-4` reaches `0.9141 / 0.5709`
- learning rate `4e-4` reaches `0.9173 / 0.5591`

Main conclusion:

- training duration matters mainly because it continues to improve semantic fields after structure is already solved
- learning rate changes semantic convergence quality more than structural stability
- the best single-stage balance in this set remains rank 16, epoch 5, learning rate `2e-4`

### Stage 2 Structure-Then-Semantics Training

- overall field exact match: `0.9245`
- overall end-to-end exact match: `0.5787`

Main conclusion:

- staged training now produces the strongest pre-canonicalization result in the repository
- explicitly separating structure-focused and semantics-focused phases improves the hardest semantic fields further

### Stage 3 Hard-Sample Continuation

- hard mining finds `561 / 1993` train samples with errors on the key semantic fields
- best continuation run: `0.9155 / 0.5433`
- strongest pre-canonicalization baseline remains `0.9245 / 0.5787`

Main conclusion:

- hard-sample mining confirms that the remaining bottleneck is concentrated in a real subset of difficult semantic examples
- however, direct hard-only continuation and heavy hard oversampling both hurt overall quality
- even the best mixed continuation remains below the strongest staged-training baseline
- the current hard-continuation recipe is therefore a useful negative result, not a new best model

### Stage 6 Action Canonicalization

- best run: `qwen25_3b_stage6_canonical_action_single_stage_epoch7_lr2e4`
- overall field exact match: `0.9341`
- overall end-to-end exact match: `0.6654`

Main conclusion:

- canonicalizing `actions_requested[0].action` is the first change that clearly breaks through the Stage 2 performance ceiling
- the main gain comes from target redesign rather than additional continuation or repair
- under the canonicalized target, a simpler single-stage run is slightly stronger than the more complex staged variants
- this result should be interpreted as a target-design improvement, not just a training-hyperparameter gain

### Stage 7 Component Canonicalization Follow-Up

- best run: `qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
- overall field exact match: `0.9402`
- overall end-to-end exact match: `0.6772`

Main conclusion:

- `component` canonicalization alone is not a strong direction
- however, joint `action + component` canonicalization does improve the best result when it is paired with staged training
- the main Stage 7 gain comes from a large improvement on `affected_systems[0].component`, which compensates for a slight drop in `action`
- this becomes the strongest trained run in the repository before any deterministic postprocessing

### Stage 8 Deterministic Postprocessing

- best run: `qwen25_3b_stage8_action_component_majority`
- overall field exact match: `0.9427`
- overall end-to-end exact match: `0.6929`

Main conclusion:

- a deterministic consistency pass can still extract additional gains after training saturates
- the useful rule is simple: refresh canonical `action` from predicted `category + summary`, and map `component` from predicted `name` using the train-set majority mapping
- almost all of the Stage 8 gain comes from the `component <- name` consistency rule; `action` refresh improves field exact match slightly but does not change end-to-end exact match by itself
- this is now the current strongest run in the repository
- this stage was validated by running the Python postprocessing script directly; the notebook is only a thin Jupyter launcher for the same script

### Stage 9 Lexical Postprocessing

- best run: `qwen25_3b_stage9_lexical_combined`
- overall field exact match: `0.9470`
- overall end-to-end exact match: `0.7205`

Main conclusion:

- a final high-precision lexical cleanup layer can still deliver another meaningful no-train gain after deterministic consistency has already been applied
- the useful gain comes mainly from promoting a small set of clearly severe cases to `priority=urgent` and `blocking=true`
- lexical `incident` relabeling does not help by itself; the Stage 9 gain is primarily a `priority` and `blocking` gain
- this is now the current strongest run in the repository
- this stage was also validated by running the Python postprocessing script directly; the notebook is only a thin Jupyter launcher for the same script

### Stage 10, Stage 11, and Stage 12 Algorithmic Exploration Branches

- Stage 10 latent-action targets are a useful negative result: directly replacing the final `action` text with template tokens destabilizes the task rather than improving it
- Stage 11 semantic-core intermediate modeling is much more reasonable than Stage 10, but still remains below the canonicalized main line
- Stage 12 semantic-slot auxiliary supervision is the strongest of these three exploration branches
- best Stage 12 run: `qwen25_3b_stage12_semantic_slot_structure_then_semantics_stage2_epoch11`
- Stage 12 best metrics: `field_exact_match = 0.9298`, `end_to_end_exact_match = 0.6496`

Main conclusion:

- explicit semantic-slot auxiliary supervision is a valid algorithmic direction
- however, on this task it still does not outperform the simpler and stronger `action + component` canonicalization line with staged training
- this makes Stage 12 a useful depth-enhancing exploration result, not a new top-level best system

### Stage 13 External Few-Shot Adaptation

- best run: `qwen25_3b_stage13_ext1024_epoch3_lr1e4`
- field exact match: `0.7512`
- end-to-end exact match: `0.0536`

Main conclusion:

- this is the first branch that materially improves the mapped external-dataset result rather than only improving in-domain performance
- the key gain is restoring schema completeness: the best adaptation run reaches `valid_json = 1.0`, `schema_compliance = 1.0`, and `missing_required_field = 0`
- external-only adaptation works better than mixing in-domain data under the current recipe
- once completeness is fixed, the remaining bottleneck becomes semantic taxonomy alignment, especially `component`, `priority`, `category`, and downstream `action`

### Stage 14 External Targeted Adaptation

- best run: `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- field exact match: `0.7517`
- end-to-end exact match: `0.0636`

Main conclusion:

- this is the strongest mapped external result in the repository so far
- it improves on the Stage 13 best adapter while preserving perfect schema completeness
- targeting all core external taxonomy fields works better than narrower targeted subsets
- a conservative continuation (`1` epoch, `5e-5`) works better than the corresponding `1e-4` variant
- after completeness has been restored by Stage 13, the next useful gains come from narrow semantic-taxonomy continuation rather than broader retraining

### Stage 16 and Stage 17 External Plateau Follow-Ups

- Stage 16 best larger-scale adaptation run:
  - `qwen25_3b_stage16_extfull_epoch2_lr5e5`
  - `field_exact_match = 0.7537`
  - `end_to_end_exact_match = 0.0542`
- Stage 16 best retrieval-guided postprocess:
  - `knn3_priority_majority`
  - `field_exact_match = 0.7501`
  - `end_to_end_exact_match = 0.0554`
- Stage 17 best field-level target redesign:
  - `qwen25_3b_stage17_redesignfull_c80_cat60_epoch2_lr5e5`
  - `field_exact_match = 0.7514`
  - `end_to_end_exact_match = 0.0530`
- Stage 17 best residual curriculum:
  - `qwen25_3b_stage17_residual_component_focused_x2_epoch1_lr5e5`
  - `field_exact_match = 0.7479`
  - `end_to_end_exact_match = 0.0512`

Main conclusion:

- none of these Stage 16/17 follow-ups beat the Stage 14 best external run
- larger-scale adaptation can nudge field-level accuracy slightly, but it does not improve external end-to-end exact match enough to set a new best
- retrieval-guided postprocess only shows a weak signal on `priority`; transferring `category / action / component` labels from nearest neighbors is too noisy
- field-level target redesign and residual curriculum both fail to outperform the simpler Stage 14 all-core continuation recipe
- the external line has therefore entered a plateau: completeness is already solved, and the remaining problem is semantic taxonomy alignment rather than structure

### Stage 18 External Component Verifier

- best verifier result:
  - `qwen25_3b_stage18_guarded_name_majority_p80`
  - `field_exact_match = 0.7517`
  - `end_to_end_exact_match = 0.0636`
- strongest learned verifier variants:
  - `qwen25_3b_stage18_component_nb_text_name_pred`: `0.7510 / 0.0489`
  - `qwen25_3b_stage18_hybrid_guarded_or_nb`: `0.7510 / 0.0489`

Main conclusion:

- the best Stage 18 result only ties the Stage 14 external best and does not exceed it
- guarded high-purity `name -> component` majority mapping makes almost no useful edits under the current external train distribution
- simple NB-style component verifiers degrade external end-to-end exact match rather than improving it
- this is a useful negative result: the remaining external bottleneck cannot be solved by a simple `component`-only verifier layered on top of the Stage 14 predictions

## What Is Still Missing

To reach the originally desired "more complete research project" level, the project still mainly needs:

- final README / summary cleanup for resume and interview use
- optionally, a cleaner constrained-decoding baseline if a better schema-compatible tool is used
- optionally, further target redesign or two-stage supervision for `category` and `priority`
- optionally, tighter deterministic handling for `category` and `priority` if a high-precision rule set can be found
- optionally, a cleaner multi-task or preference-tuning branch if the project is pushed further for research depth rather than immediate benchmark gain

## Current Next Step

Immediate next step:

- finalize top-level project narrative around the Stage 7 joint canonicalization result
- finalize top-level project narrative around the Stage 9 lexical postprocessing gain
- keep additional experimentation narrow unless it clearly improves the final research story

Expected outcome:

- one stable top-level summary of prompt-only, repair, reduced-schema target design, LoRA-rank ablations, epoch and learning-rate ablations, staged training, hard-example negative results, action canonicalization, component follow-ups, deterministic postprocessing, lexical postprocessing, and seen/unseen schema generalization
- one clear statement that broad continuation did not beat the strongest staged baseline, while joint target redesign plus staged training did
- one explicit reference comparison showing that same-family prompt-only scaling to `7B / 14B / 32B` still does not match the post-trained 3B system on this task
- one explicit cross-dataset generalization comparison showing that the trained 3B keeps a field-level advantage over larger prompt-only models, while task-specific postprocess layers fail to transfer
- one explicit external-adaptation result showing that a small amount of mapped external supervision can recover completeness and restore non-zero external end-to-end exact match
- one explicit external plateau result showing that, after completeness is solved, broader scaling and continuation variants do not beat the best narrow all-core external continuation
- one explicit Stage 18 negative result showing that a narrow component-only verifier also fails to beat the best Stage 14 external continuation
- one clear statement of what post-training solves, what repair solves, what deterministic consistency can still clean up cheaply, and what still fails semantically

## Practical Rule

Current development priority:

- prioritize final narrative quality over adding broad new experiment branches
- do not spend major effort on broad hyperparameter tuning unless it directly supports a stronger final conclusion
- keep reusable logic in `src/` and `scripts/`, not notebooks
- use `docs/results/error_analysis_taxonomy.md` as the primary failure-taxonomy reference when preparing resume or interview material
