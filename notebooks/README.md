# Notebooks

Recommended usage order:

- `00_environment_check.ipynb`: verify Python, CUDA, and GPU visibility in Jupyter.
- `01_data_inspection.ipynb`: inspect phase-1 data splits and mapped samples.
- `02_phase1_qlora_training.ipynb`: full-schema QLoRA baseline training and test export.
- `03_error_analysis.ipynb`: compare experiment reports and review current findings.
- `04_prompt_only_baseline.ipynb`: prompt-only baseline inference on the phase-1 test set.
- `05_phase1_qlora_reduced_training.ipynb`: reduced-schema QLoRA baseline training and test export.
- `06_schema_generalization_qlora.ipynb`: schema-conditioned reduced-schema QLoRA training and seen/unseen schema generalization export.
- `07_stage2_experiment_runner.ipynb`: run the next reduced-schema Stage 2 experiments, including data-regime, curriculum, and LoRA-rank ablations.
- `08_stage2_results_review.ipynb`: compare Stage 1 and Stage 2 reports after syncing new outputs back into the repo.
- `10_long_run_ablation_suite.ipynb`: batch-run longer training-time follow-up experiments, including learning-rate ablations and structure-then-semantics two-stage training.
- `14_action_canonicalization_suite.ipynb`: batch-run action-canonicalization target-design experiments with single-stage and staged training variants.
- `15_component_canonicalization_suite.ipynb`: batch-run a smaller component-canonicalization target-design suite, including a component-only control and joint action+component variants.
- `16_deterministic_postprocess_suite.ipynb`: batch-run fast deterministic postprocessing variants on the current Stage 7 best predictions without retraining.
- `17_lexical_postprocess_suite.ipynb`: batch-run fast lexical postprocessing variants on top of the Stage 8 best predictions without retraining.
- `18_big_model_reference_suite.ipynb`: batch-run larger prompt-only reference models on the canonicalized reduced-schema task and export raw / repaired / postprocessed comparisons.
- `19_external_generalization_suite.ipynb`: run cross-dataset generalization references on a new reduced-schema project-format eval set, comparing the strongest trained/system 3B line against larger prompt-only references.
- `23_external_adaptation_suite.ipynb`: build canonicalized external train/val/test splits from the customer-support dataset, then run Stage 7 checkpoint few-shot adaptation presets on the external train split and evaluate on the external test split.
- `24_external_targeted_adaptation_suite.ipynb`: mine the strongest Stage 13 external-adaptation checkpoint on the external train split, then run targeted external continuation presets focused on taxonomy-heavy fields such as component, category, and priority.
- `25_external_overnight_optimization_suite.ipynb`: batch-run a larger overnight external optimization suite that combines deeper targeted continuation on top of the Stage 14 best adapter and external-train-driven deterministic postprocess variants.
- `21_semantic_core_intermediate_suite.ipynb`: batch-run a semantic-core intermediate experiment where the model predicts a separate semantic JSON object and a deterministic renderer converts it back to the reduced schema before evaluation.
- `22_semantic_slot_supervision_suite.ipynb`: batch-run auxiliary semantic-slot supervision experiments where the model predicts both a small semantic slot object and the final JSON, then a deterministic reconcile step fuses them before evaluation.

Retired / historical branches:

- `09_constrained_decoding_eval.ipynb`: kept only as a historical failed branch reference.
- `11_end_to_end_optimization_suite.ipynb`: superseded by later target-design lines.
- `12_targeted_end_to_end_suite.ipynb`: superseded by later target-design lines.
- `13_targeted_end_to_end_refinement_suite.ipynb`: superseded by later target-design lines.
- `20_action_template_latent_suite.ipynb`: kept only as a negative-result reference.

For normal use, focus on:

- `07_stage2_experiment_runner.ipynb`
- `08_stage2_results_review.ipynb`
- `10_long_run_ablation_suite.ipynb`
- `14_action_canonicalization_suite.ipynb`
- `15_component_canonicalization_suite.ipynb`
- `16_deterministic_postprocess_suite.ipynb`
- `17_lexical_postprocess_suite.ipynb`
- `18_big_model_reference_suite.ipynb`
- `19_external_generalization_suite.ipynb`
- `23_external_adaptation_suite.ipynb`
- `24_external_targeted_adaptation_suite.ipynb`
- `25_external_overnight_optimization_suite.ipynb`
- `21_semantic_core_intermediate_suite.ipynb`
- `22_semantic_slot_supervision_suite.ipynb`

Current policy:

- keep training and inference in dedicated notebooks
- keep result comparison in `03_error_analysis.ipynb`
- keep Stage 2 ablation execution in `07_stage2_experiment_runner.ipynb`
- keep Stage 2 comparison and consolidation in `08_stage2_results_review.ipynb`
- keep long-run learning-rate and two-stage training batch execution in `10_long_run_ablation_suite.ipynb`
- keep action-canonicalization target-design experiments in `14_action_canonicalization_suite.ipynb`
- keep component-canonicalization follow-up experiments in `15_component_canonicalization_suite.ipynb`
- keep fast deterministic postprocessing experiments in `16_deterministic_postprocess_suite.ipynb`
- keep fast lexical postprocessing experiments in `17_lexical_postprocess_suite.ipynb`
- keep larger prompt-only reference comparisons in `18_big_model_reference_suite.ipynb`
- keep external-dataset generalization references in `19_external_generalization_suite.ipynb`
- keep true semantic-core intermediate experiments in `21_semantic_core_intermediate_suite.ipynb`
- keep semantic-slot auxiliary supervision experiments in `22_semantic_slot_supervision_suite.ipynb`
- avoid putting reusable preprocessing or evaluation logic into notebooks; keep that in `src/` and `scripts/`
