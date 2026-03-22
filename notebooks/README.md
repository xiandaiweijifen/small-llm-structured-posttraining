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
- `09_constrained_decoding_eval.ipynb`: evaluate a trained reduced-schema checkpoint with schema-constrained decoding and export reports in the standard format.
- `10_long_run_ablation_suite.ipynb`: batch-run longer training-time follow-up experiments, including learning-rate ablations and structure-then-semantics two-stage training.
- `11_end_to_end_optimization_suite.ipynb`: batch-run broader end-to-end optimization experiments, including hard-sample mining and staged-checkpoint continuation presets.
- `12_targeted_end_to_end_suite.ipynb`: batch-run narrower targeted continuation experiments focused on `action`, multi-error, and related semantic subsets.
- `13_targeted_end_to_end_refinement_suite.ipynb`: batch-run lighter refinement experiments around the most promising `action_or_component` continuation direction.
- `14_action_canonicalization_suite.ipynb`: batch-run action-canonicalization target-design experiments with single-stage and staged training variants.
- `15_component_canonicalization_suite.ipynb`: batch-run a smaller component-canonicalization target-design suite, including a component-only control and joint action+component variants.
- `16_deterministic_postprocess_suite.ipynb`: batch-run fast deterministic postprocessing variants on the current Stage 7 best predictions without retraining.

Current policy:

- keep training and inference in dedicated notebooks
- keep result comparison in `03_error_analysis.ipynb`
- keep Stage 2 ablation execution in `07_stage2_experiment_runner.ipynb`
- keep Stage 2 comparison and consolidation in `08_stage2_results_review.ipynb`
- keep decode-side constrained evaluation in `09_constrained_decoding_eval.ipynb`
- keep long-run learning-rate and two-stage training batch execution in `10_long_run_ablation_suite.ipynb`
- keep broader end-to-end optimization batch execution in `11_end_to_end_optimization_suite.ipynb`
- keep narrower targeted continuation batch execution in `12_targeted_end_to_end_suite.ipynb`
- keep lighter targeted refinement batch execution in `13_targeted_end_to_end_refinement_suite.ipynb`
- keep action-canonicalization target-design experiments in `14_action_canonicalization_suite.ipynb`
- keep component-canonicalization follow-up experiments in `15_component_canonicalization_suite.ipynb`
- keep fast deterministic postprocessing experiments in `16_deterministic_postprocess_suite.ipynb`
- avoid putting reusable preprocessing or evaluation logic into notebooks; keep that in `src/` and `scripts/`
