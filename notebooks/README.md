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

Current policy:

- keep training and inference in dedicated notebooks
- keep result comparison in `03_error_analysis.ipynb`
- keep Stage 2 ablation execution in `07_stage2_experiment_runner.ipynb`
- keep Stage 2 comparison and consolidation in `08_stage2_results_review.ipynb`
- avoid putting reusable preprocessing or evaluation logic into notebooks; keep that in `src/` and `scripts/`
