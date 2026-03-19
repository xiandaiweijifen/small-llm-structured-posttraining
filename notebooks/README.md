# Notebooks

Recommended usage order:

- `00_environment_check.ipynb`: verify Python, CUDA, and GPU visibility in Jupyter.
- `01_data_inspection.ipynb`: inspect phase-1 data splits and mapped samples.
- `02_phase1_qlora_training.ipynb`: full-schema QLoRA baseline training and test export.
- `03_error_analysis.ipynb`: compare experiment reports and review current findings.
- `04_prompt_only_baseline.ipynb`: prompt-only baseline inference on the phase-1 test set.
- `05_phase1_qlora_reduced_training.ipynb`: reduced-schema QLoRA baseline training and test export.

Current policy:

- keep training and inference in dedicated notebooks
- keep result comparison in `03_error_analysis.ipynb`
- avoid putting reusable preprocessing or evaluation logic into notebooks; keep that in `src/` and `scripts/`
