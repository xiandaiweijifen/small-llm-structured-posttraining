# Training Runbook

## Goal

Provide the minimum steps required to move from the local project state to the first Jupyter training run.

## Current Ready Inputs

- training data: `data/processed/phase1_sft_train.jsonl`
- validation data: `data/processed/phase1_sft_val.jsonl`
- test data: `data/samples/phase1_test.jsonl`
- train config: `configs/train/lora_phase1.yaml`
- eval config: `configs/eval/base_eval.yaml`

## Recommended First Run

Use:

- model: `Qwen/Qwen2.5-3B-Instruct`
- method: `QLoRA`
- data: phase-1 SFT train and val
- H200 recommendation: start with `per_device_train_batch_size=8` and `gradient_accumulation_steps=4`

## Jupyter Checklist

1. confirm the notebook environment can see the current project files
2. install required training libraries
3. verify GPU type and memory
4. set `base_model` in the train config
5. on H200-class GPUs, use the larger configured batch size first
6. run the first QLoRA baseline
6. save predictions on the phase-1 test set
7. bring predictions back into the local eval pipeline if needed

## Suggested First Environment Check

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
```

## Suggested First Dependencies

```bash
pip install transformers datasets peft accelerate trl bitsandbytes pyyaml jsonschema
```

## First Config To Use

Start from:

- `configs/train/lora_phase1_qwen3b.yaml`

Fallback:

- `configs/train/lora_phase1_smollm2_1p7b.yaml`

Reduced-schema follow-up:

- `configs/train/lora_phase1_qwen3b_reduced.yaml`

## Output Convention

Recommended experiment names:

- `qwen25_3b_phase1_qlora_v1`
- `smollm2_1p7b_phase1_qlora_v1`

Recommended outputs:

- checkpoints under `results/checkpoints/`
- predictions under `results/predictions/`

## When To Stop And Report Back

Report back after:

- model loads successfully
- one short pilot epoch or partial training run finishes
- you have one prediction file on the test set

At that point, the local repository already has the validation and evaluation tools needed for the next step.
