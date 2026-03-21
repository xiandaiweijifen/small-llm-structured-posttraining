# Targeted End-to-End Refinement Batch Summary

Skip completed: `True`
Baseline checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage2_structure_then_semantics_v1`

## Targeted Subsets

### action_only

- num records: `391`
- bucket counts: `{'complex': 15, 'medium': 195, 'simple': 181}`

### action_or_component

- num records: `482`
- bucket counts: `{'complex': 39, 'medium': 242, 'simple': 201}`

## Runs

### refine_action_component_half_epoch1_lr5e5

- experiment: `qwen25_3b_stage5_refine_action_component_half_epoch1_lr5e5`
- status: `completed`
- subset: `action_or_component`
- subset ratio: `0.5`
- learning rate: `5e-05`
- epochs: `1.0`
- raw field exact match: `0.9137`
- raw end-to-end exact match: `0.5433`
- repaired field exact match: `0.9137`
- repaired end-to-end exact match: `0.5433`

### refine_action_component_half_epoch075_lr5e5

- experiment: `qwen25_3b_stage5_refine_action_component_half_epoch075_lr5e5`
- status: `completed`
- subset: `action_or_component`
- subset ratio: `0.5`
- learning rate: `5e-05`
- epochs: `0.75`
- raw field exact match: `0.9209`
- raw end-to-end exact match: `0.5709`
- repaired field exact match: `0.9209`
- repaired end-to-end exact match: `0.5709`

### refine_action_component_full_epoch05_lr5e5

- experiment: `qwen25_3b_stage5_refine_action_component_full_epoch05_lr5e5`
- status: `completed`
- subset: `action_or_component`
- subset ratio: `1.0`
- learning rate: `5e-05`
- epochs: `0.5`
- raw field exact match: `0.9209`
- raw end-to-end exact match: `0.5591`
- repaired field exact match: `0.9209`
- repaired end-to-end exact match: `0.5591`

### refine_action_component_half_epoch1_lr3e5

- experiment: `qwen25_3b_stage5_refine_action_component_half_epoch1_lr3e5`
- status: `completed`
- subset: `action_or_component`
- subset ratio: `0.5`
- learning rate: `3e-05`
- epochs: `1.0`
- raw field exact match: `0.9202`
- raw end-to-end exact match: `0.5669`
- repaired field exact match: `0.9202`
- repaired end-to-end exact match: `0.5669`

### refine_action_half_epoch075_lr5e5

- experiment: `qwen25_3b_stage5_refine_action_half_epoch075_lr5e5`
- status: `completed`
- subset: `action_only`
- subset ratio: `0.5`
- learning rate: `5e-05`
- epochs: `0.75`
- raw field exact match: `0.9120`
- raw end-to-end exact match: `0.5315`
- repaired field exact match: `0.9120`
- repaired end-to-end exact match: `0.5315`

