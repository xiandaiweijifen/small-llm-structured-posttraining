# Targeted End-to-End Optimization Batch Summary

Skip completed: `True`
Baseline checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage2_structure_then_semantics_v1`

## Targeted Subsets

### action_only

- num records: `391`
- bucket counts: `{'complex': 15, 'medium': 195, 'simple': 181}`

### multi_error

- num records: `232`
- bucket counts: `{'complex': 11, 'medium': 128, 'simple': 93}`

### action_or_component

- num records: `482`
- bucket counts: `{'complex': 39, 'medium': 242, 'simple': 201}`

### action_or_label

- num records: `475`
- bucket counts: `{'complex': 21, 'medium': 247, 'simple': 207}`

## Runs

### target_action_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage4_target_action_x1_epoch1_lr5e5`
- status: `completed`
- subset: `action_only`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.9177`
- raw end-to-end exact match: `0.5472`
- repaired field exact match: `0.9177`
- repaired end-to-end exact match: `0.5472`

### target_action_x1_epoch1_lr1e4

- experiment: `qwen25_3b_stage4_target_action_x1_epoch1_lr1e4`
- status: `completed`
- subset: `action_only`
- learning rate: `0.0001`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.9105`
- raw end-to-end exact match: `0.5315`
- repaired field exact match: `0.9105`
- repaired end-to-end exact match: `0.5315`

### target_multierror_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage4_target_multierror_x1_epoch1_lr5e5`
- status: `completed`
- subset: `multi_error`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.9159`
- raw end-to-end exact match: `0.5433`
- repaired field exact match: `0.9159`
- repaired end-to-end exact match: `0.5433`

### target_action_component_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage4_target_action_component_x1_epoch1_lr5e5`
- status: `completed`
- subset: `action_or_component`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.9216`
- raw end-to-end exact match: `0.5669`
- repaired field exact match: `0.9216`
- repaired end-to-end exact match: `0.5669`

### target_action_label_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage4_target_action_label_x1_epoch1_lr5e5`
- status: `completed`
- subset: `action_or_label`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.9184`
- raw end-to-end exact match: `0.5472`
- repaired field exact match: `0.9184`
- repaired end-to-end exact match: `0.5472`

