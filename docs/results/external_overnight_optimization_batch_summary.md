# External Overnight Optimization Batch Summary

- baseline checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- source prediction experiment for postprocess: `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- external train: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_train_reduced.jsonl`
- external val: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_val_reduced.jsonl`
- external test: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_test_reduced.jsonl`
- skip completed: `True`
- skip existing mining: `True`
- inference batch size: `28`

## Training Runs

### target_allcore_x2_epoch1_lr5e5

- experiment: `qwen25_3b_stage15_target_allcore_x2_epoch1_lr5e5`
- status: `completed`
- subset: `all_core_fields`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `2`
- raw field exact match: `0.7457`
- raw end-to-end exact match: `0.0459`
- repaired field exact match: `0.7457`
- repaired end-to-end exact match: `0.0459`

### target_allcore_x1_epoch2_lr5e5

- experiment: `qwen25_3b_stage15_target_allcore_x1_epoch2_lr5e5`
- status: `completed`
- subset: `all_core_fields`
- learning rate: `5e-05`
- epochs: `2`
- target repeat: `1`
- raw field exact match: `0.7486`
- raw end-to-end exact match: `0.0477`
- repaired field exact match: `0.7486`
- repaired end-to-end exact match: `0.0477`

### target_allcore_x1_epoch1_lr3e5

- experiment: `qwen25_3b_stage15_target_allcore_x1_epoch1_lr3e5`
- status: `completed`
- subset: `all_core_fields`
- learning rate: `3e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.7467`
- raw end-to-end exact match: `0.0430`
- repaired field exact match: `0.7467`
- repaired end-to-end exact match: `0.0430`

### target_allcore_x2_epoch1_lr3e5

- experiment: `qwen25_3b_stage15_target_allcore_x2_epoch1_lr3e5`
- status: `completed`
- subset: `all_core_fields`
- learning rate: `3e-05`
- epochs: `1`
- target repeat: `2`
- raw field exact match: `0.7444`
- raw end-to-end exact match: `0.0424`
- repaired field exact match: `0.7444`
- repaired end-to-end exact match: `0.0424`

### target_component_category_x2_epoch2_lr5e5

- experiment: `qwen25_3b_stage15_target_component_category_x2_epoch2_lr5e5`
- status: `completed`
- subset: `component_or_category`
- learning rate: `5e-05`
- epochs: `2`
- target repeat: `2`
- raw field exact match: `0.7507`
- raw end-to-end exact match: `0.0436`
- repaired field exact match: `0.7507`
- repaired end-to-end exact match: `0.0436`

### target_action_category_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage15_target_action_category_x1_epoch1_lr5e5`
- status: `completed`
- subset: `action_or_category`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.7495`
- raw end-to-end exact match: `0.0459`
- repaired field exact match: `0.7495`
- repaired end-to-end exact match: `0.0459`

## Postprocess Runs

### action_refresh

- experiment: `qwen25_3b_stage15_extpp_action_refresh`
- status: `completed`
- raw field exact match: `0.7517`
- raw end-to-end exact match: `0.0636`
- repaired field exact match: `0.7517`
- repaired end-to-end exact match: `0.0636`

### component_from_name_strict

- experiment: `qwen25_3b_stage15_extpp_component_from_name_strict`
- status: `completed`
- raw field exact match: `0.7517`
- raw end-to-end exact match: `0.0636`
- repaired field exact match: `0.7517`
- repaired end-to-end exact match: `0.0636`

### component_from_name_majority

- experiment: `qwen25_3b_stage15_extpp_component_from_name_majority`
- status: `completed`
- raw field exact match: `0.7511`
- raw end-to-end exact match: `0.0177`
- repaired field exact match: `0.7511`
- repaired end-to-end exact match: `0.0177`

### action_component_majority

- experiment: `qwen25_3b_stage15_extpp_action_component_majority`
- status: `completed`
- raw field exact match: `0.7511`
- raw end-to-end exact match: `0.0177`
- repaired field exact match: `0.7511`
- repaired end-to-end exact match: `0.0177`

