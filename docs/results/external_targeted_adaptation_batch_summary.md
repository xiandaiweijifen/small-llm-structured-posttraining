# External Targeted Adaptation Batch Summary

- baseline checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage13_ext1024_epoch3_lr1e4`
- external train: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_train_reduced.jsonl`
- external val: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_val_reduced.jsonl`
- external test: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_test_reduced.jsonl`
- skip completed: `True`
- skip existing mining: `True`

## Runs

### target_component_category_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage14_target_component_category_x1_epoch1_lr5e5`
- status: `completed`
- subset: `component_or_category`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.7471`
- raw end-to-end exact match: `0.0483`
- repaired field exact match: `0.7471`
- repaired end-to-end exact match: `0.0483`

### target_component_category_x2_epoch1_lr5e5

- experiment: `qwen25_3b_stage14_target_component_category_x2_epoch1_lr5e5`
- status: `completed`
- subset: `component_or_category`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `2`
- raw field exact match: `0.7482`
- raw end-to-end exact match: `0.0589`
- repaired field exact match: `0.7482`
- repaired end-to-end exact match: `0.0589`

### target_category_priority_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage14_target_category_priority_x1_epoch1_lr5e5`
- status: `completed`
- subset: `category_or_priority`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.7477`
- raw end-to-end exact match: `0.0489`
- repaired field exact match: `0.7477`
- repaired end-to-end exact match: `0.0489`

### target_allcore_x1_epoch1_lr5e5

- experiment: `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- status: `completed`
- subset: `all_core_fields`
- learning rate: `5e-05`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.7517`
- raw end-to-end exact match: `0.0636`
- repaired field exact match: `0.7517`
- repaired end-to-end exact match: `0.0636`

### target_allcore_x1_epoch1_lr1e4

- experiment: `qwen25_3b_stage14_target_allcore_x1_epoch1_lr1e4`
- status: `completed`
- subset: `all_core_fields`
- learning rate: `0.0001`
- epochs: `1`
- target repeat: `1`
- raw field exact match: `0.7483`
- raw end-to-end exact match: `0.0430`
- repaired field exact match: `0.7483`
- repaired end-to-end exact match: `0.0430`

