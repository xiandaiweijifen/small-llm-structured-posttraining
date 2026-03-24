# External Frontier Batch Summary

- baseline checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
- source prediction experiment for postprocess: `qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- external train: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_train_reduced.jsonl`
- external val: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_val_reduced.jsonl`
- external test: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_test_reduced.jsonl`
- skip completed: `True`
- inference batch size: `32`

## Large-Scale External Adaptation Runs

### ext2048_epoch3_lr1e4

- experiment: `qwen25_3b_stage16_ext2048_epoch3_lr1e4`
- status: `completed`
- train count: `2048`
- learning rate: `0.0001`
- epochs: `3`
- raw field exact match: `0.7497`
- raw end-to-end exact match: `0.0489`
- repaired field exact match: `0.7497`
- repaired end-to-end exact match: `0.0489`

### ext4096_epoch2_lr1e4

- experiment: `qwen25_3b_stage16_ext4096_epoch2_lr1e4`
- status: `completed`
- train count: `4096`
- learning rate: `0.0001`
- epochs: `2`
- raw field exact match: `0.7471`
- raw end-to-end exact match: `0.0471`
- repaired field exact match: `0.7471`
- repaired end-to-end exact match: `0.0471`

### extfull_epoch2_lr1e4

- experiment: `qwen25_3b_stage16_extfull_epoch2_lr1e4`
- status: `completed`
- train count: `5926`
- learning rate: `0.0001`
- epochs: `2`
- raw field exact match: `0.7480`
- raw end-to-end exact match: `0.0489`
- repaired field exact match: `0.7480`
- repaired end-to-end exact match: `0.0489`

### ext2048_epoch3_lr5e5

- experiment: `qwen25_3b_stage16_ext2048_epoch3_lr5e5`
- status: `completed`
- train count: `2048`
- learning rate: `5e-05`
- epochs: `3`
- raw field exact match: `0.7527`
- raw end-to-end exact match: `0.0489`
- repaired field exact match: `0.7527`
- repaired end-to-end exact match: `0.0489`

### ext4096_epoch2_lr5e5

- experiment: `qwen25_3b_stage16_ext4096_epoch2_lr5e5`
- status: `completed`
- train count: `4096`
- learning rate: `5e-05`
- epochs: `2`
- raw field exact match: `0.7524`
- raw end-to-end exact match: `0.0459`
- repaired field exact match: `0.7524`
- repaired end-to-end exact match: `0.0459`

### extfull_epoch2_lr5e5

- experiment: `qwen25_3b_stage16_extfull_epoch2_lr5e5`
- status: `completed`
- train count: `5926`
- learning rate: `5e-05`
- epochs: `2`
- raw field exact match: `0.7537`
- raw end-to-end exact match: `0.0542`
- repaired field exact match: `0.7537`
- repaired end-to-end exact match: `0.0542`

## Retrieval-Guided Postprocess Runs

### knn1_category_action

- experiment: `qwen25_3b_stage16_knn1_category_action`
- status: `completed`
- k: `1`
- fields: `category,action`
- raw field exact match: `0.7428`
- raw end-to-end exact match: `0.0230`
- repaired field exact match: `0.7428`
- repaired end-to-end exact match: `0.0230`

### knn3_category_action_majority

- experiment: `qwen25_3b_stage16_knn3_category_action_majority`
- status: `completed`
- k: `3`
- fields: `category,action`
- raw field exact match: `0.7461`
- raw end-to-end exact match: `0.0294`
- repaired field exact match: `0.7461`
- repaired end-to-end exact match: `0.0294`

### knn3_priority_majority

- experiment: `qwen25_3b_stage16_knn3_priority_majority`
- status: `completed`
- k: `3`
- fields: `priority`
- raw field exact match: `0.7501`
- raw end-to-end exact match: `0.0554`
- repaired field exact match: `0.7501`
- repaired end-to-end exact match: `0.0554`

### knn5_allcore_majority

- experiment: `qwen25_3b_stage16_knn5_allcore_majority`
- status: `completed`
- k: `5`
- fields: `category,priority,component,action`
- raw field exact match: `0.7492`
- raw end-to-end exact match: `0.0371`
- repaired field exact match: `0.7492`
- repaired end-to-end exact match: `0.0371`

### knn5_component_only

- experiment: `qwen25_3b_stage16_knn5_component_only`
- status: `completed`
- k: `5`
- fields: `component`
- raw field exact match: `0.7514`
- raw end-to-end exact match: `0.0477`
- repaired field exact match: `0.7514`
- repaired end-to-end exact match: `0.0477`

