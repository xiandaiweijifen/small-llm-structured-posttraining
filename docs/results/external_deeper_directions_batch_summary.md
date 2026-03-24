# External Deeper Directions Batch Summary

- stage7 checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
- stage14 checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage14_target_allcore_x1_epoch1_lr5e5`
- external train: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_train_reduced.jsonl`
- external val: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_val_reduced.jsonl`
- external test: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_test_reduced.jsonl`
- skip completed: `True`

## Field-Level Target Redesign Runs

### redesign2048_c80_cat60_epoch3_lr1e4

- experiment: `qwen25_3b_stage17_redesign2048_c80_cat60_epoch3_lr1e4`
- status: `completed`
- train count: `2048`
- component threshold: `0.8`
- category threshold: `0.6`
- redesign summary: `{"input_count": 2048, "component_map_size": 1003, "category_map_size": 359, "change_counts": {"category": 246, "component": 104, "action": 246}}`
- raw field exact match: `0.7502`
- raw end-to-end exact match: `0.0465`
- repaired field exact match: `0.7502`
- repaired end-to-end exact match: `0.0465`

### redesign4096_c80_cat60_epoch2_lr1e4

- experiment: `qwen25_3b_stage17_redesign4096_c80_cat60_epoch2_lr1e4`
- status: `completed`
- train count: `4096`
- component threshold: `0.8`
- category threshold: `0.6`
- redesign summary: `{"input_count": 4096, "component_map_size": 1096, "category_map_size": 246, "change_counts": {"category": 398, "action": 398, "component": 141}}`
- raw field exact match: `0.7490`
- raw end-to-end exact match: `0.0483`
- repaired field exact match: `0.7490`
- repaired end-to-end exact match: `0.0483`

### redesignfull_c80_cat60_epoch2_lr5e5

- experiment: `qwen25_3b_stage17_redesignfull_c80_cat60_epoch2_lr5e5`
- status: `completed`
- train count: `5926`
- component threshold: `0.8`
- category threshold: `0.6`
- redesign summary: `{"input_count": 5926, "component_map_size": 1068, "category_map_size": 172, "change_counts": {"category": 428, "action": 428, "component": 186}}`
- raw field exact match: `0.7514`
- raw end-to-end exact match: `0.0530`
- repaired field exact match: `0.7514`
- repaired end-to-end exact match: `0.0530`

### redesignfull_c90_cat70_epoch2_lr5e5

- experiment: `qwen25_3b_stage17_redesignfull_c90_cat70_epoch2_lr5e5`
- status: `completed`
- train count: `5926`
- component threshold: `0.9`
- category threshold: `0.7`
- redesign summary: `{"input_count": 5926, "component_map_size": 986, "category_map_size": 56, "change_counts": {"category": 92, "action": 92, "component": 7}}`
- raw field exact match: `0.7529`
- raw end-to-end exact match: `0.0512`
- repaired field exact match: `0.7529`
- repaired end-to-end exact match: `0.0512`

## Residual Curriculum Runs

### residual_component_only_x8_epoch1_lr5e5

- experiment: `qwen25_3b_stage17_residual_component_only_x8_epoch1_lr5e5`
- status: `completed`
- subset: `component_only`
- subset size: `295`
- repeat: `8`
- mixed train size: `8286`
- raw field exact match: `0.7437`
- raw end-to-end exact match: `0.0436`
- repaired field exact match: `0.7437`
- repaired end-to-end exact match: `0.0436`

### residual_priority_only_x4_epoch1_lr5e5

- experiment: `qwen25_3b_stage17_residual_priority_only_x4_epoch1_lr5e5`
- status: `completed`
- subset: `priority_only`
- subset size: `940`
- repeat: `4`
- mixed train size: `9686`
- raw field exact match: `0.7449`
- raw end-to-end exact match: `0.0459`
- repaired field exact match: `0.7449`
- repaired end-to-end exact match: `0.0459`

### residual_component_focused_x2_epoch1_lr5e5

- experiment: `qwen25_3b_stage17_residual_component_focused_x2_epoch1_lr5e5`
- status: `completed`
- subset: `component_focused`
- subset size: `1106`
- repeat: `2`
- mixed train size: `8138`
- raw field exact match: `0.7479`
- raw end-to-end exact match: `0.0512`
- repaired field exact match: `0.7479`
- repaired end-to-end exact match: `0.0512`

### residual_component_priority_x2_epoch1_lr5e5

- experiment: `qwen25_3b_stage17_residual_component_priority_x2_epoch1_lr5e5`
- status: `completed`
- subset: `component_priority_residual`
- subset size: `2046`
- repeat: `2`
- mixed train size: `10018`
- raw field exact match: `0.7437`
- raw end-to-end exact match: `0.0377`
- repaired field exact match: `0.7437`
- repaired end-to-end exact match: `0.0377`

### residual_component_only_x8_epoch1_lr3e5

- experiment: `qwen25_3b_stage17_residual_component_only_x8_epoch1_lr3e5`
- status: `completed`
- subset: `component_only`
- subset size: `295`
- repeat: `8`
- mixed train size: `8286`
- raw field exact match: `0.7467`
- raw end-to-end exact match: `0.0412`
- repaired field exact match: `0.7467`
- repaired end-to-end exact match: `0.0412`

