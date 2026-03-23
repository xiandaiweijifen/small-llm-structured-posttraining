# External Adaptation Batch Summary

- baseline checkpoint: `/home/lyan11/small-llm-structured-posttraining/results/checkpoints/qwen25_3b_stage7_canonical_action_component_structure_then_semantics_stage2_epoch9`
- external train: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_train_reduced.jsonl`
- external val: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_val_reduced.jsonl`
- external test: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_test_reduced.jsonl`
- skip completed: `True`

## Runs

### ext64_epoch3_lr1e4

- experiment: `qwen25_3b_stage13_ext64_epoch3_lr1e4`
- status: `completed`
- train count: `64`
- mix in-domain: `False`
- raw field exact match: `0.6557`
- raw end-to-end exact match: `0.0024`
- repaired field exact match: `0.6557`
- repaired end-to-end exact match: `0.0024`

### ext256_epoch3_lr1e4

- experiment: `qwen25_3b_stage13_ext256_epoch3_lr1e4`
- status: `completed`
- train count: `256`
- mix in-domain: `False`
- raw field exact match: `0.7490`
- raw end-to-end exact match: `0.0312`
- repaired field exact match: `0.7491`
- repaired end-to-end exact match: `0.0312`

### ext512_epoch3_lr1e4

- experiment: `qwen25_3b_stage13_ext512_epoch3_lr1e4`
- status: `completed`
- train count: `512`
- mix in-domain: `False`
- raw field exact match: `0.7536`
- raw end-to-end exact match: `0.0518`
- repaired field exact match: `0.7537`
- repaired end-to-end exact match: `0.0518`

### ext1024_epoch3_lr1e4

- experiment: `qwen25_3b_stage13_ext1024_epoch3_lr1e4`
- status: `completed`
- train count: `1024`
- mix in-domain: `False`
- raw field exact match: `0.7512`
- raw end-to-end exact match: `0.0536`
- repaired field exact match: `0.7512`
- repaired end-to-end exact match: `0.0536`

### ext256_mix_epoch3_lr1e4

- experiment: `qwen25_3b_stage13_ext256_mix_epoch3_lr1e4`
- status: `completed`
- train count: `512`
- mix in-domain: `True`
- raw field exact match: `0.7425`
- raw end-to-end exact match: `0.0159`
- repaired field exact match: `0.7425`
- repaired end-to-end exact match: `0.0159`

### ext512_mix_epoch3_lr1e4

- experiment: `qwen25_3b_stage13_ext512_mix_epoch3_lr1e4`
- status: `completed`
- train count: `1024`
- mix in-domain: `True`
- raw field exact match: `0.7495`
- raw end-to-end exact match: `0.0518`
- repaired field exact match: `0.7495`
- repaired end-to-end exact match: `0.0518`

### ext1024_mix_epoch3_lr1e4

- experiment: `qwen25_3b_stage13_ext1024_mix_epoch3_lr1e4`
- status: `completed`
- train count: `2048`
- mix in-domain: `True`
- raw field exact match: `0.7432`
- raw end-to-end exact match: `0.0442`
- repaired field exact match: `0.7432`
- repaired end-to-end exact match: `0.0442`

