# External Generalization Reference Summary

- input dataset: `/home/lyan11/small-llm-structured-posttraining/data/external_generalization/gorkemsevinc_customer_support_tickets_eval_reduced.jsonl`
- dataset tag: `gorkemsevinc_cst_eval`
- target schema: `ticket_schema_v1_reduced`
- evaluation target: canonicalized reduced-schema target with canonical action and majority-mapped component
- compared tracks: strongest trained 3B, strongest 3B system line, 14B prompt-only, and 32B prompt-only references

## Leaderboard

### trained_stage7_raw

- experiment: `gorkemsevinc_cst_eval_trained_stage7_raw`
- field exact match: `0.6040`
- end-to-end exact match: `0.0000`

### trained_stage8_deterministic

- experiment: `gorkemsevinc_cst_eval_trained_stage8_deterministic`
- field exact match: `0.6030`
- end-to-end exact match: `0.0000`

### trained_stage9_lexical

- experiment: `gorkemsevinc_cst_eval_trained_stage9_lexical`
- field exact match: `0.6030`
- end-to-end exact match: `0.0000`

### prompt_14b_raw

- experiment: `gorkemsevinc_cst_eval_prompt_14b_raw`
- field exact match: `0.4201`
- end-to-end exact match: `0.0000`

### prompt_14b_repair

- experiment: `gorkemsevinc_cst_eval_prompt_14b_repair`
- field exact match: `0.4215`
- end-to-end exact match: `0.0000`

### prompt_14b_lexical

- experiment: `gorkemsevinc_cst_eval_prompt_14b_lexical`
- field exact match: `0.4201`
- end-to-end exact match: `0.0000`

### prompt_32b_raw

- experiment: `gorkemsevinc_cst_eval_prompt_32b_raw`
- field exact match: `0.4212`
- end-to-end exact match: `0.0000`

### prompt_32b_repair

- experiment: `gorkemsevinc_cst_eval_prompt_32b_repair`
- field exact match: `0.4466`
- end-to-end exact match: `0.0000`

