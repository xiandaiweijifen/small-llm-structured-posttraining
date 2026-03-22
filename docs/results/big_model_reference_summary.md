# Big Model Reference Summary

- target schema: `ticket_schema_v1_reduced`
- evaluation target: canonicalized reduced-schema target with canonical action and majority-mapped component
- tracks: raw prompt-only, repair, deterministic postprocess, lexical postprocess

## Raw Prompt-Only

### Qwen/Qwen2.5-3B-Instruct

- experiment: `qwen25_3b_reference_canonical_prompt`
- field exact match: `0.4470`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-7B-Instruct

- experiment: `qwen25_7b_reference_canonical_prompt`
- field exact match: `0.5251`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-14B-Instruct

- experiment: `qwen25_14b_reference_canonical_prompt`
- field exact match: `0.5777`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-32B-Instruct

- experiment: `qwen25_32b_reference_canonical_prompt`
- field exact match: `0.5634`
- end-to-end exact match: `0.0000`

## Prompt-Only + Repair

### Qwen/Qwen2.5-3B-Instruct

- experiment: `qwen25_3b_reference_canonical_prompt`
- field exact match: `0.4982`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-7B-Instruct

- experiment: `qwen25_7b_reference_canonical_prompt`
- field exact match: `0.5251`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-14B-Instruct

- experiment: `qwen25_14b_reference_canonical_prompt`
- field exact match: `0.5795`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-32B-Instruct

- experiment: `qwen25_32b_reference_canonical_prompt`
- field exact match: `0.5659`
- end-to-end exact match: `0.0000`

## Prompt-Only + Deterministic Postprocess

### Qwen/Qwen2.5-3B-Instruct

- experiment: `qwen25_3b_reference_canonical_prompt`
- field exact match: `0.4631`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-7B-Instruct

- experiment: `qwen25_7b_reference_canonical_prompt`
- field exact match: `0.5691`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-14B-Instruct

- experiment: `qwen25_14b_reference_canonical_prompt`
- field exact match: `0.6310`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-32B-Instruct

- experiment: `qwen25_32b_reference_canonical_prompt`
- field exact match: `0.6016`
- end-to-end exact match: `0.0000`

## Prompt-Only + Lexical Postprocess

### Qwen/Qwen2.5-3B-Instruct

- experiment: `qwen25_3b_reference_canonical_prompt`
- field exact match: `0.4696`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-7B-Instruct

- experiment: `qwen25_7b_reference_canonical_prompt`
- field exact match: `0.5744`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-14B-Instruct

- experiment: `qwen25_14b_reference_canonical_prompt`
- field exact match: `0.6382`
- end-to-end exact match: `0.0000`

### Qwen/Qwen2.5-32B-Instruct

- experiment: `qwen25_32b_reference_canonical_prompt`
- field exact match: `0.6092`
- end-to-end exact match: `0.0000`
