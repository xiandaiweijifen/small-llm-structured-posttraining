# vLLM Structured Reference Summary

- target schema: `ticket_schema_v1_reduced`
- guided backend: `None`
- compared tracks: raw vLLM prompt-only, raw+repair, and vLLM JSON-schema structured outputs

| Model | Variant | Field Exact Match | End-to-End Exact Match |
| --- | --- | ---: | ---: |
| Qwen/Qwen2.5-3B-Instruct | raw | 0.5050 | 0.0000 |
| Qwen/Qwen2.5-3B-Instruct | raw_repair | 0.5086 | 0.0000 |
| Qwen/Qwen2.5-3B-Instruct | structured_json | 0.2570 | 0.0000 |
| Qwen/Qwen2.5-7B-Instruct | raw | 0.5000 | 0.0000 |
| Qwen/Qwen2.5-7B-Instruct | raw_repair | 0.5000 | 0.0000 |
| Qwen/Qwen2.5-7B-Instruct | structured_json | 0.3071 | 0.0000 |
| Qwen/Qwen2.5-14B-Instruct | raw | 0.5716 | 0.0000 |
| Qwen/Qwen2.5-14B-Instruct | raw_repair | 0.5716 | 0.0000 |
| Qwen/Qwen2.5-14B-Instruct | structured_json | 0.3049 | 0.0000 |
| Qwen/Qwen2.5-32B-Instruct | raw | 0.5737 | 0.0000 |
| Qwen/Qwen2.5-32B-Instruct | raw_repair | 0.5744 | 0.0000 |
| Qwen/Qwen2.5-32B-Instruct | structured_json | 0.3672 | 0.0000 |
