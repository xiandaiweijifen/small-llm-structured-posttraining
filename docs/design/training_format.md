# Training Format

## Goal

Convert the project dataset into a stable instruction-tuning format before moving training workloads to Jupyter.

## Phase 1 Training Example

Each training sample should contain:

```json
{
  "sample_id": "string",
  "messages": [
    {
      "role": "system",
      "content": "You are an information extraction model. Return only JSON that matches the provided schema."
    },
    {
      "role": "user",
      "content": "Task: extract a structured ticket from the input text.\nSchema name: ticket_schema_v1\nInput text:\n..."
    },
    {
      "role": "assistant",
      "content": "{...gold json...}"
    }
  ],
  "metadata": {
    "task_name": "ticket_structured_output",
    "schema_name": "ticket_schema_v1",
    "complexity_bucket": "simple|medium|complex"
  }
}
```

## Design Choice

Use chat-style records instead of plain prompt/completion pairs.

Reason:

- easier to reuse across SFT frameworks
- closer to current instruction-tuning conventions
- easier to keep schema instructions explicit

## Phase 1 Prompt Template

System message:

- describe the model role
- require JSON-only output
- forbid extra explanation

User message:

- mention task name
- mention schema name
- include schema definition for schema-generalization runs
- include input text
- optionally include a short schema summary

Assistant message:

- gold JSON string

## Recommended Generalization Setting

For seen/unseen schema experiments, prefer:

- prompt includes full JSON schema definition
- train on the base schema
- evaluate on both base and lightly modified schema variants

This avoids making generalization depend only on schema names.

## Output File

- `data/processed/phase1_sft_<split>.jsonl`

## Note

The training formatter should not inject framework-specific fields yet.
Keep it generic and lossless so later notebooks can adapt it to different trainers.
