# Phase 1 Task Design

## Recommended Phase 1 Task

Use:

text to structured ticket JSON

Input sources can include:

- email snippets
- issue descriptions
- task handoff messages
- technical notifications

## Why This Task Is Better Than Simpler Extraction

- richer than flat classification
- naturally supports nested objects and arrays
- practical enough to sound credible in a resume
- easy to scale with synthetic data

## Phase 1 Schema Design Principles

The first schema should be:

- complex enough to expose structured-output failure modes
- stable enough for repeatable evaluation
- not so wide that annotation cost explodes

## Recommended Base Schema

```json
{
  "ticket_id": "string|null",
  "summary": "string",
  "category": "bug|feature|question|incident|task",
  "priority": "low|medium|high|urgent",
  "requires_followup": "boolean",
  "reporter": {
    "name": "string|null",
    "team": "string|null"
  },
  "affected_systems": [
    {
      "name": "string",
      "component": "string|null"
    }
  ],
  "actions_requested": [
    {
      "action": "string",
      "owner": "string|null",
      "deadline": "string|null"
    }
  ],
  "constraints": {
    "environment": "prod|staging|dev|null",
    "blocking": "boolean|null"
  }
}
```

## Why This Schema Is Good

- `enum`: `category`, `priority`, `environment`
- `boolean`: `requires_followup`
- `nullable`: multiple fields
- `nested object`: `reporter`, `constraints`
- `array of objects`: `affected_systems`, `actions_requested`
- `free-text field`: `summary`

## Complexity Buckets

Use three schema complexity buckets in evaluation:

1. Simple
   - only flat fields
2. Medium
   - includes nested object and nullable fields
3. Complex
   - includes arrays of objects plus cross-field consistency pressure

## Phase 1 Data Strategy

Start with one schema family, not many unrelated schemas.

Suggested composition:

- 200 to 500 high-quality seed examples
- 1k to 5k synthetic examples derived from templates plus paraphrasing
- explicit complexity bucket labels for each sample

## Seen / Unseen Schema Plan

Do not start with fully open-ended schema generalization.

Phase 1:

- train on one base schema family
- reserve 1 to 2 lightly modified schema variants for held-out evaluation

Example held-out variants:

- add optional field
- rename one nested field
- tighten one enum

This is enough to demonstrate schema generalization thinking without overcomplicating the first milestone.
