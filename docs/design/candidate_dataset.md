# Candidate Dataset

## Goal

Build a phase-1 candidate dataset from mapped external data before final train/val/test splitting.

## Principle

Prefer cleaner weakly supervised samples over maximum volume.

Phase 1 needs:

- enough size to train a small model
- reasonable category coverage
- usable simple / medium / complex buckets
- interpretable filtering rules

## Filtering Strategy

Use conservative rules first.

Keep a sample if:

- `summary` is non-trivial
- `input_text` is not too short and not excessively long
- exactly one affected system exists
- exactly one action exists
- `category` is in the project schema
- `priority` is in the project schema
- `reporter.name` exists

Additional source-specific control:

- keep all valid `Console-AI` samples first because quality is relatively stable
- subsample `KameronB` to reduce category skew, especially excessive `incident`

## Phase 1 Candidate Target

Initial target:

- around `2k` to `4k` candidate samples
- mixed from both sources
- all three complexity buckets represented

## Why Not Use All 27k+ Samples Immediately

- category distribution is skewed
- action fields are heuristic
- early experiments benefit from faster iteration
- smaller curated data is easier to inspect

## Output Files

- `data/samples/phase1_candidate.jsonl`
- `data/samples/phase1_candidate_profile.json`
- `data/samples/phase1_candidate_build_summary.json`
