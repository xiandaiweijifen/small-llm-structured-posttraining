# Data Sources

## Recommendation

Use a hybrid data strategy:

- external dataset as the starting point
- project-specific schema remapping
- synthetic expansion after the base pipeline is stable

## Selected External Sources

### `KameronB/synthetic-it-callcenter-tickets`

Use as:

- medium-scale external source
- good for category, routing, priority, and task-style fields

Local target directory:

- `data/raw/external/kameronb_it_callcenter_tickets/`

### `Console-AI/IT-helpdesk-synthetic-tickets`

Use as:

- smaller cleaner seed source
- better for early manual inspection and schema design checks

Local target directory:

- `data/raw/external/console_ai_it_helpdesk_tickets/`

## Direct Load Vs Local Save

### Option A: Direct load only

```python
from datasets import load_dataset

ds = load_dataset("KameronB/synthetic-it-callcenter-tickets")
```

This already gives you a usable Hugging Face `DatasetDict` or `Dataset`.

Good for:

- quick inspection
- one-off exploration

Weakness:

- relies on local Hugging Face cache
- less explicit for project reproducibility

### Option B: Load and then save locally

Recommended for this project.

```python
from datasets import load_dataset

ds = load_dataset("KameronB/synthetic-it-callcenter-tickets")
ds.save_to_disk("data/raw/external/kameronb_it_callcenter_tickets/hf_saved")
```

Or export to JSONL / CSV after loading.

Good for:

- reproducibility
- explicit project data lineage
- easier later preprocessing

## Recommended Practice For This Project

When you go to Jupyter:

1. load the dataset from Hugging Face
2. inspect columns and split names
3. save a local copy under `data/raw/external/...`
4. export a normalized subset into `data/raw/exports/`
5. build our project dataset from that exported copy

## Why Not Rely Only On Cache

Hugging Face cache works, but it is not the cleanest project contract.

For a resume project, it is better to keep:

- a documented source
- a fixed local storage path
- an explicit preprocessing script

## Current Rule

- `data/raw/` stays gitignored
- processed project-ready samples go to `data/samples/` or `data/processed/`
