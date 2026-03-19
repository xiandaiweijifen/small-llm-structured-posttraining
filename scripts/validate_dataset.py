"""Validate a project-format dataset jsonl file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import validate_records
from src.data.io import load_jsonl
from src.schemas.registry import get_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a project-format dataset file.")
    parser.add_argument("--input", required=True, help="Path to dataset jsonl.")
    parser.add_argument(
        "--task-name",
        default="ticket_structured_output",
        help="Expected task_name.",
    )
    parser.add_argument(
        "--schema-name",
        default="ticket_schema_v1",
        help="Expected schema_name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    schema = get_schema(args.schema_name)
    validate_records(
        records,
        schema=schema,
        task_name=args.task_name,
        schema_name=args.schema_name,
    )
    print(f"Validation passed for {args.input} ({len(records)} records)")


if __name__ == "__main__":
    main()
