"""Recompute complexity buckets for an existing project-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.complexity import relabel_record_complexity
from src.data.io import dump_jsonl, load_jsonl
from src.data.profiling import profile_dataset
from src.evaluation.reporting import write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute complexity buckets for a dataset.")
    parser.add_argument("--input", required=True, help="Path to input dataset jsonl.")
    parser.add_argument("--output", required=True, help="Path to relabeled dataset jsonl.")
    parser.add_argument(
        "--profile-output",
        required=True,
        help="Path to output profile summary json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    relabeled_records = [relabel_record_complexity(record) for record in records]
    dump_jsonl(args.output, relabeled_records)
    write_json_report(args.profile_output, profile_dataset(relabeled_records))
    print(f"Relabeled dataset written to {args.output}")
    print(f"Profile written to {args.profile_output}")


if __name__ == "__main__":
    main()
