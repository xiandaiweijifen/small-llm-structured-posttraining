"""Build train/val/test splits from phase-1 source samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import (
    build_dataset,
    summarize_split_counts,
    write_dataset_splits,
)
from src.data.io import load_jsonl
from src.evaluation.reporting import write_json_report
from src.schemas.ticket_schema import TICKET_SCHEMA
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase-1 dataset splits.")
    parser.add_argument("--input", required=True, help="Path to source samples jsonl.")
    parser.add_argument(
        "--config",
        default="configs/dataset/phase1_ticket.yaml",
        help="Path to dataset config yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/samples",
        help="Directory to write phase-1 train/val/test splits.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Shuffle seed for split assignment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    records = load_jsonl(args.input)

    split_records = build_dataset(
        records=records,
        schema=TICKET_SCHEMA,
        task_name=config["task_name"],
        schema_name=config["schema_name"],
        split_config=config["splits"],
        shuffle_seed=args.shuffle_seed,
    )
    write_dataset_splits(args.output_dir, split_records)

    summary = summarize_split_counts(split_records)
    report_path = f"{args.output_dir}/phase1_split_summary.json"
    write_json_report(report_path, summary)
    print(f"Dataset splits written to {args.output_dir}")
    print(f"Split summary written to {report_path}")
    empty_splits = [
        split_name
        for split_name, split_summary in summary.items()
        if split_summary["num_samples"] == 0
    ]
    if empty_splits:
        print(
            "Warning: empty splits detected: "
            + ", ".join(empty_splits)
            + ". Add more samples or pre-assign metadata.split for tiny datasets."
        )


if __name__ == "__main__":
    main()
